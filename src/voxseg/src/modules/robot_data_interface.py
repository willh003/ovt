import rospy
from rospy import Publisher
from rospy import Subscriber as RospySubscriber

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped 
from std_msgs.msg import Int32, Float32
from voxseg.msg import DepthImageInfo, Classes, ImageArray
from voxseg.srv import ImageSeg

from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber as SyncedSubscriber

from contextlib import contextmanager
import numpy as np
import time

from modules.utils import *
from modules.real_data_cfg import IMAGE_TOPIC
from modules.voxseg_root_dir import VOXSEG_ROOT_DIR
from modules.ovseg.open_vocab_seg.ws_ovseg_model import WSImageEncoder
from modules.ovseg.playground import get_turbo_image

# Method cutout from Wild Visual Navigation--------------------#
from typing import List, Sequence

def request_timer(callback_func):
    def wrapper(*args, **kwargs):
        current_time = time.time()
        
        if not hasattr(callback_func, 'last_call_time'):
            callback_func.last_call_time = current_time
        else:
            elapsed_time = current_time - callback_func.last_call_time
            print(f"Time since last callback: {elapsed_time:.2f} seconds")
            callback_func.last_call_time = current_time
        
        return callback_func(*args, **kwargs)
    
    return wrapper    

class OVTDataInterface:
    def __init__(self):
        """
        For now, you will only be able to define exactly three classes
        This is because each needs its own layer in elevation_mapping_cupy, which has a separate yaml
        """


        self.data_interface_node = rospy.get_param('/ovt/DATA_INTERFACE_NODE')
        self.class_topic = rospy.get_param('/ovt/CLASS_TOPIC')
        self.ovt_request_service = rospy.get_param('/ovt/REQUEST_SERVICE')
        self.batch_size = rospy.get_param('/ovt/BATCH_SIZE')
        self.change_rate_topic = rospy.get_param('/ovt/CHANGE_RATE_TOPIC')


        self.bridge = CvBridge()

        self.tick = 0
        self.rate = rospy.get_param('/ovt/COMPUTE_PERIOD')
        self.classes = list(rospy.get_param('/ovt/CLASSES'))
        self.base_name = rospy.get_param('/ovt/BASE_NAME')
        self.use_large = rospy.get_param('/ovt/USE_LARGE')

        if self.use_large:
            config='configs/ovt.yaml'
        else:
            config='configs/ovt_small.yaml'
        self.encoder =  WSImageEncoder(root_dir=VOXSEG_ROOT_DIR, config=config, use_large=self.use_large)
        
        self.device = rospy.get_param('ovt/DEVICE')
        

        rospy.init_node(self.data_interface_node)

        # User Input Subscriptions
        RospySubscriber(self.change_rate_topic, Float32, callback=self.change_rate)
        RospySubscriber(self.class_topic, Classes, self.class_name_callback)

        # Elevation Mapping Plugin Publishers
        self.define_pub_register()

        self.mask_pub = Publisher('/ovt/masked_image', RosImage, queue_size=10)

        # Robot Input Subscriptions
        robot_image_topic = rospy.get_param('/ovt/ROBOT_IMAGE_TOPIC')
        RospySubscriber(robot_image_topic, CompressedImage, callback=self.image_callback,queue_size=1)
    
        print('OVT Interface Initialized')
        self.time_request = time.time()
        rospy.spin()

    def define_pub_register(self):
        prob_publisher_topics = list(rospy.get_param('/ovt/PROB_TOPICS'))
        self.class_pub_register = {}
        for i, topic in enumerate(prob_publisher_topics):
            pub = Publisher(topic, RosImage, queue_size=10)
            self.class_pub_register[self.classes[i]] = pub

    def _handle_compute_request(self, images, classes):

        """
        assumes only 1 item in images
        """
        bridge = CvBridge()
        
        # Update from the most recent tensors 
        images_msg = list(images)
        classes = [str(c) for c in classes]
        images = torch_from_img_array_msg(images_msg).float().to(self.device)
        t1  = rospy.get_time() # replace with torch events
        
        # perform computation
        class_probs = self.encoder.call_with_classes(images, classes, use_adapter=True)        
        classifications = torch.argmax(class_probs[0], dim=0)
        classifications = classifications / classifications.max()
        
        # get and publish masks
        masked_overlay, mask, image, cv2_overlay = get_turbo_image(images[0], classifications)
        mask_msg = bridge.cv2_to_imgmsg(cv2_overlay, header=images_msg[0].header)
        self.mask_pub.publish(mask_msg)
        print(f"CLIP Inference and ROS Serialization Time: {rospy.get_time() - t1}")

        # Save images
        if rospy.get_param('/ovt/SAVE_IMAGES'):
            base_path = os.path.join(VOXSEG_ROOT_DIR, 'output')
            if not os.path.exists(base_path):
                os.mkdir(base_path)
            time = rospy.get_time()
        
            masked_overlay.save(os.path.join(base_path, f'{self.base_name}_mask_{time}.jpg'))
            image.save(os.path.join(base_path, f'{self.base_name}_image_{time}.jpg'))

        # publish probabilities
        all_probs_msg = []
        for i, multi_channel_probs in enumerate(class_probs):
            c, _, _ = multi_channel_probs.size()

            corresponding_image_msg = images_msg[i]
            separate_channel_probs = []
            for j in range(c):
                channel_probs = multi_channel_probs[j]
                probs_img_msg = bridge.cv2_to_imgmsg(channel_probs.numpy(), header=corresponding_image_msg.header)
                
                separate_channel_probs.append(probs_img_msg)

            probs_msg = ImageArray(images = separate_channel_probs)
            all_probs_msg.append(probs_msg)

        return all_probs_msg
        
    def publish_probs_and_tfs(self, pixel_probs_list):
        """
        Inputs:
            pixel_probs: A list of length n containing ImageArray[c] objects, where c is the number of classes

        Publishes a MaskAndTF msg with self.mask_tf_pub for each class, for each image in the buffer
        
        """
        for pixel_probs_msg in pixel_probs_list:
            pixel_probs = pixel_probs_msg.images
            for i, prob_image_msg in enumerate(list(pixel_probs)):
                current_class = self.classes[i]
                pub = self.class_pub_register[current_class]
                pub.publish(prob_image_msg)
        
    def image_callback(self,
                 input_rgb_image: CompressedImage):
        self.tick += 1
        if self.tick % self.rate != 0:
            return
        
        self.compute_and_publish(input_rgb_image)
    
    @request_timer
    def compute_and_publish(self,
                 input_rgb_image: CompressedImage):
        image_list = [input_rgb_image]
        
        all_probs_msg = self._handle_compute_request(image_list, self.classes)
        self.publish_probs_and_tfs(all_probs_msg)
    
    def class_name_callback(self, msg):
        classes = list(msg.classes)
        self.classes=classes
        self.define_pub_register()
        print(f'Classes recieved: {classes}')

    def change_rate(self, msg):
        new_rate = float(msg.data)
        self.rate = new_rate

