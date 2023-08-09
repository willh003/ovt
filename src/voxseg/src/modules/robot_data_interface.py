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

from modules.utils import *
from modules.real_data_cfg import IMAGE_TOPIC
from modules.voxseg_root_dir import VOXSEG_ROOT_DIR
from modules.ovseg.open_vocab_seg.ws_ovseg_model import WSImageEncoder

# Method cutout from Wild Visual Navigation--------------------#
from typing import List, Sequence
from liegroups import SE3, SO3
def transformation_matrix_of_pose(pose : Sequence[float]):
    """Convert a translation and rotation into a 4x4 transformation matrix.
 
    Args:
        pose (float subscriptable @ 0..6): A 7 element array representing the pose [tx,ty,tz,q0,qx,qy,qz,qw].

    Returns:
        4x4 Transformation Matrix \in SE3 following the ordering convention specified"""
    quat = np.array(pose[3:])
    quat = quat / np.linalg.norm(quat)
    matrix = SE3(rot=SO3.from_quaternion(quat, ordering='xyzw'), trans=pose[:3]).as_matrix() # Check order (wxyz looks correct for orbit footpath)
    matrix = matrix.astype(np.float32)
    return matrix
#--------------------------------------------------------------#

def pose_of_tf(tf : TransformStamped) -> np.ndarray:
    tx = tf.transform.translation.x
    ty = tf.transform.translation.y
    tz = tf.transform.translation.z
    qx = tf.transform.rotation.x
    qy = tf.transform.rotation.y
    qz = tf.transform.rotation.z
    qw = tf.transform.rotation.w
    return np.array([tx,ty,tz,qx,qy,qz,qw])


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
        self.change_buffer_size_topic = rospy.get_param('/ovt/CHANGE_BUFFER_SIZE_TOPIC')


        self.bridge = CvBridge()

        self.tick = 0
        self.rate = rospy.get_param('/ovt/COMPUTE_PERIOD')
        self.classes = list(rospy.get_param('/ovt/CLASSES'))

        self.encoder = WSImageEncoder(VOXSEG_ROOT_DIR, config='configs/ovt.yaml')
        self.device = rospy.get_param('ovt/DEVICE')
        
        # self.buffer contains tuples of (RosImage, CameraInfo representing camera for that image)
        self.buffer = TriggerBuffer(maxlen = self.batch_size, fn=self.request_computation,  clear_on_trigger=True)
        self._lock_buffer = False

        rospy.init_node(self.data_interface_node)

        # User Input Subscriptions
        RospySubscriber(self.change_rate_topic, Float32, callback=self.change_rate)
        RospySubscriber(self.change_buffer_size_topic, Int32, callback = self.change_buffer_size)
        RospySubscriber(self.class_topic, Classes, self.class_name_callback)

        # Elevation Mapping Plugin Publishers
        prob_publisher_topics = list(rospy.get_param('/ovt/PROB_TOPICS'))
        self.class_pub_register = {}
        for i, topic in enumerate(prob_publisher_topics):
            pub = Publisher(topic, RosImage, queue_size=1000)
            self.class_pub_register[self.classes[i]] = pub

        # Robot Input Subscriptions
        robot_image_topic = rospy.get_param('/ovt/ROBOT_IMAGE_TOPIC')
        RospySubscriber(robot_image_topic, CompressedImage, callback=self.image_callback,queue_size=10000)
    
        print('OVT Interface Initialized')
        rospy.spin()
            

    def request_computation(self, buffer):
        """
        This is bound to the trigger buffer, so it is a function on the buffer

        Buffer contains tuples of images and extrinsics

        Inputs:
            images: numpy array of images, shape (B, 3, H, W)
            
            tfs: numpy array of tfs, shape (B, 4, 4)
        """
        with self.lock_buffer(buffer) as buffer_freeze:
            all_probs_msg = self._handle_compute_request(buffer_freeze, self.classes)
            self.publish_probs_and_tfs(all_probs_msg)

    @contextmanager
    def lock_buffer(self, buffer):

        buffer_freeze = list(buffer).copy()
        self._lock_buffer=True

        try:
            yield buffer_freeze
        finally:
            self._lock_buffer=False


    def _handle_compute_request(self, images, classes):
        # Update frpom the most recent tensors 
        images_msg = list(images)
        
        classes = [str(c) for c in classes]
        
        images = torch_from_img_array_msg(images_msg).float().to(self.device)
        t1  = rospy.get_time()
        class_probs = self.encoder.call_with_classes(images, classes, use_adapter=False)
        
        #### DEBUG
        print(f"CLIP Inference Time: {rospy.get_time() - t1}")
        mask = torch.argmax(class_probs[0], dim=0).float()
        cv2_mask = get_cv2_mask(mask)
        self.save_img(images[0].permute(1,2,0).cpu().numpy(),base='rgb')
        self.save_img(cv2_mask,base='semantic')


        all_probs_msg = []
        bridge = CvBridge()
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

        if not self._lock_buffer:
            self.buffer.append(input_rgb_image)
            print(f"{len(self.buffer)} images in buffer")
        else:
            print('buffer currently locked')    
        
    def class_name_callback(self, msg):
        classes = list(msg.classes)
        print(f'Classes recieved: {classes}')

    def change_rate(self, msg):
        new_rate = float(msg.data)
        self.rate = new_rate

    def change_buffer_size(self, msg):
        new_size = int(msg.data)

        new_buffer = deque(maxlen=new_size)

        for i in self.buffer:
            new_buffer.append(i)
        
        self.buffer = new_buffer

        if new_size < len(self.buffer):
            rospy.logwarn('Warning: decreased buffer size, resulting in lost data')


    def save_img(self, img, base='rgb'):
        
        cv2.imwrite(F"{VOXSEG_ROOT_DIR}/output/{base}_{self.tick // self.rate}.png", img)


class RobotDataInterface:
    def __init__(self):


        self.bridge = CvBridge()

        self.tick = 0
        self.rate = 50
        
        rospy.init_node(self.data_interface_node)

        # TODO: Pull topic names from some config

        # Cannot just use tf_sub as it has no header...
        # instead we need to add an intermediate sub/pub trio. 
        self.tf_main_sub = RospySubscriber("/tf", TFMessage, callback=self.publish_tf_list_to_specific_tfs)

        # Create subscribers for the image topics
        self.rgb_sub = SyncedSubscriber("/wide_angle_camera_front/image_color_rect/compressed", CompressedImage)
        self.depth_sub = SyncedSubscriber("/depth_camera_front_upper/depth/image_rect_raw", Image)

        # Publishers for the individual tf topics
        self.tf_odom_pub = Publisher("/tf_odom", TransformStamped, queue_size=10) # NOTE: Arbitrary number
        self.tf_rgb_pub = Publisher("/tf_rgb", TransformStamped, queue_size=10)
        self.tf_depth_pub = Publisher("/tf_depth", TransformStamped, queue_size=10)

        # Create subscribers for the tf topics
        self.tf_odom_sub = SyncedSubscriber("/tf_odom", TransformStamped)
        self.tf_rgb_sub = SyncedSubscriber("/tf_rgb", TransformStamped)
        self.tf_depth_sub = SyncedSubscriber("/tf_depth", TransformStamped)

        self.data_pub = Publisher(IMAGE_TOPIC, ImageArray, queue_size=10)

        # Synchronize the topics
        ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, 
                                           self.tf_odom_sub, self.tf_rgb_sub, self.tf_depth_sub], 
                                           slop=0.1, queue_size=10) # NOTE: 0.1 default, 10 Arbitrary number
        ats.registerCallback(self.callback)
        rospy.spin()
    
    def publish_tf_list_to_specific_tfs(self, tf_msg):
        # get camera tfs
        tfs_list : List[TransformStamped] = tf_msg.transforms

        # find transforms of interest        
        rgb_frame_id = "wide_angle_camera_front_camera_parent" #rgb_img
        depth_frame_id = "depth_camera_front_upper_depth_optical_frame" #depth_img
        base_frame_id = "base"

        for tf in tfs_list:
            if tf.child_frame_id == base_frame_id:
                self.tf_odom_pub.publish(tf)
            elif tf.child_frame_id == rgb_frame_id:
                self.tf_rgb_pub.publish(tf)
            elif tf.child_frame_id == depth_frame_id:
                self.tf_depth_pub.publish(tf)

    def callback(self,
                 input_rgb_image: CompressedImage, 
                 input_depth_image: Image, 
                 odom_transform: TransformStamped, 
                 rgb_transform: TransformStamped, 
                 depth_transform: TransformStamped):
        self.tick += 1
        if self.tick % self.rate != 0:
            return

        try:
            # Convert compressed RGB image to OpenCV format and then to numpy array
            decoded_rgb_image = self.bridge.compressed_imgmsg_to_cv2(input_rgb_image, desired_encoding="bgr8")
            rgb_image_array = np.array(decoded_rgb_image)

            # Convert depth image to OpenCV format and then to numpy array
            decoded_depth_image = self.bridge.imgmsg_to_cv2(input_depth_image, desired_encoding="passthrough")
            decoded_depth_image = cv2.rotate(decoded_depth_image, cv2.ROTATE_180)
            depth_image_array = np.array(decoded_depth_image) / 1000 # NOTE: this converts mm to meters
            
            # Extract transformation matrices from input transformations
            odom_to_base_transform = transformation_matrix_of_pose(pose_of_tf(odom_transform))
            base_to_rgb_transform = transformation_matrix_of_pose(pose_of_tf(rgb_transform))
            base_to_depth_transform = transformation_matrix_of_pose(pose_of_tf(depth_transform))

            # Compute global transformations for RGB and depth images
            global_rgb_transform = (odom_to_base_transform @ base_to_rgb_transform).flatten()
            global_depth_transform = (odom_to_base_transform @ base_to_depth_transform).flatten()

            # Create message objects from the numpy arrays and transformations
            rgb_image_message = get_image_msg(rgb_image_array, input_depth_image.header.stamp)
            depth_image_message = get_depth_msg(depth_image_array, input_depth_image.header.stamp)
            rgb_transform_message = get_cam_msg(global_rgb_transform)
            depth_transform_message = get_cam_msg(global_depth_transform)

            # Publish the extracted and processed information
            self.data_pub.publish(DepthImageInfo(rgb_image=rgb_image_message, 
                                                depth_image=depth_image_message, 
                                                cam_extrinsics=rgb_transform_message, 
                                                depth_extrinsics=depth_transform_message))
            
            rospy.loginfo("Saved synchronized images with timestamp: %s", input_depth_image.header.stamp)

        except Exception as e:
            # Consider adding some error logging here
            rospy.logerr(f"Error processing images: {str(e)}")

    def save_rgb(self, input_rgb_image: CompressedImage):
        try:
            # Convert compressed RGB image to OpenCV format and then to numpy array
            decoded_rgb_image = self.bridge.compressed_imgmsg_to_cv2(input_rgb_image, desired_encoding="bgr8")
            cv2.imwrite(F"output/rgb_{self.rgb_batchnum}_{input_rgb_image.header.stamp.secs}_{input_rgb_image.header.stamp.nsecs}.png", decoded_rgb_image)
            rospy.loginfo("Saved rgb, stamp@ %s", input_rgb_image.header.stamp.secs)
        except CvBridgeError as e:
            # Consider adding some error logging here
            rospy.logerr(f"Error processing images: {str(e)}")

    def save_depth(self, input_depth_image: Image):
        try:
            # Convert depth image to OpenCV format and then to numpy array
            decoded_depth_image = self.bridge.imgmsg_to_cv2(input_depth_image, desired_encoding="32FC1")
            decoded_depth_image = cv2.rotate(decoded_depth_image, cv2.ROTATE_180)
            depth_img_vis = (1 - 255*decoded_depth_image/decoded_depth_image.max()).astype('uint8')
            depth_img_vis = cv2.applyColorMap(depth_img_vis, cv2.COLORMAP_JET)

            cv2.imwrite(F"output/depth_{self.depth_batchnum}_{input_depth_image.header.stamp.secs}_{input_depth_image.header.stamp.nsecs}.jpg", depth_img_vis)
            rospy.loginfo("Saved depth, stamp@:%s", input_depth_image.header.stamp.secs)
        except CvBridgeError as e:
            # Consider adding some error logging here
            rospy.logerr(f"Error processing images: {str(e)}")
