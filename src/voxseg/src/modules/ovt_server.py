import rospy

from voxseg.msg import Classes
from voxseg.srv import ImageSeg, ImageSegResponse
from cv_bridge import CvBridge

import numpy as np
import json
import torch

from modules.real_data_cfg import *
from modules.utils import convert_dictionary_array_to_dict
from modules.ovseg.open_vocab_seg.ws_ovseg_model import WSImageEncoder


SEG_REQUEST_SERVICE = 'ovt_srv'

class OVT:
    def __init__(self):
        """
        Subscribes to the class name and image topics

        The VoxelWorld arguments are defined by config.WORLD_CONFIG
        
        BATCH_SIZE is the number of images to accumulate before projecting them into the world. 
        Set it to None to only perform projections on a compute_request
        """
        use_prompts = False
        self.classes = ['Traversable', 'Untraversable', 'Obstacle']
        self.prompts = {}
        self.groups = {}
        self.use_prompts = use_prompts

        self.encoder = WSImageEncoder(VOXSEG_ROOT_DIR, config='configs/ovt.yaml')
        
        rospy.init_node(SERVER_NODE, anonymous=True)
        rospy.Subscriber(CLASS_TOPIC, Classes, self._class_name_callback)
        rospy.Service(SEG_REQUEST_SERVICE, ImageSeg, self._handle_compute_request)
    
        print('Backend Has Been Initialized')

        rospy.spin()

   
    def _handle_compute_request(self, req):
        # Update frpom the most recent tensors 
        images = list(req.images)
        h = int(images[0].height) 
        w = int(images[0].width)
        bridge = CvBridge()

        all_images = np.zeros((len(images), 3, h, w))
        for image_msg, i in enumerate(images):
            rgb_msg = image_msg.rgb_image

            img = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
            np_image = np.array(img)
            np_image = np_image[:, :, ::-1] # convert to BGR
            np_image = np.moveaxis(np_image, -1, 0) # convert to 3,H,W

            all_images[i] = np_image


        
            

        # voxel_response = ImageSeg(data=flattened_voxels, 
        #           origin=origin,
        #           resolutions=vec_resolution,
        #           size_x=x,
        #           size_y=y,
        #           size_z=z)
        
        # # publish to voxel topic (only for use with rviz while simulation is running)
        # voxel_msg = VoxelGrid(header= voxel_response.header,
        #                 data=voxel_response.data,
        #                 origin = voxel_response.origin,
        #                 resolutions = voxel_response.resolutions,
        #                 size_x=voxel_response.size_x,
        #                 size_y=voxel_response.size_y,
        #                 size_z=voxel_response.size_z
        #                 )
        # self.voxel_pub.publish(voxel_msg)

        print('Computation Complete')
        
        return voxel_response


    def KV_list_to_dict(self, kvs):
        """
        kvs: list of voxseg.msg.StrArrKV messages 
        """
        r = {}
        for kv in kvs:
            key = str(kv.key)
            values = list(kv.values)
            r[key] = [str(value) for value in values]

        return r

    def unserialize_string(self, ser):
        """
        ser: str, json serialized
        """
        return json.loads(ser)

    def _class_name_callback(self, msg):

        prompts = convert_dictionary_array_to_dict(msg.prompts)
        groups = convert_dictionary_array_to_dict(msg.groups)


        classes = list(msg.classes)
        use_prompts = bool(msg.use_prompts)
        self.classes = classes
        self.prompts = prompts
        self.groups = groups
        self.use_prompts = use_prompts

        print(f'Classes recieved: {classes}')
