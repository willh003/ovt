import rospy
#from costmap_2d.msg import VoxelGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image
from voxseg.msg import DepthImageInfo, TransformationMatrix, Classes, StrArrKV, VoxelGrid
from voxseg.srv import VoxelComputation

from cv_bridge import CvBridge
import numpy as np
import torch
from typing import List, Dict, Union
import json

from modules.config import CLIENT_NODE, CLASS_TOPIC, IMAGE_TOPIC, VOXEL_TOPIC, VOXEL_REQUEST_SERVICE
from modules.utils import *

class VoxSegClient:
    def __init__(self):
        """
        initialize the frontend node
        """

        rospy.init_node(CLIENT_NODE, anonymous=True)

        # Important: initialize the pubs before starting to publish
        self.class_pub = rospy.Publisher(CLASS_TOPIC, Classes, queue_size=10)
        self.image_pub = rospy.Publisher(IMAGE_TOPIC, DepthImageInfo, queue_size=10)

    def publish_depth_image(self, image, depth_map, extrinsics):
        """
        Should be called alongside image_callback in the simulation
        Inputs:
            image: a numpy array containing rgb image data, shape (h,w,c)

            depth_map: a numpy array containing depth data, size (h,w)

            extrinsics: a numpy array containing camera extrinsics, size (4,4)

        """
        timestamp = rospy.Time.now()
        full_msg = DepthImageInfo()
        full_msg.rgb_image = get_image_msg(image, timestamp)
        full_msg.depth_image = get_depth_msg(depth_map, timestamp)
        full_msg.cam_extrinsics = get_cam_msg(extrinsics)

        self.image_pub.publish(full_msg)

    def publish_class_names(self, names: Union[List[str], None]=['other'], 
                            groups:Union[Dict[str, List[str]], None]=None,
                            prompts: Union[Dict[str, List[str]], None]=None, 
                            use_prompts=False):
        """
        Should be called whenever the user enters class names in the extension window
        
        Either names or prompts must be specified.
        
        Inputs:
            names: list of class identifiers. If None, defaults to the identifiers in prompts

            prompts: dictionary of class identifier to corresponding prompts
            
            groups: dictionary of group identifier to corresponding class identifiers. Default behavior is that each class identifier gets its own group.
            
            use_prompts: True to use the user-defined prompts, False to use automatically generated prompts with names
        """
        class_msg = Classes()

        class_msg.use_prompts = use_prompts
        if use_prompts:
            if prompts == None:
                raise Exception('use_prompts set to True, but no prompts were specified')
            class_msg.prompts = convert_dict_to_dictionary_array(prompts)
            class_msg.classes = list(prompts.keys())
        else:
            class_msg.classes = names

        if not groups:
            class_msg.groups = convert_dict_to_dictionary_array({name: [name] for name in class_msg.classes})
        else:
            class_msg.groups = convert_dict_to_dictionary_array(groups)


        self.class_pub.publish(class_msg)

    def request_voxel_computation(self, min_pts_in_voxel=0):
        """
        Publishes the VoxelGrid message returned by the request to VOXEL_TOPIC

        Inputs:
            min_pts_in_voxel: the minimum number of points to consider a voxel valid for inference
        
        Returns:
            torch.tensor representing the voxels and torch.tensor representing the world dim
        """
        rospy.wait_for_service(VOXEL_REQUEST_SERVICE)
        try:
            compute_data_service = rospy.ServiceProxy(VOXEL_REQUEST_SERVICE, VoxelComputation)
            voxel_response = compute_data_service(min_pts_in_voxel)

            voxels, world_dim= voxels_from_srv(voxel_response)
            return voxels, world_dim
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return None



if __name__=='__main__':
    try:
        frontend = VoxSegClient()
    except rospy.ROSInterruptException:
        pass