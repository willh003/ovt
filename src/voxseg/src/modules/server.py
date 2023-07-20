import rospy
from costmap_2d.msg import VoxelGrid
from geometry_msgs.msg import Point32, Vector3
from voxseg.msg import DepthImageInfo, TransformationMatrix, Classes
from voxseg.srv import VoxelComputation, VoxelComputationResponse
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import torch

from modules.data import BackendData
from modules.config import *


class VoxSegServer:
    def __init__(self, world):
        """
        Inputs:
            world: a VoxelWorld object
        
        Creates the vseg node
        Subscribes to the class name and image topics
        """
        self.world = world 
        self.data = BackendData()
        
        rospy.init_node(SERVER_NODE, anonymous=True)
        rospy.Subscriber(IMAGE_TOPIC, DepthImageInfo, self._depth_image_callback)
        rospy.Subscriber(CLASS_TOPIC, Classes, self._class_name_callback)
        rospy.Service(VOXEL_REQUEST_SERVICE, VoxelComputation, self._handle_compute_request)

        print('Backend Setup')

        rospy.spin() # not sure if this will be needed here

    def _handle_compute_request(self, req):
        image_tensor, depths, cam_locs = self.data.get_tensors()

        # NOTE: update the world as images come in instead
        self.world.batched_update_world(image_tensor, depths, cam_locs)
        voxel_classes = self.world.get_voxel_classes(self.data.classes, min_points_in_voxel=int(req.min_pts_in_voxel))

        x,y,z = voxel_classes.size()

        flattened_voxels = voxel_classes.flatten().int().tolist()
        origin = Point32(self.world.voxel_origin[0], self.world.voxel_origin[1], self.world.voxel_origin[2])
        
        resolution = self.world.compute_resolution()
        vec_resolution = Vector3(x=resolution[0], y=resolution[1],z=resolution[2])

        voxel_msg = VoxelGrid(data=flattened_voxels, 
                  origin=origin,
                  resolutions=vec_resolution,
                  size_x=x,
                  size_y=y,
                  size_z=z)
        
        voxel_response = VoxelComputationResponse(voxels=voxel_msg)
        
        return voxel_response


    def _depth_image_callback(self, msg):

        image_msg = msg.rgb_image
        depths_msg = msg.depth_image
        extrinsics_msg = msg.cam_extrinsics

        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        np_image = np.array(img)
        # convert to BGR
        np_image = np_image[:, :, ::-1]

        depths = bridge.imgmsg_to_cv2(depths_msg, desired_encoding="passthrough")
        np_depths = np.array(depths)

        extrinsics_1d = np.array(extrinsics_msg.matrix)
        extrinsics_2d = extrinsics_1d.reshape(4,4)

        self.data.add_depth_image(np_image, np_depths, extrinsics_2d)

    def _class_name_callback(self, msg):
        classes = list(msg.classes)
        self.data.add_classes(classes)
        