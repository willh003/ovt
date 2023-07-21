import rospy
from costmap_2d.msg import VoxelGrid
from geometry_msgs.msg import Point32, Vector3
from voxseg.msg import DepthImageInfo, TransformationMatrix, Classes
from voxseg.srv import VoxelComputation, VoxelComputationResponse
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import torch

from modules.data import BackendData
from modules.voxel_world import VoxelWorld
from modules.config import world_config, VOXSEG_ROOT_DIR, SERVER_NODE, IMAGE_TOPIC, RESET_TOPIC, CLASS_TOPIC, VOXEL_REQUEST_SERVICE


class VoxSegServer:
    def __init__(self, batch_size=5):
        """
        Subscribes to the class name and image topics

        The VoxelWorld arguments are defined by config.world_config

        Inputs: 
            batch_size: the number of images to accumulate before projecting them into the world. 
            None to only perform projections on a compute_request
        """
        self.world = VoxelWorld(**world_config, root_dir=VOXSEG_ROOT_DIR) 
        self.data = BackendData(device='cuda', batch_size=batch_size)

        # keep track of number of images seen so far
        self.img_count = 0
        self.batch_size = batch_size
        
        rospy.init_node(SERVER_NODE, anonymous=True)
        rospy.Subscriber(IMAGE_TOPIC, DepthImageInfo, self._depth_image_callback)
        rospy.Subscriber(CLASS_TOPIC, Classes, self._class_name_callback)
        rospy.Subscriber(RESET_TOPIC, String, self._reset_callback)
        rospy.Service(VOXEL_REQUEST_SERVICE, VoxelComputation, self._handle_compute_request)

        print('Backend Setup')

        rospy.spin() # not sure if this will be needed here

    def _handle_compute_request(self, req):
        
        # Get the last tensors (will be None if in batch mode and no tensors have 
        # been added since the last time get_tensors was called)
        tensors = self.data.get_tensors(world=self.world)
        if tensors:
            image_tensor, depths, cam_locs = tensors
            self.world.batched_update_world(image_tensor, depths, cam_locs)

        voxel_classes = self.world.get_voxel_classes(self.data.classes, min_points_in_voxel=int(req.min_pts_in_voxel))

        x,y,z = voxel_classes.size()

        voxel_classes_unsigned = voxel_classes + 1 # need to convert them to bytes for ros, but -1 classes will cause issues with this
        flattened_voxels = voxel_classes_unsigned.flatten().byte().tolist()
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
        print(f'Depth Image {len(self.data.all_images)} Received')

        # Update the world as images come in, if batch size is defined
        if self.batch_size:
            self.img_count += 1
            if self.img_count == self.batch_size:
                image_tensor, depths, cam_locs = self.data.get_tensors(world=self.world)
                self.world.batched_update_world(image_tensor, depths, cam_locs)
                self.img_count = 0

    def _reset_callback(self, msg):
        self.world.reset_world()
        self.img_count = 0
        print('World has been reset')

    def _class_name_callback(self, msg):
        classes = list(msg.classes)
        self.data.add_classes(classes)

        print(f'Classes recieved: {classes}')
        