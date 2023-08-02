import rospy
#from costmap_2d.msg import VoxelGrid
from geometry_msgs.msg import Point32, Vector3
from voxseg.msg import DepthImageInfo, WorldInfo, Classes, VoxelGrid
from voxseg.srv import VoxelComputation, VoxelComputationResponse
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import json
import torch

from modules.data import BackendData, UnalignedData
from modules.voxel_world import VoxelWorld

#from modules.config import *
from modules.real_data_cfg import *
from modules.utils import convert_dictionary_array_to_dict

class VoxSegServer:
    def __init__(self):
        """
        Subscribes to the class name and image topics

        The VoxelWorld arguments are defined by config.WORLD_CONFIG
        
        BATCH_SIZE is the number of images to accumulate before projecting them into the world. 
        Set it to None to only perform projections on a compute_request
        """
        self.cams_aligned = CAMS_ALIGNED

        self.world = VoxelWorld(**WORLD_CONFIG, root_dir=VOXSEG_ROOT_DIR) 

        if self.cams_aligned:
            self.data = BackendData(device='cuda', batch_size=BATCH_SIZE)
        else:
            self.data = UnalignedData(device='cuda', batch_size=BATCH_SIZE)

        # keep track of number of images seen so far
        self.img_count = 0
        self.batch_size = BATCH_SIZE
        self.recast_required = True
        
        rospy.init_node(SERVER_NODE, anonymous=True)
        rospy.Subscriber(IMAGE_TOPIC, DepthImageInfo, self._depth_image_callback)
        rospy.Subscriber(CLASS_TOPIC, Classes, self._class_name_callback)
        rospy.Subscriber(WORLD_DIM_TOPIC, WorldInfo, self._world_dim_callback, buff_size=100000) # SET BUFFER SIZE
        rospy.Subscriber(RESET_TOPIC, String, self._reset_callback)
        rospy.Service(VOXEL_REQUEST_SERVICE, VoxelComputation, self._handle_compute_request)
        self.voxel_pub = rospy.Publisher(VOXEL_TOPIC, VoxelGrid, queue_size=10)
    
        print('Backend Has Been Initialized')

        rospy.spin()

    def _update_world(self):
        """
        Updates the server's self.world object
        - gets the most recent image and depth tensors 
        - casts them into the world and updates the voxel representation
        - handles cases where images and depths are not aligned
        """

        # tensors will be None if in batch mode and no tensors have 
        # been added since the last time get_tensors was called
        tensors = self.data.get_tensors(world=self.world)

        if tensors and self.cams_aligned:
            image_tensor, depths, cam_locs = tensors
            self.world.batched_update_world(image_tensor, depths, cam_locs, K_RGB.float())
        elif tensors and not self.cams_aligned:
            image_tensor, depth_tensor, rgb_extr_tensor, depth_extr_tensor = tensors
            self.world.batched_update_world(image_tensor, depth_tensor, rgb_extr_tensor,K_RGB.float(), depth_extr_tensor , K_DEPTH.float())
        
        self.img_count = 0
        torch.cuda.synchronize() # wait for all world updates before doing inference

    def _handle_compute_request(self, req):
        # Update from the most recent tensors 

        if self.recast_required:
            self._update_world()
            self.recast_required = False

        min_pts_in_voxel = req.min_pts_in_voxel
        if self.data.use_prompts:
            voxel_classes = self.world.get_classes_by_groups(self.data.prompts, self.data.groups, min_points_in_voxel=min_pts_in_voxel)
        else:
            voxel_classes = self.world.get_classes_by_groups(self.data.classes, self.data.groups, min_points_in_voxel=min_pts_in_voxel)

        x,y,z = voxel_classes.size()

        voxel_classes_unsigned = voxel_classes + 1 # need to convert them to bytes for ros, but -1 classes will cause issues with this
        flattened_voxels = voxel_classes_unsigned.flatten().byte().tolist()
        origin = Point32(self.world.voxel_origin[0], self.world.voxel_origin[1], self.world.voxel_origin[2])
        
        resolution = self.world.compute_resolution()
        vec_resolution = Vector3(x=resolution[0], y=resolution[1],z=resolution[2])

        voxel_response = VoxelComputationResponse(data=flattened_voxels, 
                  origin=origin,
                  resolutions=vec_resolution,
                  size_x=x,
                  size_y=y,
                  size_z=z)
        
        # publish to voxel topic (only for use with rviz while simulation is running)
        voxel_msg = VoxelGrid(header= voxel_response.header,
                        data=voxel_response.data,
                        origin = voxel_response.origin,
                        resolutions = voxel_response.resolutions,
                        size_x=voxel_response.size_x,
                        size_y=voxel_response.size_y,
                        size_z=voxel_response.size_z
                        )
        self.voxel_pub.publish(voxel_msg)

        print('Computation Complete')
        
        return voxel_response

    def _world_dim_callback(self, msg):
        print('Updating World Dim')
        self.world.update_dims(msg.world_dim, msg.grid_dim)
        self.data.reset_buffers()
        self.data.fill_buffers()
        self.recast_required = True



    def _depth_image_callback(self, msg):

        image_msg = msg.rgb_image
        depths_msg = msg.depth_image
        rgb_extrinsics_msg = msg.cam_extrinsics
        rgb_extrinsics = np.array(rgb_extrinsics_msg).reshape(4,4)


        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        np_image = np.array(img)
        # convert to BGR
        np_image = np_image[:, :, ::-1]

        depths = bridge.imgmsg_to_cv2(depths_msg, desired_encoding="passthrough")
        np_depths = np.array(depths)

        if self.cams_aligned:
            self.data.add_depth_image(np_image, np_depths, rgb_extrinsics)
        else:
            depth_extrinsics_msg = msg.depth_extrinsics
            depth_extrinsics = np.array(depth_extrinsics_msg).reshape(4,4)
            self.data.add_depth_image(np_image, np_depths, rgb_extrinsics, depth_extrinsics)

        print(f'Depth Image {len(self.data.all_images)} Received')

        self.recast_required = True
        # Update the world as images come in, if batch size is defined
        if self.batch_size:
            self.img_count += 1
            if self.img_count == self.batch_size:
                self._update_world()
                self.recast_required = False        
                

    def _reset_callback(self, msg):
        self.world.reset_world()
        self.data.reset_all()
        self.recast_required = True
        self.img_count = 0
        print('World has been reset')

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
        self.data.add_class_info(classes, prompts, groups, use_prompts)

        print(f'Classes recieved: {classes}')
