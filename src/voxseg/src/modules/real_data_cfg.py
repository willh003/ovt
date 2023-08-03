
import torch
import os

VOXSEG_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

################ SERVER PARAMS #################

# the number of images to accumulate before running unprojection and ovseg inference
# None to wait until clip inference is called
BATCH_SIZE = None 

WORLD_CONFIG = {
    'world_dim': (30,30,5),
    'grid_dim': (40,40,10),
    'voxel_origin': (0,0,0),
    'embed_size': 768,
}


K_RGB = torch.as_tensor([[575.6050407221768, 0.0, 745.7312198525915, 0.0],
                        [0.0, 578.564849365178, 519.5207040671075, 0.0],
                       [ 0.0, 0.0, 1.0, 0.0],
                       [ 0.0, 0.0, 0.0, 1.0]])
K_DEPTH = torch.as_tensor([[423.54608154296875, 0.0, 427.6981506347656, 0.0],
                            [0.0, 423.54608154296875, 240.177734375, 0.0],
                       [ 0.0, 0.0, 1.0, 0.0],
                       [ 0.0, 0.0, 0.0, 1.0]])


CAMS_ALIGNED = False

################# ROS INFO ####################
IMAGE_TOPIC = 'voxseg_image_topic'
CLASS_TOPIC = 'voxseg_classes_topic'
VOXEL_TOPIC = 'voxseg_voxels_topic'
WORLD_DIM_TOPIC = 'voxseg_world_dim_topic'
RESET_TOPIC = 'voxseg_reset_topic'

SERVER_NODE = 'voxseg_backend'
CLIENT_NODE = 'voxseg_frontend'

VOXEL_REQUEST_SERVICE = 'voxel_request'
OVT_REQUEST_SERVICE = 'ovt_srv'

RVIZ_NODE = 'voxseg_rviz'
MARKER_TOPIC = 'voxseg_marker_topic'



