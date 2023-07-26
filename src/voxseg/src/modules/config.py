import torch

VOXSEG_ROOT_DIR = '/home/pcgta/Documents/eth/voxseg/src/voxseg/src'

################ SERVER PARAMS #################

# the number of images to accumulate before running unprojection and ovseg inference
# None to wait until clip inference is called
BATCH_SIZE = None 

WORLD_CONFIG = {
    'world_dim': (20,20,5),
    'grid_dim': (40,40,10),
    'voxel_origin': (0,0,0),
    'embed_size': 768,
    'cam_intrinsics': torch.Tensor([[-575.6040,    0.0000,  360.0000,    0.0000],
                    [   0.0000, -575.6040,  270.0000,    0.0000],
                    [   0.0000,    0.0000,    1.0000,    0.0000],
                    [   0.0000,    0.0000,    0.0000,    1.0000]])
}

################# ROS INFO ####################
IMAGE_TOPIC = 'voxseg_image_topic'
CLASS_TOPIC = 'voxseg_classes_topic'
VOXEL_TOPIC = 'voxseg_voxels_topic'
WORLD_DIM_TOPIC = 'voxseg_world_dim_topic'
RESET_TOPIC = 'voxseg_reset_topic'

SERVER_NODE = 'voxseg_backend'
CLIENT_NODE = 'voxseg_frontend'

VOXEL_REQUEST_SERVICE = 'voxel_request'

RVIZ_NODE = 'voxseg_rviz'
MARKER_TOPIC = 'voxseg_marker_topic'



