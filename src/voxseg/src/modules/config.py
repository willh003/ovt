import torch

VOXSEG_ROOT_DIR = '/home/pcgta/Documents/eth/voxseg/src/voxseg/src'

IMAGE_TOPIC = 'voxseg_image_topic'
CLASS_TOPIC = 'voxseg_classes_topic'
VOXEL_TOPIC = 'voxseg_voxels_topic'
RESET_TOPIC = 'voxseg_reset_topic'

SERVER_NODE = 'voxseg_backend'
CLIENT_NODE = 'voxseg_frontend'

VOXEL_REQUEST_SERVICE = 'voxel_request'

RVIZ_NODE = 'voxseg_rviz'
MARKER_TOPIC = 'voxseg_marker_topic'

world_config = {
    'world_dim': (20,20,5),
    'grid_dim': (40,40,10),
    'voxel_origin': (0,0,0),
    'embed_size': 768,
    'cam_intrinsics': torch.Tensor([[-575.6040,    0.0000,  360.0000,    0.0000],
                    [   0.0000, -575.6040,  270.0000,    0.0000],
                    [   0.0000,    0.0000,    1.0000,    0.0000],
                    [   0.0000,    0.0000,    0.0000,    1.0000]])

}