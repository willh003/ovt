from typing import Callable
import torch
from torch import Tensor

# library imports
from playground import VoxelWorld
from ws.utils import *

#---------------------#
#   hyperparameters   #
#---------------------#
image_width, image_height = 248, 160
UNIT_INTRINSICS = Tensor([[1,0,0],[0,1,0],[image_width/2, image_height/2, 1]])

#--------------------#
#   world_to_voxel   #
#--------------------#
def world_to_voxel_test():
    __VOXES_1 = VoxelWorld(world_dim=(11,13,17),grid_dim=(7,5,3),embed_dim=1024,cam_intrinsics=UNIT_INTRINSICS)
    world_to_voxel : Callable[[Tensor],Tensor] = lambda locs : __VOXES_1.world_to_voxel(locs)

    __corners = Tensor([[0,0,0],[0,0,17],[0,13,0],[0,13,17],[11,0,0],[11,0,17],[11,13,0],[11,13,17]])
    __corners_moved_in = (__corners*Tensor([6/7,4/5,2/3])) + Tensor([11/(2*7),13/(2*5),17/(2*3)])

    assert world_to_voxel(__corners).eq(Tensor(         [[0,0,0],[0,0,3],[0,5,0],[0,5,3],[7,0,0],[7,0,3],[7,5,0],[7,5,3]])).all()
    assert world_to_voxel(__corners_moved_in).eq(Tensor([[0,0,0],[0,0,2],[0,4,0],[0,4,2],[6,0,0],[6,0,2],[6,4,0],[6,4,2]])).all()

#--------------------#
#   image_to_world   #
#--------------------#
def image_to_world_test():
    __VOXES_2 = VoxelWorld(world_dim=(11,13,17),grid_dim=(7,5,3),embed_dim=1024,cam_intrinsics=UNIT_INTRINSICS) 
    image_to_world : Callable[[Tensor, Tensor, Tensor],Tensor] = lambda image, depth, extrinsics : __VOXES_2.image_to_world(image,depth,extrinsics)
    pass

#-----------------#
#   voxel tests   #
#-----------------#
def voxel_tests():
    __VOXES_3 = VoxelWorld(world_dim=(50,50,5),grid_dim=(15,15,5),embed_dim=1024,cam_intrinsics=UNIT_INTRINSICS) 
    __DUMMY_OCCUPANCIES = torch.randn((15,15,5))
    __VOXES_3.grid_count = __DUMMY_OCCUPANCIES

    visualize_voxel_occupancy(__VOXES_3)





    # TEST METHODS
    #print("Trying voxel tests")
    #voxel_tests()
