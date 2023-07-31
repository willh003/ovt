import os
import multiprocessing as mp
import time
import torch
from typing import Union, List, Dict

from modules.utils import *

from modules.ovseg.open_vocab_seg.ws_ovseg_model import WSTextFeatureModel, WSImageEncoder


WINDOW_NAME = "OVSeg"

class VoxelWorld:
    def __init__(self, world_dim, grid_dim, embed_size, cam_intrinsics, voxel_origin=(0,0,0), device='cuda', root_dir=None):
        """
        world_dim: tuple (3,), dimension of world (probably in m)

        grid_dim: tuple (3,), dimension of the voxel grid (indices per axis)

        voxel_origin: tuple (3,), desired origin of the voxel grid w.r.t the world

        embed_size: int, dimension of the feature embeddings

        cam_intrinsics: torch.FloatTensor (4,4): intrinsics matrix of camera being used 

        root_dir: the directory from which python is being called (PYTHON_PATH). None if running from current directory
        """
        self.device = device
        self.cam_intrinsics = cam_intrinsics.to(self.device)
        t0 = time.time()
        self.predictor = WSImageEncoder(root_dir = root_dir)
        t1 = time.time()
        print(f'Image Embeddor Load Time: {t1 - t0}')

        model_name = self.predictor.cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME
        mp_depth = self.predictor.cfg.MODEL.CLIP_ADAPTER.MASK_PROMPT_DEPTH

        
        self.text_model = WSTextFeatureModel(model_name, mp_depth)
        t2 = time.time()
        print(f'Clip load time: {t2- t1}')

        # Initialize voxels which represent the world.
        gx, gy, gz = grid_dim

        self.world_dim = torch.FloatTensor(world_dim).to(device)
        self.grid_dim = torch.FloatTensor((gx,gy,gz)).to(device)
        self.embed_size = embed_size
        self.voxel_origin = torch.FloatTensor(voxel_origin).to(device)

        self.voxels = torch.zeros((gx+2,gy+2,gz+2,embed_size),device=device)
        self.grid_count = torch.zeros((gx+2,gy+2,gz+2),device=self.device) # for the running average
    
    def update_dims(self, world_dim, grid_dim):
        """
        world_dim: array_like, shape (3,)
        grid_dim: array_like, shape(3,)
        """
        gx, gy, gz = grid_dim

        self.world_dim = torch.FloatTensor(world_dim).to(self.device)
        self.grid_dim = torch.FloatTensor((gx,gy,gz)).to(self.device)

        self.voxels = torch.zeros((gx+2,gy+2,gz+2,self.embed_size),device=self.device)
        self.grid_count = torch.zeros((gx+2,gy+2,gz+2),device=self.device) # for the running average

        print(self.voxels.size())

    def compute_resolution(self) -> torch.Tensor:
        """
        returns the resolution given self.grid_dim and self.world_dim 
        resolution is defined as voxels per world space unit, for each dim
        """
        return self.grid_dim / self.world_dim

    def reset_world(self):
        self.voxels = torch.zeros_like(self.voxels)
        self.grid_count = torch.zeros_like(self.grid_count)
    
    def _aligned_update_world(self, rgb, K_rgb, T_rgb, d, K_d, T_d):
        """
        Inputs:
        rgb: torch.Tensor, (B, 3, h, w)
        K_rgb: (B, 4, 4), rgb camera intrinsics
        T_rgb: (B, 4, 4), rgb camera extrinsics
        d: torch.Tensor, (B, 1, h, w)
        K_d: (B, 4, 4), depth camera intrinsics
        T_d: (B, 4, 4), depth camera extrinsics
        """

        B, h, w, _= rgb.size()
        valid_world_locs, valid_rgb_pixels = align_depth_to_rgb(rgb, K_rgb, T_rgb, d, K_d, T_d)
        voxel_locs = self.world_to_voxel(valid_world_locs)

        embeddings = self.get_image_embeddings(rgb)
        embeddings_upsampled = interpolate_features(embeddings, h, w)
        
        valid_embeddings = embeddings_upsampled[valid_rgb_pixels]


        self.voxels, self.grid_count = update_grids_aligned(valid_embeddings, voxel_locs, self.voxels, self.grid_count)

    def _update_world(self, images, depths, cam_extrinsics):
        """
        Behavior:
            Run the model on image, and update the corresponding voxels
        """
        
        t1 = time.time()
        world_locs = self.image_to_world(depths.to(self.device), cam_extrinsics.to(self.device))
        voxel_locs = self.world_to_voxel(world_locs)
        t2 = time.time()
        print(f'Pixel projection time: {t2 - t1}')

        embeddings = self.get_image_embeddings(images)
        t3 = time.time()
        print(f'Image embedding time: {t3 - t2}')

        self.voxels, self.grid_count = update_grids(embeddings, voxel_locs, self.voxels, self.grid_count)

        t4 = time.time()
        print(f'World update time: {t4 - t3}')

    def get_image_embeddings(self, images):
        preds = self.predictor.batch_call(images)
        logits = preds['pred_logits'].to(self.device)
        seg = preds['pred_masks'].to(self.device)
        max_indices = torch.argmax(seg, dim=1) # NOTE: we are guessing that seg is the values/probabilities for each segment, and we can just take the max

        batch, _, height, width = seg.size()
        batch, _, feature_size = logits.size()

        selected_features = torch.zeros(batch, height, width, feature_size, device=self.device)
        for i in range(batch): # NOTE: not sure how to batch this, but there must be a way
            selected_features[i] = logits[i][max_indices[i]]
        
        return selected_features

    def batched_update_world(self, all_images, all_depths, all_extrinsics, max_batch_size = 10, update_func=None):
        """
        Runs the model on all_images, in batches of size max_batch_size
        
        max_batch_size avoids memory overflow when dealing with large sets of images
        """
        splits = list(range(0, len(all_images), max_batch_size))
        splits.append(len(all_images)) # ensure the leftovers are still included

        for i in range(len(splits) - 1): 
            start = splits[i]
            end = splits[i+1]

            images = all_images[start:end]
            depths = all_depths[start:end]
            extrinsics = all_extrinsics[start:end]

            self._update_world(images, depths, extrinsics)



    def world_to_voxel(self, world_locs : torch.Tensor,):
        """
        Input:
            locs: torch.FloatTensor, shape (batch, *, 3): the world coordinates of each pixel in an input image
            
        Returns:
            torch.IntTensor, shape (batch, *, 3): the voxel indices of each pixel

        Math:
            p           \in [-W/2, W/2)
            ^^/W        \in [-1/2, 1/2)
            ^^ + 1/2    \in [0,1)
            ^^*G        \in [0,G)
            floor(^^)   \in [0,1,...G-1]
            ^^ + 1      \in [0,1,...G+1] (to include -1 and G for out-of-domain points) 
            ^^ < 0 : ^^ = 0     (enforces domain)
            ^^ > G+1 : ^^ = G+1 (enforces domain)
        """
        voxel_locs = world_locs - self.voxel_origin
        voxel_locs= (voxel_locs/self.world_dim) # NOTE: Cannot be piped (tensor.a().b()... because CUDA gets mad)
        voxel_locs = voxel_locs + 0.5
        voxel_locs = voxel_locs * self.grid_dim
        voxel_locs = voxel_locs.floor()
        voxel_locs = voxel_locs + 1
        voxel_locs = voxel_locs.clamp_min(0)
        voxel_locs = voxel_locs.clamp_max(self.grid_dim + 1)
        return voxel_locs
    
    
    def image_to_world(self, depth, cam_extrinsics):
        """
        Inputs:
            depth: torch.tensor, (batch, 1, height, width)

            cam_extrinsics: torch.tensor, (batch, 4,4), represents transformation matrix of camera in world coordinates T_CIW
        Returns:
            torch.tensor, (batch, height, width, 3): the world coordinates of each pixel in image, given camera intrinsics and extrinsics
            if these coordinates are greater than world boundaries, they are clipped to the boundaries.
        Requires:
            intrinsics are constant for all images in the batch
        """
        batches, _, height, width = depth.size()

        pixels = get_all_pixels(height, width)

        px_w = unproject(self.cam_intrinsics, cam_extrinsics, pixels, depth.squeeze(1))
        return px_w.nan_to_num().permute(0,2,3,1)
    
    def get_voxel_classes(self, classes: Union[List[str], Dict[str, List[str]]], min_points_in_voxel=0):
        """
        Inputs:
            classes: list of class names, or dictionary of {class name: [prompts for class]}
        
        Returns:
            tensor with shape (*self.grid_dim), with each element representing the class label of that voxel
        """
        t1 = time.time()
        x,y,z,_ = self.voxels.size()
        
        flat_voxels = self.voxels.flatten(end_dim = -2)
        non_empty_voxel_mask = self.grid_count.flatten() > min_points_in_voxel # considered empty if no point was found in the voxel

        voxels_for_inference = flat_voxels.cuda()[non_empty_voxel_mask.cuda()]

        if type(classes) == dict:
            labeled_voxel_classes = self.text_model.get_nearest_classes(voxels_for_inference, classes,manual_prompts=True).float()
        else:
            labeled_voxel_classes = self.text_model.get_nearest_classes(voxels_for_inference, classes,manual_prompts=False).float()

        all_voxel_classes = torch.ones(x*y*z).cuda() * -1 # -1 means no label
        all_voxel_classes[non_empty_voxel_mask] = labeled_voxel_classes

        all_voxel_classes_reshaped = all_voxel_classes.view(x,y,z)
        print(f'Clip inference and similarity calculation time: {time.time()- t1}')

        return all_voxel_classes_reshaped

    def get_classes_by_groups(self, classes, groups, min_points_in_voxel):
        """
        classes: [class1, class2, ...]
        groups: {group1: [class1, class3], group2: [class2], ...}
        Requires: a class is in a group iff it is in classes
        """
        t1 = time.time()
        x,y,z,_ = self.voxels.size()
        
        flat_voxels = self.voxels.flatten(end_dim = -2)
        non_empty_voxel_mask = self.grid_count.flatten() > min_points_in_voxel # considered empty if no point was found in the voxel

        voxels_for_inference = flat_voxels.cuda()[non_empty_voxel_mask.cuda()]

        if type(classes) == dict:
            labeled_voxel_classes = self.text_model.get_nearest_classes(voxels_for_inference, classes,manual_prompts=True).float()
            class_list = list(classes.keys())
        else:
            labeled_voxel_classes = self.text_model.get_nearest_classes(voxels_for_inference, classes,manual_prompts=False).float()
            class_list = classes

        all_voxel_classes = torch.ones(x*y*z).cuda() * -1 # -1 means no label
        all_voxel_classes[non_empty_voxel_mask] = labeled_voxel_classes


        class_to_group = {}
        keys = list(groups.keys())
        for i in range(len(keys)):
            group_key = keys[i]
            for cls in groups[group_key]:
                class_to_group[class_list.index(cls)] = i

        all_voxel_groups = torch.as_tensor([class_to_group.get(i.item(), i.item()) for i in all_voxel_classes])

        all_voxel_groups_reshaped = all_voxel_groups.view(x,y,z)
        print(f'Clip inference and similarity calculation time: {time.time()- t1}')

        return all_voxel_groups_reshaped
        

def batch_test():
    #classes = ['ground', 'rock', 'brick', 'fire hydrant', 'heavy machinery']
    prompts={'traversable': ['traversable','an image of easy terrain','an image of flat ground','flat ground', 'ground that could be walked on', 'easy to walk on', 'it is easy to walk on this'],
             'untraversable': ['untraversable','an image of challenging terrain', 'dangerous terrain', 'an image of bumpy terrain', 'difficult to walk on', 'it is difficult to walk on this'],
             'obstacle': ['obstacle','an obstacle', 'this is an obstacle', 'an image of an obstacle', 'this is an image of an obstacle', 'this is in the way']
             }
    data_dir = '/home/pcgta/Documents/eth/wild_visual_navigation/wild_visual_navigation_orbit/feat-extract-out/test_16'
    
    K = torch.Tensor([[-575.6040,    0.0000,  360.0000,    0.0000],
                    [   0.0000, -575.6040,  270.0000,    0.0000],
                    [   0.0000,    0.0000,    1.0000,    0.0000],
                    [   0.0000,    0.0000,    0.0000,    1.0000]])

    print('STARTING')
    t1 = time.time()
    # below are params for the forest path test
    #world = VoxelWorld(world_dim = (60,80,20), grid_dim = (60,70, 30), voxel_origin=(-28, 0,10), embed_size= 768, cam_intrinsics=K)
    world = VoxelWorld(world_dim = (14,30,4), grid_dim = (42,60, 12), voxel_origin=(0, 0,2), embed_size= 768, cam_intrinsics=K)

    t2 = time.time()
    print(f'World init time: {t2 - t1}')

    images, depths, cam_locs = load_images(os.path.abspath(data_dir))
    
    print(f'Data load time: {time.time() - t2}')
    image_tensor = world.predictor.image_list_to_tensor(images) 

    start = 0
    end = -1

    if len(image_tensor) == 1:
        end = 1
    
    assert end <= len(image_tensor)
    world.batched_update_world(image_tensor[start:end], depths[start:end], cam_locs[start:end])

    voxel_classes = world.get_voxel_classes(prompts)
    visualize_voxel_classes(voxel_classes, [c for c in prompts.keys()], save_dir=None)


    # voxel_classes = world.get_voxel_classes(classes, min_points_in_voxel=i)

    # visualize_voxel_classes(voxel_classes, classes, save_dir='/home/pcgta/Documents/eth/media/construction-world-c-tests', base_name=f'construction_c_{i}')

    #visualize_voxel_occupancy(world)

    

if __name__ == "__main__":
    batch_test()
