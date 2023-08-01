from typing import Tuple, Union
from collections import deque
import torch
import numpy as np


class BackendData:
    def __init__(self, device='cuda', batch_size=None):
        """
        Stores image, depth, intrinsics, and class information. 
        
        NOTE: BackendData objects assume that the depth and rgb cameras have the same intrinsics and extrinsics.
        For unaligned cameras, use UnalignedData.

        Inputs:
            batch_size: size of data buffers. Warning: data may be lost if get_tensors is called with a higher period than batch_size
        """
        self.all_images = []
        self.all_depths = []
        self.rgb_extrinsics = []

        if batch_size:
            self.recent_image_data = deque(maxlen = batch_size)
            self.recent_depth_data = deque(maxlen = batch_size)
            self.recent_rgb_extr = deque(maxlen = batch_size)
        else:
            self.recent_image_data = deque()
            self.recent_depth_data = deque()
            self.recent_rgb_extr = deque()

        self.classes = []
        self.prompts = {}
        self.use_prompts = False
        self.device = device

        self.image_data_start = 0
    

    def add_depth_image(self, image, depths, rgb_extrinsics):
        """
        Inputs:
            image: np array of size (height, width, channels), representing a BGR image
            
            depths: np array of size (height, width)
            
            rgb_extrinsics: np array of size (4,4)
        """
        self.all_images.append(image)
        self.all_depths.append(depths)
        self.rgb_extrinsics.append(rgb_extrinsics)

        self.recent_image_data.append(image)
        self.recent_depth_data.append(depths)
        self.recent_rgb_extr.append(rgb_extrinsics)

    def reset_buffers(self):
        """
        Call this in order to only look at new images that are added
        
        Still retains the old images, but functions like get_tensors will start from the index of new data
        """
        self.recent_image_data.clear()
        self.recent_depth_data.clear()
        self.recent_rgb_extr.clear()

    def reset_all(self):
        """
        Call this in order to reset all past images (blank slate)
        """
        self.all_images = []
        self.all_depths = []
        self.rgb_extrinsics = []

        self.reset_buffers()

    def add_class_info(self, classes, prompts, groups, use_prompts):
        """
        Inputs:
            classes: string list containing classes
            prompts: dict containing classes and corresponding prompts
            groups: dict of groups and corresponding classes 
            use_prompts: whether to use manually designated prompts
        """
        self.classes = classes
        self.prompts = prompts
        self.groups = groups
        self.use_prompts = use_prompts

    def get_all_tensors(self, world):

        depths_np = np.stack(self.all_depths)
        image_tensor = world.predictor.image_list_to_tensor(self.all_images)
        extrinsics_np = np.stack(self.rgb_extrinsics)

        depth_tensor = torch.from_numpy(depths_np).to(self.device)
        depth_tensor = depth_tensor.unsqueeze(1)
        extr_tensor = torch.from_numpy(extrinsics_np).float().to(self.device)

        self.reset_buffers() # clear recent data, so the same data isn't projected multiple times

        return image_tensor, depth_tensor, extr_tensor

    def get_tensors(self, world):
        """
        Inputs:
            world: the world for this data (must contain a predictor)
        Returns: 
            The depths, images, and extrinsics that have been added since the last get_tensors() call, as torch.tensors 
                images: b, 3, h, w
                
                depths: b, 1, h, w
                
                extrinsics: b, 4, 4
            If no tensors have been added, then None
        """
        if len(self.recent_image_data) == 0:
            return None

        depths_np = np.stack(self.recent_depth_data)
        image_tensor = world.predictor.image_list_to_tensor(list(self.recent_image_data))
        extrinsics_np = np.stack(self.recent_rgb_extr)

        depth_tensor = torch.from_numpy(depths_np).to(self.device)
        depth_tensor = depth_tensor.unsqueeze(1)
        extr_tensor = torch.from_numpy(extrinsics_np).float().to(self.device)

        self.reset_buffers() # clear recent data, so the same data isn't projected multiple times

        return image_tensor, depth_tensor, extr_tensor
    
class UnalignedData(BackendData):
    def __init__(self, device='cuda', batch_size=None):
        """
        A BackendData object, which also stores depth camera extrinsics.
        This is for the case when the depth and RGB cameras are not aligned
        """
        super().__init__(device=device, batch_size=batch_size)
        
        self.depth_extrinsics = []

        if batch_size:
            self.recent_depth_extr = deque(maxlen = batch_size)
        else:
            self.recent_depth_extr = deque()

    def add_depth_image(self, image, depths, rgb_extr, depth_extr):
        super().add_depth_image(image, depths, rgb_extr)
        self.recent_depth_extr.append(depth_extr)
        self.depth_extrinsics.append(depth_extr)
        
    def reset_buffers(self):
        super().reset_buffers()
        self.recent_depth_extr.clear()

    def reset_all(self):
        self.depth_extrinsics = []
        self.reset_all()

    def get_all_tensors(self, world):
        depth_extr_np = np.stack(self.depth_extrinsics)
        depth_extr_tensor = torch.from_numpy(depth_extr_np).to(self.device)
        image_tensor, depth_tensor, rgb_extr_tensor = super().get_all_tensors(world)

        self.reset_buffers()
        return image_tensor, depth_tensor, rgb_extr_tensor, depth_extr_tensor

    def get_tensors(self, world):
        tensors = super().get_tensors(world)
        if tensors is not None:
            image_tensor, depth_tensor, rgb_extr_tensor = tensors
        else:
            return None

        depth_extr_np = np.stack(self.depth_extrinsics)
        depth_extr_tensor = torch.from_numpy(depth_extr_np).float().to(self.device)

        return image_tensor, depth_tensor, rgb_extr_tensor, depth_extr_tensor