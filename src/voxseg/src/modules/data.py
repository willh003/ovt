from typing import Tuple, Union
from collections import deque
import torch
import numpy as np

class BackendData:
    def __init__(self, device='cuda', batch_size=None):
        """
        batch_size: size of data buffers. Warning: data may be lost if get_tensors is called with a higher period than batch_size
        """
        self.all_images = []
        self.all_depths = []
        self.all_extrinsics = []

        if batch_size:
            self.recent_image_data = deque(maxlen = batch_size)
            self.recent_depth_data = deque(maxlen = batch_size)
            self.recent_extr_data = deque(maxlen = batch_size)
        else:
            self.recent_image_data = deque()
            self.recent_depth_data = deque()
            self.recent_extr_data = deque()

        self.classes = []
        self.device = device

        self.image_data_start = 0
    

    def add_depth_image(self, image, depths, extrinsics):
        """
        Inputs:
            image: np array of size (height, width, channels), representing a BGR image
            
            depths: np array of size (height, width)
            
            extrinsics: np array of size (4,4)
        """
        self.all_images.append(image)
        self.all_depths.append(depths)
        self.all_extrinsics.append(extrinsics)

        self.recent_image_data.append(image)
        self.recent_depth_data.append(depths)
        self.recent_extr_data.append(extrinsics)

    def reset_buffers(self):
        """
        Call this in order to only look at new images that are added
        
        Still retains the old images, but functions like get_tensors will start from the index of new data
        """
        self.recent_image_data.clear()
        self.recent_depth_data.clear()
        self.recent_extr_data.clear()

    def reset_all(self):
        """
        Call this in order to reset all past images (blank slate)
        """
        self.all_images = []
        self.all_depths = []
        self.all_extrinsics = []

        self.reset_buffers()

    def add_classes(self, classes):
        """
        Inputs:
            classes: string list containing classes
        """
        self.classes = classes

    def get_tensors(self, world) -> Union[Tuple, None]:
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
        extrinsics_np = np.stack(self.recent_extr_data)

        depth_tensor = torch.from_numpy(depths_np).to(self.device)
        depth_tensor = depth_tensor.unsqueeze(1)
        extr_tensor = torch.from_numpy(extrinsics_np).float().to(self.device)

        self.reset_buffers() # clear recent data, so the same data isn't projected multiple times

        return image_tensor, depth_tensor, extr_tensor