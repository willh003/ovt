import torch
import numpy as np

class BackendData:
    def __init__(self, device):
        self.images = []
        self.depths = []
        self.extrinsics = []
        self.classes = []
        self.device = device

    def add_depth_image(self, image, depths, extrinsics):
        """
        Inputs:
            image: np array of size (height, width, channels), representing a BGR image
            
            depths: np array of size (height, width)
            
            extrinsics: np array of size (4,4)
        """
        self.images.append(image)
        self.depths.append(depths)
        self.extrinsics.append(extrinsics)

    def add_classes(self, classes):
        """
        Inputs:
            classes: string list containing classes
        """
        self.classes = classes

    def get_tensors(self, world):
        """
        Inputs:
            world: the world for this data (must contain a predictor)
        Returns: the current depths, images, and extrinsics as torch.tensors
            images: b, 3, h, w
            
            depths: b, 1, h, w
            
            extrinsics: b, 4, 4
        """
        depths = np.stack(self.depths)
        image_tensor  = world.predictor.image_list_to_tensor(self.images)
        extrinsics = np.stack(self.extrinsics)

        depth_tensor = torch.from_numpy(depths).to(self.device)
        depth_tensor = depth_tensor.unsqueeze(1)
        extr_tensor = torch.from_numpy(extrinsics).float().to(self.device)

        return image_tensor, depth_tensor, extr_tensor