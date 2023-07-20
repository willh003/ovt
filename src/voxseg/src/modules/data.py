import torch

class BackendData:
    def __init__(self):
        self.images = []
        self.depths = []
        self.extrinsics = []
        self.classes = []

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
        Returns the depths, images, and extrinsics currently stored as torch tensors
        """
        depths = torch.stack(self.depths)
        images  = world.predictor.image_list_to_tensor(self.images)
        cam_locs = torch.stack(self.cam_locs)

        return images, depths, cam_locs