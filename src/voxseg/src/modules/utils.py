import torch
import torch.nn.functional as F
import os
from PIL import Image
from detectron2.data.detection_utils import read_image

# python imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # NOTE: Do not get rid of this, the import modifies the matplotlib module. It's stupid.
import matplotlib.cm as cm
from matplotlib.pyplot import figure
from matplotlib.patches import Patch


def load_images(directory):
    image_files = sorted([f for f in os.listdir(directory) if f.startswith('img_') and f.endswith('.jpg')])
    depth_files = sorted([f for f in os.listdir(directory) if f.startswith('depth_') and f.endswith('.pt')])
    cam_loc_files = sorted([f for f in os.listdir(directory) if f.startswith('cam_loc_') and f.endswith('.pt')])

    images = []
    depth_tensors = []
    cam_loc_tensors = []

    for img_file, depth_file, cam_loc_file in zip(image_files, depth_files, cam_loc_files):
        # Load image into PIL
        img_path = os.path.join(directory, img_file)
        image = read_image(img_path, format="BGR")
        images.append(image)

        # Load depth tensor
        depth_path = os.path.join(directory, depth_file)
        depth_tensor = torch.load(depth_path)
        depth_tensors.append(depth_tensor)

        # Load camera location tensor
        cam_loc_path = os.path.join(directory, cam_loc_file)
        cam_loc_tensor = torch.load(cam_loc_path)
        cam_loc_tensors.append(cam_loc_tensor)

    if len(depth_tensors) == 1:
        depths = depth_tensors[0][None]
        cam_locs = cam_loc_tensors[0][None]
    else:
        # Convert the lists into a single tensor
        depths = torch.stack(depth_tensors)
        cam_locs = torch.stack(cam_loc_tensors)

    return images, depths, cam_locs



def update_grids(feature_map, voxels, feature_grid, grid_count):
    """
    Inputs:
        feature_map: (batch, network img height, network img width, feature size)

        voxels: (batch, original img height, original img width, 3)

        feature_grid: (self.grid_dim, feature size)

        grid_count: (self.grid_dim)
    Returns:
        feature_grid, grid_count after updating the selected voxels with the new features

    """

    batch, original_height, original_width, _ = voxels.size()
    feature_map_upsampled = interpolate_features(feature_map, original_height, original_width)

    _, new_h, new_w, _ = feature_map_upsampled.size()

    for i in range(batch):
        xv = voxels[i, :, :, 0].long()
        yv = voxels[i, :, :, 1].long()
        zv = voxels[i, :, :, 2].long()        
        assert 0 <= xv.min() and xv.max() < feature_grid.size()[0]
        assert 0 <= yv.min() and yv.max() < feature_grid.size()[1]
        assert 0 <= zv.min() and zv.max() < feature_grid.size()[2]

        # NOTE: this is technically wrong (duplicates are not considered, so it will just add one of the features)
        feature_grid[xv, yv, zv] += feature_map_upsampled[i]
        grid_count[xv, yv, zv] += torch.ones((new_h, new_w)).cuda()


        # NOTE: the below runs it for every single pixel. It is super slow
        # stack = torch.stack((xv.flatten(),yv.flatten(),zv.flatten())).T
        # g = torch.zeros_like(grid_count)
        # for row in stack:
        #     g[row[0],row[1],row[2]] += 1
    
    return feature_grid, grid_count

def interpolate_features(feature_map, new_height, new_width):
    """
    Inputs:
        feature_map: torch.tensor (batch, network img height, network img width, feature size)
    
    Returns:
        feature_map down or upsampled to have size (batch, new_height, new_width, feature_size)
    """

    input = feature_map.permute(0, 3, 1, 2)

    # large-ish batch sizes (>7, for img dim 540x720) result in
    # RuntimeError: upsample_nearest_nhwc only supports output tensors with less than INT_MAX elements
    # this is a hacky workaround
    if len(feature_map) > 7:  
        batch, _, _, features = feature_map.size()
        output = torch.zeros((batch, features, new_height, new_width), device=feature_map.device)
        for i in range(len(input)): 
            output[i] = F.interpolate(input[i][None], size=(new_height, new_width), mode='nearest').squeeze()
        return output.permute(0,2, 3, 1)
    else:
        output = F.interpolate(input, size=(new_height, new_width), mode='nearest')
        return output.permute(0,2, 3, 1)

def get_all_pixels(numrows, numcols):
    """
    Inputs:
        numrows, numcols: rows and columns in the image

    Returns:
        torch.tensor, of shape (numrows*numcols, 3), containing homogenous coordinates for each pixel

    """

    rows, cols = torch.meshgrid((torch.arange(numrows), torch.arange(numcols)),indexing='ij') # stride flat pixels
    pixels = torch.stack((cols,rows),2).float()

    # Reshape to 1D arrays
    oned_pixels = pixels.reshape(-1, 2)
    ones = torch.ones_like(oned_pixels)
    homo_pixels = torch.concat((oned_pixels,ones),1).cuda()
    return homo_pixels



def expand_to_batch_size(tensor, batches):
    """
    Inputs:
        tensor: shape S

        batches: int

    Returns:
        torch.tensor, of shape (batches, *S)
    """

    return torch.repeat_interleave(tensor[None], batches, dim=0) 


def unproject(intrinsics, extrinsics, pixels, depth, max_depth = 1000):
    Kinv = torch.inverse(intrinsics)
    # get pixel coordinates in the camera plane
    cam_pts = (Kinv @ pixels.T).T
    # normalize camera coordinates (plane -> sphere)
    cam_pts_norm = cam_pts / (torch.norm(cam_pts[:, :3], dim=1)[None].T)

    batches, height, width = depth.size()
    all_wld_pts = torch.zeros((batches, 3, height, width)).cuda()

    # for each image
    for i in range(len(extrinsics)):
        flat_depth = depth[i].flatten()[None]

        # scale cam coordinates with depth (raycast them out)
        cam_pts_depth = (cam_pts_norm.T * flat_depth).T
        cam_pts_depth[:,3] = 1  # ensure homogeneous coordinates

        wld_pts = (extrinsics[i].cuda() @ cam_pts_depth.T).T  # Transform camera coordinates to world coordinates
        
        wld_pts_unhomo = wld_pts[:, :3]

        all_wld_pts[i] = wld_pts_unhomo.permute(1,0).view(3, height, width)

    return all_wld_pts



def visualize_voxel_classes(voxels, classes, save_dir=None, base_name='default'):
    np_voxels = voxels.to(device='cpu').numpy() + 1 # put them in range [0, len(classes)]
    voxel_classes_scaled = np_voxels / np_voxels.max() # for the colormap, put them in [0,1]
    occupied_voxels = voxel_classes_scaled > 0 # default class is -1, so do not show it

    colors = np.empty(occupied_voxels.shape, dtype=object)

    colormap = cm.get_cmap('turbo')
    for i in range(voxel_classes_scaled.shape[0]):
        for j in range(voxel_classes_scaled.shape[1]):
            for k in range(voxel_classes_scaled.shape[2]):
                if occupied_voxels[i, j, k]:
                    colors[i, j, k] = colormap(voxel_classes_scaled[i, j, k])

    fig = figure()
    fig.add_subplot(projection='3d')
    ax = fig.gca()
    ax.voxels(occupied_voxels, facecolors=colors, edgecolor='k')
    class_idx = range(len(classes))

    legend_elements = [Patch(facecolor=colormap((c+1) / len(class_idx)), edgecolor='k', label=f'Class {classes[c]}') for c in class_idx]
    ax.legend(handles=legend_elements)

    if save_dir:
        filename = base_name + "_".join(classes) + '.png'
        path = os.path.join(save_dir, filename)
        plt.savefig(path)

    plt.show()

def visualize_voxel_occupancy(vxw):
    occupancy = vxw.grid_count.to(device='cpu').numpy()

    occupancy_norm = occupancy / occupancy.max()

    occupied_voxels = (occupancy_norm > 0)

    colors = np.empty(occupied_voxels.shape, dtype=object)
    colormap = cm.get_cmap('turbo')
    for i in range(occupancy_norm.shape[0]):
        for j in range(occupancy_norm.shape[1]):
            for k in range(occupancy_norm.shape[2]):
                if occupied_voxels[i, j, k]:
                    colors[i, j, k] = colormap(occupancy_norm[i, j, k])

    fig = figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(occupied_voxels, facecolors=colors, edgecolor='k')

    # Create a colorbar with labels showing the range of values
    cbar = plt.colorbar(cm.ScalarMappable(cmap=colormap), ax=ax, shrink=0.6)
    cbar.set_label('Occupancy Range')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f"{occupancy.min():.2f}", f"{occupancy.max():.2f}"])

    plt.show()

