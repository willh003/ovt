import torch
import torch.nn.functional as F

import os
from PIL import Image
from detectron2.data.detection_utils import read_image
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Vector3, Point, Quaternion, Pose
from voxseg.srv import VoxelComputationResponse
#from costmap_2d.msg import VoxelGrid

# python imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # NOTE: Do not get rid of this, the import modifies the matplotlib module. It's stupid.
import matplotlib.cm as cm
from matplotlib.pyplot import figure
from matplotlib.patches import Patch

from std_msgs.msg import String
import json
import rospy



def get_image_msg(image, timestamp) -> Image:
    """
    Inputs:
        image: a numpy array containing rgb image data, shape (h,w,c)

        depth_map: a numpy array containing depth data, size (h,w)

        extrinsics: a numpy array containing camera extrinsics, size (4,4)

    """
    h,w,c = image.shape
    img_msg = Image()
    img_msg.width = w
    img_msg.height = h
    img_msg.encoding = "rgb8"  # Set the encoding to match your image format
    img_msg.data = image.tobytes()
    img_msg.header.stamp = timestamp
    img_msg.header.frame_id = 'img_frame'

    return img_msg


def get_depth_msg(depth_map, timestamp) -> Image:
    """
    depth_map: a numpy array containing depth data, size (h,w)
    """
    h,w = depth_map.shape
    depth_msg = Image()
    depth_msg.height = h
    depth_msg.width = w
    depth_msg.encoding = '32FC1'  # Assuming single-channel depth map
    depth_msg.step = w * 4  # Size of each row in bytes
    depth_msg.data = depth_map.astype(np.float32).tobytes()
    depth_msg.header.stamp = timestamp
    depth_msg.header.frame_id = 'depth_frame'

    return depth_msg

def load_images(directory):
    """
    Returns: 
        images: list[np.ndarray, shape (h, w, c)]

        depth: torch.tensor (b, 1, h, w)

        cam_locs: torch.tensor (b, 4, 4)

        (for each file titled img_*, depth_*, cam_loc_* in directory)
    """
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

def update_grids_aligned(feature_map, voxels, feature_grid, grid_count):
    """
    Inputs:
        feature_map: (batch, N, feature size), representing features to insert into voxels
        
        voxels: (batch, N, 3), representing voxel indices
        
        feature_grid: (self.grid_dim, feature size)

        grid_count: (self.grid_dim)

    Returns:
        feature_grid, grid_count after updating the selected voxels with the new features
    """

    B, N, _ = feature_map.size()

    # for each batch
    for i in range(B):
        xv = voxels[i, :, 0].long()
        yv = voxels[i, :, 1].long()
        zv = voxels[i, :, 2].long() 

        feature_grid[xv, yv, zv] += feature_map[i]
        grid_count[xv, yv, zv] += torch.ones(N).cuda()

    return feature_grid, grid_count

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

def align_depth_to_rgb(rgb, K_rgb, T_rgb, d, K_d, T_d):
    """
    Inputs:
        rgb: torch.Tensor, (B, 3, h, w)
        K_rgb: (B, 4, 4), rgb camera intrinsics
        T_rgb: (B, 4, 4), rgb camera extrinsics
        d: torch.Tensor, (B, 1, h, w)
        K_d: (B, 4, 4), depth camera intrinsics
        T_d: (B, 4, 4), depth camera extrinsics
    Returns:
        Two tensors, shape (B, M, 3) and (B, M, 2). The first tensor contains world coords of the depths,
        and the second contains the pixel coordinates in rgb image space for those depths.
    """
    h_d, w_d = d.size()
    h_rgb, w_rgb, _ = rgb.size()
    pixels_d = get_all_pixels(h_d, w_d)
    d_in_wld = unproject(K_d, T_d, pixels_d, d)

    d_in_rgb = project(K_rgb, T_rgb, d_in_wld)
    boundary_mask = (d_in_rgb[:,:,0] < h_rgb) & (d_in_rgb[:,:,0] > 0) & (d_in_rgb[:,:,1] < w_rgb) & (d_in_rgb[:,:,1] > 0)
    
    d_px_in_rgb = d_in_rgb[boundary_mask].floor().long()
    d_in_wld_clean = d_in_wld[boundary_mask].nan_to_num().permute(0,2,3,1)

    return d_in_wld_clean, d_px_in_rgb
    
        

def project(intrinsics, extrinsics, points):
    """
    Inputs:
        intrinsics: Bx4x4
        extrinsics: Bx4x4
        points: Bxnx4
    Returns:
        torch.LongTensor, Bxnx2, coordinates in image space
    """
    image_pts= (intrinsics @ torch.inverse(extrinsics) @ points)

    image_pts_normalized = image_pts / image_pts[:, 2, :] # divide by homogenous
    px= torch.floor(image_pts_normalized).long()
    px = px[:, :2, :].permute(0, 2, 1)

    return px

def unproject(intrinsics, extrinsics, pixels, depth):
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

############### ROS ##################

def convert_dict_to_dictionary_array(dictionary):

    # Convert the Python dictionary to serialized JSON strings
    dictionary_jsons = [json.dumps({key: value}) for key, value in dictionary.items()]

    # Convert the serialized JSON strings to ROS String messages
    dictionary_json_msgs = [str(json_str) for json_str in dictionary_jsons]


    return dictionary_json_msgs

def convert_dictionary_array_to_dict(dictionary_jsons):
    # Create an empty dictionary to store the result
    result_dict = {}

    # Deserialize the JSON strings and populate the result dictionary
    for json_str in dictionary_jsons:
        try:
            data = json.loads(json_str)
            result_dict.update(data)
        except ValueError as e:
            rospy.logerr("Error while deserializing JSON string: %s", str(e))

    return result_dict

def voxels_from_srv(msg: VoxelComputationResponse):
    """
    Given a VoxelComputationRespons msg, extract the voxel structure containing classes
    Returns:
        voxels: torch.tensor, shape grid_dim, with values representing classes. NOTE: the first specified class is 0, and the empty class is -1.
        world_dim: torch.tensor, shape(3,), represents the xyz dimensions of the world
    """
    voxels: torch.Tensor = torch.as_tensor(msg.data).float()
    voxels -= 1 # the backend adds 1 to the voxel classes, in order to encode the array in bytes
    voxel_grid_shape = (msg.size_x, msg.size_y, msg.size_z)
    voxels = voxels.view(*voxel_grid_shape)

    resolutions = torch.as_tensor([msg.resolutions.x, msg.resolutions.y, msg.resolutions.z])
    grid_dim = torch.as_tensor(tuple(voxels.size()))
    world_dim = grid_dim / resolutions
    return voxels, world_dim

################ Visualization ################

def get_ros_markers(voxels : torch.Tensor, world_dim: torch.Tensor, classes=['other']) -> MarkerArray:
    """
    Inputs:
        voxels: shape (x,y,z), containing values in [0, n), where n represents the number of classes
        world_dim: shape (3)
        classes: list of classes

    Returns:
        a MarkerArray containing markers for each class and corresponding colors
    """
    

    size_x, size_y, size_z = voxels.size()
    world_x, world_y, world_z = world_dim[0], world_dim[1], world_dim[2]
    resolution = torch.Tensor([world_x/size_x, world_y/size_y, world_z/size_z])

    colormap = cm.get_cmap('turbo') 
    num_classes = len(classes)
    all_colors = {}

    voxel_classes_scaled = voxels / num_classes

    grid = MarkerArray()
    count = 0
    for i in range(size_x):
        for j in range(size_y):
            for k in range(size_z):
                value = voxel_classes_scaled[i,j,k]
                if value >= 0: # -1 indicates the voxel is empty
                    loc = Point(x=i*resolution[0], y=j*resolution[1], z=k*resolution[2])
                    quat = Quaternion(x=0,y=0,z=0,w=1)
                    pose_msg = Pose(position = loc, orientation=quat)

                    color = colormap(value)

                    # hacky workaround to get the legend to work
                    # sets are unordered, but order matters for the colors
                    all_colors[int(voxels[i,j,k].item())] = color

                    color_msg = ColorRGBA(r=color[0],g=color[1],b=color[2],a=color[3])
                    marker = Marker()
                    marker.header.frame_id='world'
                    marker.id=count
                    count+=1
                    marker.color=color_msg
                    marker.pose=pose_msg
                    marker.type=marker.CUBE
                   # marker.lifetime = rospy.Duration(secs=10)
                    marker.scale = Vector3(x=resolution[0],y=resolution[1],z=resolution[2])
                    grid.markers.append(marker)

    # Create Markers for Legend
    for i in all_colors.keys():
        loc = Point(x=0, y=0, z=i*2+5)
        quat = Quaternion(x=0,y=0,z=0,w=1)
        pose_msg = Pose(position = loc, orientation=quat)

        color = all_colors[i]
        color_msg = ColorRGBA(r=color[0],g=color[1],b=color[2],a=color[3])
        text_marker = Marker() 
        text_marker.type = marker.TEXT_VIEW_FACING
        text_marker.header.frame_id='world'
        text_marker.id=count
        text_marker.color=color_msg
        text_marker.pose=pose_msg
        text_marker.text= f"Class {int(i)+1}: {classes[int(i)] if i < len(classes) else 'undefined'}"
        text_marker.scale = Vector3(x=1.5,y=1.5,z=1.5)
        grid.markers.append(text_marker)
        count+=1

    return grid


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

