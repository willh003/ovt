#!/home/pcgta/mambaforge/envs/ovseg/bin/python

import rospy
from costmap_2d.msg import VoxelGrid
from geometry_msgs.msg import Point32, Vector3, Point, Quaternion, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Header
import matplotlib.cm as cm
import torch


def get_ros_voxels(voxels : torch.Tensor, world_dim: torch.Tensor) -> MarkerArray:
    """
    Inputs:
        voxels: shape (x,y,z), containing values in [0, n), where n represents the number of classes
        world_dim: shape (3)

    Returns:
        a MarkerArray containing markers for each class and corresponding colors
    """
    

    size_x, size_y, size_z = voxels.size()
    world_x, world_y, world_z = world_dim[0], world_dim[1], world_dim[2]
    resolution = torch.Tensor([world_x/size_x, world_y/size_y, world_z/size_z])

    colormap = cm.get_cmap('turbo') 
    voxel_classes_scaled = voxels / voxels.max()

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

    return grid


def publish_markers(markers: MarkerArray,topic: str = 'voxel_grid_array', publish_rate=1):
    """
    Inputs:
        topic: the topic to publish the markers to

        markers: an array of markers to render. Rviz gets laggy around 100x100x50 (below 1 fps) 
        
        publish_rate: the rate (hz) at which to publish
    
    """
    publisher = rospy.Publisher(topic, MarkerArray, queue_size=1)
    rospy.init_node('voxel_grid_publisher', anonymous=True)
    publisher.publish(markers)

def test_publish_markers(topic):

    publisher = rospy.Publisher(topic, MarkerArray, queue_size=1)
    rospy.init_node('voxel_grid_publisher', anonymous=True)
    rate = rospy.Rate(hz=1)
    while not rospy.is_shutdown():
        density = .2
        voxels = torch.rand(50,50,50)
        display_mask = voxels < (1-density)
        voxels[display_mask] = -1 # basically a binomial
        voxels[~display_mask] *= 5

        world_dim=torch.Tensor([50,50,50])
        grid=get_ros_voxels(voxels,world_dim)

        publisher.publish(grid)
        rate.sleep()



if __name__=='__main__':


    try:
        test_publish_markers('voxel_grid_array')
    except rospy.ROSInterruptException:
        pass


