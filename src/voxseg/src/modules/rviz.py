#!/home/sean/mambaforge/envs/ovseg/bin/python

import rospy
#from costmap_2d.msg import VoxelGrid
from visualization_msgs.msg import Marker, MarkerArray
from voxseg.msg import Classes, VoxelGrid
import matplotlib.cm as cm
import torch
import json
from modules.utils import get_ros_markers, voxels_from_msg, convert_dictionary_array_to_dict
from modules.config import VOXEL_TOPIC, CLASS_TOPIC, RVIZ_NODE, MARKER_TOPIC


class MarkerPublisher:
    def __init__(self):
        self.classes = []
        
        rospy.init_node(RVIZ_NODE, anonymous=True)
        rospy.Subscriber(VOXEL_TOPIC, VoxelGrid, self._publish_markers_callback)
        rospy.Subscriber(CLASS_TOPIC, Classes, self._class_name_callback)
        self.pub = rospy.Publisher(MARKER_TOPIC, MarkerArray, queue_size=1)

        print('RViz Communication Layer Setup')

        rospy.spin() # not sure if this will be needed here

    def _publish_markers_callback(self, msg):
        voxels, world_dim = voxels_from_msg(msg)
        markers = get_ros_markers(voxels, world_dim, self.classes)
        
        self.pub.publish(markers)
        print('Published Markers')

    def _class_name_callback(self, msg):
        """
        Visualize the groups
        """
        groups = convert_dictionary_array_to_dict(msg.groups)

        self.classes = [str(key) for key in groups.keys()]

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
        grid=get_ros_markers(voxels,world_dim)

        publisher.publish(grid)
        rate.sleep()


if __name__=='__main__':
    try:
        test_publish_markers('voxel_grid_array')
    except rospy.ROSInterruptException:
        pass


