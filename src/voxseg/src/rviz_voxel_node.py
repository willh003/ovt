#!/home/pcgta/mambaforge/envs/ovseg/bin/python

import rospy
from costmap_2d.msg import VoxelGrid
from visualization_msgs.msg import Marker, MarkerArray
from modules.publish_markers import test_publish_markers

def init_node():

    # Initialize the ROS node
    rospy.init_node('rviz_voxel_node', anonymous=True)
    # Spin to keep the node running

    rospy.Publisher('voxel_grid_array', MarkerArray, queue_size=1)

    rospy.spin()

if __name__=='__main__':
        # you could name this function
    try:
        test_publish_markers('voxel_grid_array')
        #init_node()
    except rospy.ROSInterruptException:
        pass