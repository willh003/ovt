#!/home/pcgta/mambaforge/envs/ovseg/bin/python

import rospy
from modules.rviz import MarkerPublisher, test_publish_markers

if __name__=='__main__':
    try:
        marker_pub = MarkerPublisher()
    except rospy.ROSInterruptException:
        pass