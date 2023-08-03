#!/usr/bin/env python
import rospy
from modules.robot_data_interface import RobotDataInterface

if __name__ == '__main__':
    RobotDataInterface()
    rospy.loginfo("Set up (rgb,depth) image publisher node")
    rospy.spin()
