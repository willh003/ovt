#!/usr/bin/env python

import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import struct
import ctypes

def pcl_callback(msg):
    data = np.frombuffer(msg.data, dtype=np.int8)

    xyz = np.array([[0,0,0]])
    rgb = np.array([[0,0,0]])
    gen = pc2.read_points(msg,field_names=['x','y','z'], skip_nans=True)
    int_data=  list(gen)
        
    for x in int_data:
        # prints r,g,b values in the 0-255 range
                    # x,y,z can be retrieved from the x[0],x[1],x[2]
        xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)

    breakpoint()

def main():
    rospy.init_node('rosbage_listener_node', anonymous=True)
    rospy.Subscriber("compslam_lio/map", pc2.PointCloud2, pcl_callback)
    rospy.spin()

if __name__ == "__main__":
    main()

