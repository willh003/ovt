#!/usr/bin/env python

import rospy
from sensor_msgs.msg import CompressedImage, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        
        # Create subscribers for the image topics
        self.rgb_sub = Subscriber("/wide_angle_camera_front/image_color_rect/compressed", CompressedImage)
        self.depth_sub = Subscriber("/depth_camera_front_upper/depth/image_rect_raw", Image)
        
        # Synchronize the topics
        ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg):
        try:
            # Convert RGB compressed image to OpenCV format
            rgb_img = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            #cv2.imwrite(f'rgb_{rgb_msg.header.stamp}.jpg', rgb_img)
            rgb_img_np = np.array(rgb_img)
            
            # Convert depth image to OpenCV format
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            #cv2.imwrite(f'depth_{depth_msg.header.stamp}.png', depth_img)
            depth_img_np = np.array(depth_img)

            # TODO: Make this work!!!
            # self.data.add_depth_image(np_image, np_depths, extrinsics_2d)
            # print(f'Depth Image {len(self.data.all_images)} Received')
            
            
            rospy.loginfo("Saved synchronized images with timestamp: %s", rgb_msg.header.stamp)
            
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':

    rospy.init_node('image_saver_node')
    rospy.loginfo("Set up (rgb,depth) image saver node")
    ImageSaver()
    rospy.spin()
