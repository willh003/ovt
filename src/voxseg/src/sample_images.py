#!/usr/bin/env python
import rospy
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()

        self.skip_every = 5 #-1 sets it to save every
        self.batch_size = 66

        self.rgb_num = 0
        self.rgb_batchtick = 0
        self.rgb_batchnum = 0

        self.depth_num = 0
        self.depth_batchtick = 0
        self.depth_batchnum = 0

        rospy.init_node('image_sampler_node')

        # Create subscribers for the image topics
        #self.rgb_sub = rospy.Subscriber("/wide_angle_camera_front/image_color_rect/compressed", CompressedImage, callback=self.save_rgb)
        self.depth_sub = rospy.Subscriber("/depth_camera_front_upper/depth/image_rect_raw", Image, callback=self.save_depth)

        rospy.spin()

    def save_rgb(self, input_rgb_image: CompressedImage):
        if self.rgb_num < self.skip_every:
            self.rgb_num += 1
            return
        self.rgb_num=0

        if self.rgb_batchtick < self.batch_size:
            self.rgb_batchtick += 1
        else:
            self.rgb_batchtick = 0 
            self.rgb_batchnum += 1
        try:
            # Convert compressed RGB image to OpenCV format and then to numpy array
            decoded_rgb_image = self.bridge.compressed_imgmsg_to_cv2(input_rgb_image, desired_encoding="bgr8")
            cv2.imwrite(F"rgbout/rgb_{self.rgb_batchnum}_{input_rgb_image.header.stamp.secs}_{input_rgb_image.header.stamp.nsecs}.png", decoded_rgb_image)
            rospy.loginfo("Saved rgb, stamp@ %s", input_rgb_image.header.stamp.secs)
        except CvBridgeError as e:
            # Consider adding some error logging here
            rospy.logerr(f"Error processing images: {str(e)}")

    def save_depth(self, input_depth_image: Image):
        if self.depth_num < self.skip_every:
            self.depth_num += 1
            return
        self.depth_num=0

        if self.depth_batchtick < self.batch_size:
            self.depth_batchtick += 1
        else:
            self.depth_batchtick = 0 
            self.depth_batchnum += 1
        try:
            # Convert depth image to OpenCV format and then to numpy array
            decoded_depth_image = self.bridge.imgmsg_to_cv2(input_depth_image, desired_encoding="32FC1")
            decoded_depth_image = cv2.rotate(decoded_depth_image, cv2.ROTATE_180)
            depth_img_vis = (1 - 255*decoded_depth_image/decoded_depth_image.max()).astype('uint8')
            depth_img_vis = cv2.applyColorMap(depth_img_vis, cv2.COLORMAP_JET)

            cv2.imwrite(F"depthout/depth_{self.depth_batchnum}_{input_depth_image.header.stamp.secs}_{input_depth_image.header.stamp.nsecs}.jpg", depth_img_vis)
            rospy.loginfo("Saved depth, stamp@:%s", input_depth_image.header.stamp.secs)
        except CvBridgeError as e:
            # Consider adding some error logging here
            rospy.logerr(f"Error processing images: {str(e)}")

if __name__ == '__main__':
    rospy.loginfo("Set up (rgb,depth) image saver node")
    ImageSaver()
    rospy.spin()
