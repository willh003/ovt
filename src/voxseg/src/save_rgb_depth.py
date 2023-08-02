#!/usr/bin/env python
import rospy
from rospy import Publisher
from rospy import Subscriber as RospySubscriber
from sensor_msgs.msg import CompressedImage, Image
from tf2_msgs.msg import TFMessage
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber as SyncedSubscriber
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from modules.data import UnalignedData
from modules.config import BATCH_SIZE, IMAGE_TOPIC
from geometry_msgs.msg import TransformStamped 
from voxseg.msg import DepthImageInfo
from modules.utils import get_depth_msg, get_image_msg, get_cam_msg
import yaml

# Method cutout from Wild Visual Navigation--------------------#
from typing import List, Sequence
from liegroups import SE3, SO3
def transformation_matrix_of_pose(pose : Sequence[float]):
    """Convert a translation and rotation into a 4x4 transformation matrix.
 
    Args:
        pose (float subscriptable @ 0..6): A 7 element array representing the pose [tx,ty,tz,q0,qx,qy,qz,qw].

    Returns:
        4x4 Transformation Matrix \in SE3 following the ordering convention specified"""
    quat = np.array(pose[3:])
    quat = quat / np.linalg.norm(quat)
    matrix = SE3(rot=SO3.from_quaternion(quat, ordering='xyzw'), trans=pose[:3]).as_matrix() # Check order (wxyz looks correct for orbit footpath)
    matrix = matrix.astype(np.float32)
    return matrix
#--------------------------------------------------------------#

def pose_of_tf(tf : TransformStamped) -> np.ndarray:
    tx = tf.transform.translation.x
    ty = tf.transform.translation.y
    tz = tf.transform.translation.z
    qx = tf.transform.rotation.x
    qy = tf.transform.rotation.y
    qz = tf.transform.rotation.z
    qw = tf.transform.rotation.w
    return np.array([tx,ty,tz,qx,qy,qz,qw])


class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        
        rospy.init_node('image_saver_node')

        # TODO: Pull topic names from some config

        # Cannot just use tf_sub as it has no header...
        # instead we need to add an intermediate sub/pub trio. 
        self.tf_main_sub = RospySubscriber("/tf", TFMessage, callback=self.publish_tf_list_to_specific_tfs)

        # Create subscribers for the image topics
        self.rgb_sub = SyncedSubscriber("/wide_angle_camera_front/image_color_rect/compressed", CompressedImage)
        self.depth_sub = SyncedSubscriber("/depth_camera_front_upper/depth/image_rect_raw", Image)

        # Create subscribers for the tf topics
        self.tf_odom_sub = SyncedSubscriber("/tf_odom", TransformStamped)
        self.tf_rgb_sub = SyncedSubscriber("/tf_rgb", TransformStamped)
        self.tf_depth_sub = SyncedSubscriber("/tf_depth", TransformStamped)

        # Publishers for the individual tf topics
        self.tf_odom_pub = Publisher("/tf_odom", TransformStamped, queue_size=10) # NOTE: Arbitrary number
        self.tf_rgb_pub = Publisher("/tf_rgb", TransformStamped, queue_size=10)
        self.tf_depth_pub = Publisher("/tf_depth", TransformStamped, queue_size=10)

        self.data_pub = Publisher(IMAGE_TOPIC, DepthImageInfo, queue_size=10)

        # Synchronize the topics
        ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, 
                                           self.tf_odom_sub, self.tf_rgb_sub, self.tf_depth_sub], 
                                           slop=0.1, queue_size=10) # NOTE: 0.1 default, 10 Arbitrary number
        ats.registerCallback(self.callback)
        rospy.spin()
    
    def publish_tf_list_to_specific_tfs(self, tf_msg):
        # get camera tfs
        tfs_list : List[TransformStamped] = tf_msg.transforms

        # find transforms of interest        
        rgb_frame_id = "wide_angle_camera_front_camera_parent" #rgb_img
        depth_frame_id = "depth_camera_front_upper_depth_optical_frame" #depth_img
        base_frame_id = "base"

        for tf in tfs_list:
            if tf.child_frame_id == base_frame_id:
                self.tf_odom_pub.publish(tf)
            elif tf.child_frame_id == rgb_frame_id:
                self.tf_rgb_pub.publish(tf)
            elif tf.child_frame_id == depth_frame_id:
                self.tf_depth_pub.publish(tf)

    def callback(self,
                 input_rgb_image: CompressedImage, 
                 input_depth_image: Image, 
                 odom_transform: TransformStamped, 
                 rgb_transform: TransformStamped, 
                 depth_transform: TransformStamped):
        try:
            # Convert compressed RGB image to OpenCV format and then to numpy array
            decoded_rgb_image = self.bridge.compressed_imgmsg_to_cv2(input_rgb_image, desired_encoding="bgr8")
            rgb_image_array = np.array(decoded_rgb_image)

            # Convert depth image to OpenCV format and then to numpy array
            decoded_depth_image = self.bridge.imgmsg_to_cv2(input_depth_image, desired_encoding="passthrough")
            decoded_depth_image = cv2.rotate(decoded_depth_image, cv2.ROTATE_180)
            depth_image_array = np.array(decoded_depth_image)
            
            # Extract transformation matrices from input transformations
            odom_to_base_transform = transformation_matrix_of_pose(pose_of_tf(odom_transform))
            base_to_rgb_transform = transformation_matrix_of_pose(pose_of_tf(rgb_transform))
            base_to_depth_transform = transformation_matrix_of_pose(pose_of_tf(depth_transform))

            # Compute global transformations for RGB and depth images
            global_rgb_transform = (odom_to_base_transform @ base_to_rgb_transform).flatten()
            global_depth_transform = (odom_to_base_transform @ base_to_depth_transform).flatten()

            # Create message objects from the numpy arrays and transformations
            rgb_image_message = get_image_msg(rgb_image_array, input_depth_image.header.stamp)
            depth_image_message = get_depth_msg(depth_image_array, input_depth_image.header.stamp)
            rgb_transform_message = get_cam_msg(global_rgb_transform)
            depth_transform_message = get_cam_msg(global_depth_transform)

            # Publish the extracted and processed information
            self.data_pub.publish(DepthImageInfo(rgb_image=rgb_image_message, 
                                                depth_image=depth_image_message, 
                                                cam_extrinsics=rgb_transform_message, 
                                                depth_extrinsics=depth_transform_message))
            
            rospy.loginfo("Saved synchronized images with timestamp: %s", input_depth_image.header.stamp)

        except Exception as e:
            # Consider adding some error logging here
            rospy.logerr(f"Error processing images: {str(e)}")


if __name__ == '__main__':
    rospy.loginfo("Set up (rgb,depth) image saver node")
    ImageSaver()
    rospy.spin()
