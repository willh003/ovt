#!/home/pcgta/.local/share/ov/pkg/isaac_sim-2022.2.0/python.sh
import rospy
from costmap_2d.msg import VoxelGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int32MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image
from voxseg.msg import DepthImageInfo, TransformationMatrix, Classes
from voxseg.srv import VoxelComputation

from cv_bridge import CvBridge
import numpy as np
from typing import List

from modules.config import *


class VoxSegClient:
    def __init__(self):
        """
        initialize the frontend node
        """

        rospy.init_node(CLIENT_NODE, anonymous=True)

    def publish_depth_image(self, image, depth_map, extrinsics):
        """
        Should be called alongside image_callback in the simulation
        Inputs:
            image: a numpy array containing rgb image data, shape (h,w,c)

        """
        timestamp = rospy.Time.now()
        image_msg = self._get_image_msg(image, timestamp)
        depth_msg = self._get_depth_msg(depth_map, timestamp)
        cam_msg = self._get_cam_msg(extrinsics, timestamp)

        full_msg = DepthImageInfo()
        full_msg.rgb_image = image_msg
        full_msg.depth_image = depth_msg
        full_msg.cam_extrinsics = cam_msg

        
        pub = rospy.Publisher(IMAGE_TOPIC, DepthImageInfo, queue_size=10)
        pub.publish(full_msg)

    def publish_class_names(self, names: List[str]):
        """
        Should be called whenever the user enters class names in the extension window
        names: list of class names
        """
        class_msg = Classes()
        class_msg.classes = names

        pub = rospy.Publisher(CLASS_TOPIC, DepthImageInfo, queue_size=10)
        pub.publish(class_msg)

    def request_voxel_computation(self):
        rospy.wait_for_service('compute_data')
        try:
            compute_data_service = rospy.ServiceProxy(VOXEL_REQUEST_SERVICE, VoxelComputation)
            response = compute_data_service()
            return response.result
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
            return None

    def _get_image_msg(self, image, timestamp) -> Image:
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


    def _get_depth_msg(self, depth_map, timestamp) -> Image:
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


    def _get_cam_msg(self, extrinsics, timestamp) -> TransformationMatrix:
        """
        extrinsics: a numpy array containing camera extrinsics, size (4,4)
        """
        dim_row = MultiArrayDimension(label="row", size=4, stride=4*4)  # Stride is not needed for contiguous arrays
        dim_col = MultiArrayDimension(label="col", size=4, stride=4)
        layout = MultiArrayLayout()
        layout.dim = [dim_row, dim_col]
        layout.data_offset = 0

        cam_msg = TransformationMatrix()
        cam_msg.layout = layout
        cam_msg.matrix = np.reshape(extrinsics, (16,)).tolist()

        return cam_msg



if __name__=='__main__':
    try:
        frontend = VoxSegClient()
    except rospy.ROSInterruptException:
        pass