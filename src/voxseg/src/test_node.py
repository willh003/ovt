#!/home/pcgta/mambaforge/envs/ovseg/bin/python

import rospy
from std_msgs.msg import Int32
from voxseg.msg import TransformationMatrix
from modules.client import VoxSegClient
import numpy as np

def test_send_images(client: VoxSegClient):
    rate = rospy.Rate(hz=5)
    for i in range(5):
        image = np.random.random((540,720,3)) * 256
        depths = np.random.random((540,720))
        extrinsics = np.random.random((4,4))
        client.publish_depth_image(image, depths, extrinsics)
        #client.request_voxel_computation()
        rate.sleep()

def test_send_classes(client: VoxSegClient):
    client.publish_class_names(['traversable', 'untraversable'])

def test_request_computation(client: VoxSegClient):
    client.request_voxel_computation()

def test_all(client: VoxSegClient):
    test_send_images(client)
    test_send_classes(client)
    test_request_computation(client)
    rospy.spin()

if __name__ == '__main__':
    client = VoxSegClient()
    test_all(client)
