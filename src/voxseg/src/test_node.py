#!/home/pcgta/mambaforge/envs/ovseg/bin/python

import rospy
from std_msgs.msg import Int32
from voxseg.msg import TransformationMatrix
from modules.client import VoxSegClient
from modules.server import VoxSegServer
import numpy as np


def test_client_image(client: VoxSegClient):

    rate = rospy.Rate(hz=.2)
    while not rospy.is_shutdown():
        image = np.random.random((540,720,3)) * 256
        depths = np.random.random((540,720))
        extrinsics = np.random.random((4,4))

        client.publish_depth_image(image, depths, extrinsics)
        #client.request_voxel_computation()
        rate.sleep()


if __name__ == '__main__':
    client = VoxSegClient()
    test_client_image(client)
