#!/usr/bin/env python

import sys
import unittest
import rospy
import roslaunch
import os
import rospkg
from std_msgs.msg import Int32
from modules.client import VoxSegClient
from modules.utils import load_images
from send_classes import send_classes
import numpy as np


class TestCase(unittest.TestCase):
    def setUp(self):
        self.client = VoxSegClient()

    def test_1(self):
        
        rate = rospy.Rate(hz=5)
        image = np.random.random((540,720, 3))
        depth = np.random.random((540,720))
        extrinsics = np.random.random((4,4))

        self.client.publish_depth_image(image, depth, extrinsics)


    def test_2(self):
        class_names = ['equipment', 'fire hydrant', 'rocks', 'bricks', 'ground']
        send_classes(self.client, class_names, groups=None)
    
    def test_5(self):
        class_names = ['equipment', 'fire hydrant', 'rocks', 'bricks', 'ground']
        groups = {'terrain':['rocks', 'bricks', 'ground'], 'items':['fire hydrant', 'equipment']}

        send_classes(self.client, class_names, groups=groups)

    def test_3(self):
        prompts = {'machinery': ['bulldozer', 'backhoe', 'heavy machinery', 'machinery'],
                'equipment': ['barrel', 'crate', 'tarp'],
                    'detritus': ['rocks', 'detritus', 'bricks'],
                    'fire hydrants': ['fire hydrant'],
                    'ground': ['ground']
        }

        send_classes(self.client, prompts, groups=None)


    def test_4(self):
        groups = {'terrain':['detritus', 'ground'], 'items':['fire hydrants', 'equipment', 'machinery']}
        prompts = {'machinery': ['bulldozer', 'backhoe', 'heavy machinery', 'machinery'],
                'equipment': ['barrel', 'crate', 'tarp'],
                    'detritus': ['rocks', 'detritus', 'bricks'],
                    'fire hydrants': ['fire hydrant'],
                    'ground': ['ground']
        }
        
        send_classes(self.client, prompts, groups=groups)

def launch_from_file(launch_file):
    rospack = rospkg.RosPack()
    launch_file_path = os.path.join(rospack.get_path('voxseg'),'launch',launch_file)
    roslaunch_process = roslaunch.parent.ROSLaunchParent(
    rospy.get_param("/run_id", default="test_run"),
    [launch_file_path],
    )

    roslaunch_process.start()
    rospy.sleep(15)
    return roslaunch_process


if __name__ == '__main__':
    
    server_process = launch_from_file('server.launch')

    # Perform your tests here using the nodes started by the launch file
    # ...
    PKG = 'voxseg'
    import rosunit
    rosunit.unitrun(PKG, 'test_bare_bones', TestCase)

    compute_process = launch_from_file('request_computation.launch')
    compute_process.shutdown()
    # Stop the launch file after the tests are complete
    server_process.shutdown()

    