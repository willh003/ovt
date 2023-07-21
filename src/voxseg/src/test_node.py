#!/home/pcgta/mambaforge/envs/ovseg/bin/python

import rospy
from std_msgs.msg import Int32
from voxseg.msg import TransformationMatrix
from modules.client import VoxSegClient
from modules.utils import load_images
import numpy as np

def test_send_images(client: VoxSegClient, image_dir=None):
    rate = rospy.Rate(hz=5)
    images, depths, cam_locs = load_images(image_dir)
    for i in range(len(images)):
        image = images[i]
        depths_np = depths[i].squeeze().cpu().numpy()
        extrinsics = cam_locs[i].cpu().numpy()
        
        client.publish_depth_image(image, depths_np, extrinsics)
        #client.request_voxel_computation()
        rate.sleep()

def test_send_classes(client: VoxSegClient):
    client.publish_class_names(['traversable', 'untraversable'])

def test_request_computation(client: VoxSegClient):
    client.request_voxel_computation()

def test_all(client: VoxSegClient):
    data_dir = '/home/pcgta/Documents/eth/wild_visual_navigation/wild_visual_navigation_orbit/feat-extract-out/test_16'
    test_send_images(client, data_dir)
    test_send_classes(client)
    test_request_computation(client)
    rospy.spin()
    

if __name__ == '__main__':
    client = VoxSegClient()
    test_all(client)
