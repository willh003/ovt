#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from voxseg.msg import TransformationMatrix
from modules.client import VoxSegClient
from modules.utils import load_images
import numpy as np

def send_images(client: VoxSegClient, image_dir=None):
    rate = rospy.Rate(hz=5)
    images, depths, cam_locs = load_images(image_dir)
    for i in range(5):
        image = images[i]

        depths_np = depths[i].squeeze().cpu().numpy()
        extrinsics = cam_locs[i].cpu().numpy()
        
        client.publish_depth_image(image, depths_np, extrinsics)
        #client.request_voxel_computation()
        rate.sleep()

    
def main():
    client = VoxSegClient()

    # Directory containing image_*.png, depth_*.pt, cam_loc_*.pt
    # test_16 contains the pics that actually worked!!!
    data_dir ='/home/pcgta/Documents/eth/wild_visual_navigation/wild_visual_navigation_orbit/feat-extract-out/test_26'
    send_images(client, data_dir)


if __name__ == '__main__':
    main()