#!/usr/bin/env python

from modules.client import VoxSegClient
import rospy

def request_computation(client: VoxSegClient):
    client.request_voxel_computation()
    
if __name__ == '__main__':
    client = VoxSegClient()
    request_computation(client)