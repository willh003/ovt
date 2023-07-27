#!/usr/bin/env python

import rospy
from modules.client import VoxSegClient
from modules.utils import load_images


def send_images(client: VoxSegClient, image_dir=None):
    rate = rospy.Rate(hz=5)
    images, depths, cam_locs = load_images(image_dir)
    for i in range(len(images)):
        image = images[i]
        depths_np = depths[i].squeeze().cpu().numpy()
        extrinsics = cam_locs[i].cpu().numpy()
        
        client.publish_depth_image(image, depths_np, extrinsics)
        #client.request_voxel_computation()
        rate.sleep()

def send_classes(client: VoxSegClient, classes=None, groups=None):

    if type(classes) == list:
        client.publish_class_names(classes, groups, None, False)
    elif type(classes) == dict:
        client.publish_class_names(None, groups, classes, True)

def request_computation(client: VoxSegClient):
    client.request_voxel_computation()

def test_all(client: VoxSegClient):
    data_dir = '/home/pcgta/Documents/eth/wild_visual_navigation/wild_visual_navigation_orbit/feat-extract-out/test_16'
    send_images(client, data_dir)
    send_classes(client)
    request_computation(client)
    rospy.spin()
    
def main():
    client = VoxSegClient()

   
    # NOTE: THIS LINE IS NECESSARY 
    # If the rate isn't called before publishing the message, it just won't publish
    # Probably means that this just slows it down enough to work
    rospy.Rate(hz=5).sleep()
    #groups = {'terrain':['detritus', 'ground'], 'items':['machinery', 'equipment', 'fire hydrants']}
    groups = {'terrain':['equipment', 'ground'], 'items':['fire hydrant', 'rocks', 'bricks']}


    # prompts={'traversable': ['traversable','an image of easy terrain','an image of flat ground','flat ground', 'ground that could be walked on', 'easy to walk on', 'it is easy to walk on this'],
    #         'untraversable': ['untraversable','an image of challenging terrain', 'dangerous terrain', 'an image of bumpy terrain', 'difficult to walk on', 'it is difficult to walk on this'],
    #         'obstacle': ['obstacle','an obstacle', 'this is an obstacle', 'an image of an obstacle', 'this is an image of an obstacle', 'this is in the way']
    #         }
    
    prompts = {'machinery': ['bulldozer', 'backhoe', 'heavy machinery', 'machinery'],
               'equipment': ['barrel', 'crate', 'tarp'],
                'detritus': ['rocks', 'detritus', 'bricks'],
                'fire hydrants': ['fire hydrant'],
                'ground': ['ground']
    }
    #send_classes(client, prompts, groups=None)
#!/home/pcgta/mambaforge/envs/ovseg/bin/python

import rospy
from modules.client import VoxSegClient
from modules.utils import load_images


def send_images(client: VoxSegClient, image_dir=None):
    rate = rospy.Rate(hz=5)
    images, depths, cam_locs = load_images(image_dir)
    for i in range(len(images)):
        image = images[i]
        depths_np = depths[i].squeeze().cpu().numpy()
        extrinsics = cam_locs[i].cpu().numpy()
        
        client.publish_depth_image(image, depths_np, extrinsics)
        #client.request_voxel_computation()
        rate.sleep()

def send_classes(client: VoxSegClient, classes=None, groups=None):

    if type(classes) == list:
        client.publish_class_names(classes, groups, None, False)
    elif type(classes) == dict:
        client.publish_class_names(None, groups, classes, True)

def request_computation(client: VoxSegClient):
    client.request_voxel_computation()

def test_all(client: VoxSegClient):
    data_dir = '/home/pcgta/Documents/eth/wild_visual_navigation/wild_visual_navigation_orbit/feat-extract-out/test_16'
    send_images(client, data_dir)
    send_classes(client)
    request_computation(client)
    rospy.spin()
    
def main():
    client = VoxSegClient()

   
    # NOTE: THIS LINE IS NECESSARY 
    # If the rate isn't called before publishing the message, it just won't publish
    # Probably means that this just slows it down enough to work
    rospy.Rate(hz=5).sleep()
    #groups = {'terrain':['detritus', 'ground'], 'items':['machinery', 'equipment', 'fire hydrants']}
    groups = {'terrain':['equipment', 'ground'], 'items':['fire hydrant', 'rocks', 'bricks']}


    # prompts={'traversable': ['traversable','an image of easy terrain','an image of flat ground','flat ground', 'ground that could be walked on', 'easy to walk on', 'it is easy to walk on this'],
    #         'untraversable': ['untraversable','an image of challenging terrain', 'dangerous terrain', 'an image of bumpy terrain', 'difficult to walk on', 'it is difficult to walk on this'],
    #         'obstacle': ['obstacle','an obstacle', 'this is an obstacle', 'an image of an obstacle', 'this is an image of an obstacle', 'this is in the way']
    #         }
    
    prompts = {'machinery': ['bulldozer', 'backhoe', 'heavy machinery', 'machinery'],
               'equipment': ['barrel', 'crate', 'tarp'],
                'detritus': ['rocks', 'detritus', 'bricks'],
                'fire hydrants': ['fire hydrant'],
                'ground': ['ground']
    }
    #send_classes(client, prompts, groups=None)


    #UNCOMMENT BELOW AND COMMENT ABOVE TO USE CLASS NAMES INSTEAD
    class_names = ['heavy machinery', 'fire hydrant', 'rocks', 'bricks', 'ground']
    send_classes(client, class_names, groups=None)

if __name__ == '__main__':
    main()