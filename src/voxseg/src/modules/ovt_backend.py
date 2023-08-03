import rospy

from voxseg.msg import Classes, CustomChannelImage
from voxseg.srv import ImageSeg, ImageSegResponse

import numpy as np
import json
import torch

from modules.config import *
from modules.utils import *
from modules.ovseg.open_vocab_seg.ws_ovseg_model import WSImageEncoder




class OVTServer:
    def __init__(self):
        """
        Subscribes to the class name topic

        Attaches to seg request service
        """
        use_prompts = False
        self.classes = ['Traversable', 'Untraversable', 'Obstacle']
        self.prompts = {}
        self.groups = {}
        self.use_prompts = use_prompts

        self.encoder = WSImageEncoder(VOXSEG_ROOT_DIR, config='configs/ovt.yaml')
        
        rospy.init_node(SERVER_NODE, anonymous=True)
        rospy.Subscriber(CLASS_TOPIC, Classes, self._class_name_callback)
        rospy.Service(OVT_REQUEST_SERVICE, ImageSeg, self._handle_compute_request)
    
        print('Backend Has Been Initialized')

        rospy.spin()

   
    def _handle_compute_request(self, req):
        # Update frpom the most recent tensors 

        images = torch_from_img_array_msg(req.images).float().to(DEVICE)
        class_probs = self.encoder.call_with_classes(images, self.classes, use_adapter=False)

        breakpoint()
        class_probs_msg = []
        for img in class_probs:
            c, h, w = img.size()

            flattened_data = img.float().flatten().numpy()
            img_msg = CustomChannelImage(data=flattened_data, height=h, width=w, num_channels=c)
            class_probs_msg.append(img_msg)

        response = ImageSegResponse(pixel_probs = class_probs_msg)

        return response

    def _class_name_callback(self, msg):

        prompts = convert_dictionary_array_to_dict(msg.prompts)
        groups = convert_dictionary_array_to_dict(msg.groups)

        self.classes = list(msg.classes)
        self.prompts = prompts
        self.groups = groups
        self.use_prompts = bool(msg.use_prompts)

        print(f'Classes recieved: {self.classes}')

