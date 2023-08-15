import rospy

from voxseg.msg import ImageArray
from voxseg.srv import ImageSeg, ImageSegResponse

from torchvision.transforms import ToPILImage

from modules.voxseg_root_dir import VOXSEG_ROOT_DIR
from modules.utils import *
from modules.ovseg.open_vocab_seg.ws_ovseg_model import WSImageEncoder

class OVTServer:
    def __init__(self):
        """
        Only exists to perform computations. Does not handle any data manipulation or storage
        """
        self.server_node = rospy.get_param('/ovt/SERVER_NODE')
        self.ovt_request_service = rospy.get_param('/ovt/REQUEST_SERVICE')
        self.device = rospy.get_param('ovt/DEVICE')

        self.encoder = WSImageEncoder(VOXSEG_ROOT_DIR, config='configs/ovt.yaml')
        
        rospy.init_node(self.server_node, anonymous=True)
        rospy.Service(self.ovt_request_service, ImageSeg, self._handle_compute_request)
    
        print('Backend Has Been Initialized')

        rospy.spin()

   
    def _handle_compute_request(self, req):
        # Update frpom the most recent tensors 
        images_msg = list(req.images)
        
        classes = [str(c) for c in req.classes]
        
        images = torch_from_img_array_msg(images_msg).float().to(self.device)
        class_probs = self.encoder.call_with_classes(images, classes, use_adapter=False)

        all_probs_msg = []
        bridge = CvBridge()
        for i, multi_channel_probs in enumerate(class_probs):
            c, _, _ = multi_channel_probs.size()

            corresponding_image_msg = images_msg[i]
            separate_channel_probs = []
            for j in range(c):
                channel_probs = multi_channel_probs[j]

                
                
                probs_img_msg = bridge.cv2_to_imgmsg(channel_probs.numpy(), header=corresponding_image_msg.header)
                
                separate_channel_probs.append(probs_img_msg)

            probs_msg = ImageArray(images = separate_channel_probs)
            all_probs_msg.append(probs_msg)

        response = ImageSegResponse(prob_images = all_probs_msg)
        return response



