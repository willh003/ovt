import os
import argparse
import glob
import multiprocessing as mp
import time
import cv2
import tqdm

from detectron2.config import get_cfg

from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from open_vocab_seg import add_ovseg_config

from open_vocab_seg.utils import VisualizationDemo
from detectron2.engine.defaults import DefaultPredictor

import torch
import torch.nn.functional as F

from open_vocab_seg.ws_ovseg_model import WSImageEncoder

from PIL import Image
import numpy as np

def get_turbo_image(img, mask):
    """
    img: torch.tensor, (3, h, w) vals between 0,1
    mask: torch.tensor, (h,w) vals between 0,256
    returns: PIL Image
    """
    img_np = img.permute(1,2,0).cpu().numpy()
    mask_np = mask.cpu().numpy()*255.0

    color_map = cv2.applyColorMap(mask_np.astype(np.uint8), cv2.COLORMAP_TURBO)


    # Overlay the color map on the original image
    overlay = cv2.addWeighted(img_np.astype(np.uint8), 0.7, color_map, 0.3, 0)
    output = Image.fromarray(overlay)

    return output, Image.fromarray(color_map)

def save_masks(images, probs, base_name):
    for i, (image, prob_mask) in enumerate(zip(images, probs)):
        
        classifications = torch.argmax(prob_mask, dim=0)
        classifications = classifications / classifications.max()
        masked_overlay, mask = get_turbo_image(image, classifications)
        masked_overlay.save(os.path.join('test_output', f'{base_name}_{i}.jpg'))
        mask.save(os.path.join('test_output', f'{base_name}_mask_{i}.jpg'))
        

def load_images(directory):

    image_files = sorted([f for f in os.listdir(directory) if f.startswith('img_') and f.endswith('.jpg')])

    images = []


    for img_file in image_files:
        # Load image into PIL
        img_path = os.path.join(directory, img_file)
        image = read_image(img_path, format="BGR")
        image_copy = image.copy()
        
        images.append(torch.from_numpy(image_copy).permute(2,0,1))
    
    return torch.stack(images)

def encoder_test(inp_folder, classes):
    images = load_images(inp_folder)
    encoder =  WSImageEncoder(config='configs/ovt.yaml')

    # t1 = time.time()
    adapt = encoder.call_with_classes(images, classes, use_adapter=True)
    # print(f'time: {time.time() - t1}')

    no_adapt = encoder.call_with_classes(images, classes, use_adapter=False)
    
    
    save_masks(images, no_adapt, "no_adapt")
    save_masks(images, adapt, "adapt")
    
    

    # diff = adapt - no_adapt
    

def quick_test(inp_file, classes):
    

    class_str = ''.join([f"'{elem}' " for elem in classes])

    cmd = f"python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names {class_str} --input {inp_file} --output ./pred --opts MODEL.WEIGHTS models/ovseg_swinbase_vitL14_ft_mpt.pth"
    print(cmd)
    os.system(cmd)


if __name__=='__main__':
    # predictor = Lightweight()
    # predictor.run(['test_data/test_18/img_640.jpg'], ['excavator', 'other'])
   # quick_test('batch_test/cubesphereconetorus/test_0/img_260.jpg', ['shiny'])
    #encoder_test('test_data/site_test_2', ['equipment', 'ground'])
    encoder_test('test_data/banana_apple', ['banana', 'apple', 'other'])
    #quick_test('test_data/site_test/img_600.jpg', )
#quick_test()
'''
python train_net.py --num-gpu 1 --eval-only --config-file configs/ovseg_R101c_vitB_bs32_120k.yaml MODEL.WEIGHTS /home/pcgta/Documents/playground/ov-seg/models/ovseg_R101c_vitB16_ft_mpt.pth.pt DATASETS.TEST \(\"ade20k_sem_seg_val\",\) 


python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'Oculus' 'Ukulele'  --input ./resources/demo_samples/sample_03.jpeg --output ./pred --opts MODEL.WEIGHTS 
'''