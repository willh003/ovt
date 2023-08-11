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

from moviepy.editor import ImageSequenceClip        


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
    overlay = cv2.addWeighted(img_np.astype(np.uint8), 0.4, color_map, 0.6, 0)
    output = Image.fromarray(overlay)

    return output, Image.fromarray(color_map)

def save_masks(images, probs, base_name, img_nums=None):
    if img_nums == None:
        img_nums = range(len(images))
    for i, (image, prob_mask) in enumerate(zip(images, probs)):
        
        classifications = torch.argmax(prob_mask, dim=0)
        print(classifications.float().mean())
        classifications = classifications / classifications.max()
        masked_overlay, mask = get_turbo_image(image, classifications)

        num = img_nums[i]
        print(num)
        masked_overlay.save(os.path.join('test_output', f'{base_name}_{num}.jpg'))
        mask.save(os.path.join('test_output', f'{base_name}_mask_{num}.jpg'))
        

def load_images(directory):

    image_files = [f for f in os.listdir(directory) if (f.startswith('img_') or f.startswith('rgb_'))
     and (f.endswith('.jpg') or f.endswith('.png'))]

    images = []
    image_num = []

    


    for img_file in image_files:
        # Load image into PIL
        img_path = os.path.join(directory, img_file)
        image = read_image(img_path, format="BGR")
        image_copy = image.copy()



        num=img_file.split('_')[1].split('.')[0]
        image_num.append(num)
        
        images.append(torch.from_numpy(image_copy).permute(2,0,1))
    
    return torch.stack(images), image_num

def encoder_test(inp_folder, classes):
    images, img_nums = load_images(inp_folder)
    encoder =  WSImageEncoder(config='configs/ovt.yaml')
    print('model loaded')
    
    splits = list(range(0, len(images), 10))
    splits.append(len(images)) # ensure the leftovers are still included
    
    all_masks = []
    for i in range(len(splits) - 1): 
        start = splits[i]
        end = splits[i+1]

        cur_images = images[start:end]
        t1 = time.time()

        adapt = encoder.call_with_classes(cur_images, classes, use_adapter=True)
        all_masks = all_masks + [img for img in adapt]
        
        t2 = time.time()
        print(F"inference time: {t2-t1}")

        print(img_nums[start:end])
        

    all_masks_torch = torch.stack(all_masks)
    save_masks(images, all_masks_torch, 'adapt', img_nums)
        

    # diff = adapt - no_adapt
    
def time_test(inp_folder, classes):
    total_time_adapt = 0
    total_time_no_adapt = 0

    images = load_images(inp_folder)
    encoder =  WSImageEncoder(config='configs/ovt.yaml')
    
    for i, image in enumerate(images):
        t1 = time.time()
        adapt = encoder.call_with_classes(image[None], classes, use_adapter=True)
        t2 = time.time()
        no_adapt = encoder.call_with_classes(image[None], classes, use_adapter=False)
        t3 = time.time()
        
        if i > 0:
            total_time_adapt += (t2-t1)  
            total_time_no_adapt += (t3-t2)

    total_time_adapt /= (len(images) - 1)
    total_time_no_adapt /= (len(images) - 1)

    print(f'Total adapt time: {total_time_adapt}')
    print(f'Total no adapt time: {total_time_no_adapt}')


def quick_test(inp_file, classes):
    

    class_str = ''.join([f"'{elem}' " for elem in classes])

    cmd = f"python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names {class_str} --input {inp_file} --output ./pred --opts MODEL.WEIGHTS models/ovseg_swinbase_vitL14_ft_mpt.pth"
    print(cmd)
    os.system(cmd)


if __name__=='__main__':
    # predictor = Lightweight()
    # predictor.run(['test_data/test_18/img_640.jpg'], ['excavator', 'other'])
    #quick_test('test_data/real_site/img_40.png', ['untraversable', 'traversable ground', 'obstacle'])
    encoder_test('test_data/real_site_all', ['something an Anymal robot could walk on ', 'other '])
    #prefix = "You are a robot in a simulation environment. This photo is {} "
   # encoder_test('test_data/test_gazebo', {'untraversable': prefix.format("untraversable"), 'traversable': prefix.format("traversable")})
    #encoder_test('test_data/banana_apple', ['banana', 'apple', 'other'])
    #quick_test('test_data/site_test/img_600.jpg', )
#quick_test()
'''
python train_net.py --num-gpu 1 --eval-only --config-file configs/ovseg_R101c_vitB_bs32_120k.yaml MODEL.WEIGHTS /home/pcgta/Documents/playground/ov-seg/models/ovseg_R101c_vitB16_ft_mpt.pth.pt DATASETS.TEST \(\"ade20k_sem_seg_val\",\) 


python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'Oculus' 'Ukulele'  --input ./resources/demo_samples/sample_03.jpeg --output ./pred --opts MODEL.WEIGHTS 
'''