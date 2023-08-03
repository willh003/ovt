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


WINDOW_NAME = "OVSeg"
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
    # adapt = encoder.call_with_classes(images, classes, use_adapter=True)
    # print(f'time: {time.time() - t1}')

    no_adapt = encoder.call_with_classes(images, classes, use_adapter=False)

    # diff = adapt - no_adapt
    breakpoint()

def quick_test(inp_file, classes):
    

    class_str = ''.join([f"'{elem}' " for elem in classes])

    cmd = f"python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names {class_str} --input {inp_file} --output ./pred --opts MODEL.WEIGHTS models/ovseg_swinbase_vitL14_ft_mpt.pth"
    print(cmd)
    os.system(cmd)


if __name__=='__main__':
    # predictor = Lightweight()
    # predictor.run(['test_data/test_18/img_640.jpg'], ['excavator', 'other'])
   # quick_test('batch_test/cubesphereconetorus/test_0/img_260.jpg', ['shiny'])
    encoder_test('test_data/site_test_2', ['equipment', 'ground'])
    #quick_test('test_data/site_test/img_600.jpg', )
#quick_test()
'''
python train_net.py --num-gpu 1 --eval-only --config-file configs/ovseg_R101c_vitB_bs32_120k.yaml MODEL.WEIGHTS /home/pcgta/Documents/playground/ov-seg/models/ovseg_R101c_vitB16_ft_mpt.pth.pt DATASETS.TEST \(\"ade20k_sem_seg_val\",\) 


python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'Oculus' 'Ukulele'  --input ./resources/demo_samples/sample_03.jpeg --output ./pred --opts MODEL.WEIGHTS 
'''