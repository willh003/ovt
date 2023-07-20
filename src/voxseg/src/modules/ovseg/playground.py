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

from ws.utils import *

WINDOW_NAME = "OVSeg"

def quick_test(inp_file, classes):
    

    class_str = ''.join([f"'{elem}' " for elem in classes])

    cmd = f"python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names {class_str} --input {inp_file} --output ./pred --opts MODEL.WEIGHTS models/ovseg_swinbase_vitL14_ft_mpt.pth"
    print(cmd)
    os.system(cmd)

class Lightweight:
    """
    Implements ovseg inference in a usable manner
    """

    def __init__(self, model_type = 'ws'):
        if model_type == 'ws':
            config_file = 'configs/ovseg_ws_demo.yaml'
            opts = ['MODEL.WEIGHTS', 'models/ovseg_swinbase_vitL14_ft_mpt.pth']
        elif model_type == 'large':
            config_file = 'configs/ovseg_swinB_vitL_demo.yaml'
            opts = ['MODEL.WEIGHTS', 'models/ovseg_swinbase_vitL14_ft_mpt.pth']
        elif model_type == 'small':
            config_file = 'configs/ovseg_R101c_vitB_bs32_120k.yaml'
            opts = ['MODEL.WEIGHTS', 'models/ovseg_R101c_vitB16_ft_mpt.pth.pt']
        
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        add_ovseg_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        cfg.freeze()

        self.demo = VisualizationDemo(cfg)


    def run(self, input, class_names, output = './pred.png'):
        """
        Currently only supports single images at a time. we may want to make this batched
        """
        setup_logger(name="fvcore")
        logger = setup_logger()

        if len(input) == 1:
            input = glob.glob(os.path.expanduser(input[0]))
            assert input, "The input path(s) was not found"
        for path in tqdm.tqdm(input, disable=not output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = self.demo.run_on_image(img, class_names)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            if output:
                if os.path.isdir(output):
                    assert os.path.isdir(output), output
                    out_filename = os.path.join(output, os.path.basename(path))
                else:
                    assert len(input) == 1, "Please specify a directory with output"
                    out_filename = output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit

if __name__=='__main__':
    # predictor = Lightweight()
    # predictor.run(['test_data/test_18/img_640.jpg'], ['excavator', 'other'])
    quick_test('batch_test/cubesphereconetorus/test_0/img_260.jpg', ['shiny'])
    #quick_test('test_data/site_test/img_600.jpg', ['equipment', 'ground'])
#quick_test()
'''
python train_net.py --num-gpu 1 --eval-only --config-file configs/ovseg_R101c_vitB_bs32_120k.yaml MODEL.WEIGHTS /home/pcgta/Documents/playground/ov-seg/models/ovseg_R101c_vitB16_ft_mpt.pth.pt DATASETS.TEST \(\"ade20k_sem_seg_val\",\) 


python demo.py --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'Oculus' 'Ukulele'  --input ./resources/demo_samples/sample_03.jpeg --output ./pred --opts MODEL.WEIGHTS 
'''