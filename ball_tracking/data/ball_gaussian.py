import argparse
from pathlib import Path 
from fastbook import *
import pandas as pd
import numpy as np
import logging
import re
import math
import torch
from torch import tensor
from ball_tracking.data.data_module import BaseDataModule

VARIANCE = 10
KERNEL_SIZE = 20 
HEIGHT, WIDTH = 720, 1280

logging.basicConfig(level=logging.DEBUG)

def gaussian_kernel(size: int, variance: int):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2+y**2)/float(2*variance))
    return g


class BallGaussianDataModule(BaseDataModule):
    """Class for generating inputs and targets
       Inputs are file locations for input images
       Target is array with gaussian around ball location
    """
    def __init__(self, args: argparse.Namespace=None) -> None:
        super().__init__(args)
        self.kernel_size = self.args.get("kernel_size",KERNEL_SIZE)
        self.variance = self.args.get("variance", VARIANCE)
        self.kernel = self.args.get("kernel", "gaussian")

    @staticmethod
    def add_to_argparser(parser):
        parser = BaseDataModule.add_to_argparser(parser)
        parser.add_argument("--kernel_size", type=int, default=KERNEL_SIZE, help="size of kernel around ball coords")
        parser.add_argument("--variance", type=int, default=VARIANCE, help="varince of kernel around ball coords")
        parser.add_argument("--kernel", type=str, default='gaussian', help="distribution func to use around ball coords")
        return parser

    def get_y(self, label_file='Label.csv'):
        s_get_y = super().get_y(label_file=label_file)
        def _get_y(f):
            x,y = list(map(int, s_get_y(f)))
            size, variance = self.kernel_size, self.variance
            x,y = min(x, WIDTH-1), min(y, HEIGHT-1)
            logging.info(f'x: {x}, y: {y}, size: {size}, variance: {variance}')
            if self.kernel=='gaussian': kernel_func = gaussian_kernel
            out_arr = np.zeros((HEIGHT+2*size, WIDTH+2*size))
            logging.debug(f'Ball Coords: {(x,y)},Types: {type(x), type(y)}, Output Arr Shape: {out_arr.shape}, X Slice: {slice(x, x+2*size+1)}, Y Slice: {slice(y, y+2*size+1)}')
            logging.debug(f'out arr sliced shape: {out_arr[slice(y, y+2*size+1), slice(x, x+2*size+1)].shape}, gaussian kernal shape: {kernel_func(size, variance).shape}')
            out_arr[slice(y, y+2*size+1), slice(x, x+2*size+1)] = kernel_func(size, variance)*(self.output_classes-1)
            return tensor(out_arr[size:-size, size:-size]).short()
        return _get_y

    def get_blocks(self):
        blocks = super().get_blocks()
        codes = np.arange(self.output_classes)
        blocks[-1] = MaskBlock(codes)
        return blocks

    def print_info(self):
        super().print_info()
        logging.info(f'Gaussian Size: {self.kernel_size}, Variance: {self.variance}')
        sample_f = self.get_valid_files()(self.train_data_path)[0]
        logging.info(f'Shape of Gaussian Output: {self.get_y()(sample_f).shape}')

if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser = BallGaussianDataModule.add_to_argparser(parser)
    args = parser.parse_args()
    ball_g = BallGaussianDataModule(args)
    ball_g.print_info()
    db = ball_g.get_db()
    db.summary(ball_g.train_data_path)
    # python data/ball_gaussian.py --train_data_path /Users/sanjaykarinje/Downloads/Dataset --infer_data_path
    # /Users/sanjaykarinje/Downloads/match_frames --num_inp_images 3 --target_img_position 4 --kernel_size 20
    # --variance 10

