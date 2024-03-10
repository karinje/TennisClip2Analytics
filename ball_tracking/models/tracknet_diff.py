import argparse
from fastbook import * 
import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
from typing import Dict, Union
from ball_tracking.models.tracknet import TrackNet
logging.basicConfig(level=logging.DEBUG)

class TrackNetDiff(TrackNet):
    """Create TrackNet Diff which uses difference between inps to feed the model"""
    def forward(self, *args):
        if len(args)==1:
            x_cat = args[0]
        elif len(args)==3:
            all_is = (args[1]-args[0], args[1], args[2]-args[1])
            x_cat = torch.cat(all_is, dim=1)
        elif len(args)==5:
            all_is = (args[2]-args[0], args[2]-args[1], args[2], args[3]-args[2], args[4]-args[2]) 
            x_cat = torch.cat(all_is, dim=1)
        return self.us(self.ds(x_cat))*self.output_scale

    @staticmethod
    def add_to_argparser(parser):
        parser.add_argument("--final_act", type=str, default=None, help="final activation")
        return parser
    
    def print_info(self):
        logging.info("Model Details-------------------------------------------------------------------------------------------")
        logging.info(f'num_inp_imgs: {self.num_inp_images}, output_classes: {self.output_classes},\
        final_act: {self.final_act}, output_scale: {self.output_scale}')



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = TrackNetDiff.add_to_argparser(parser)
    args = parser.parse_args()
    data_config = {"num_inp_images": 3, "output_classes": 256}
    model = TrackNetDiff(data_config, args)
    test_inp = [torch.randn((2,3,36,64)) for _ in range(3)]
    test_out = model(*test_inp)
    if model.output_classes==1:
        test_target = torch.randn((2,32,64))
        loss = MSELossFlat(axis=1)(test_out, test_target)
        loss.backward()
        loss_calc = torch.pow(torch.squeeze(test_out)-test_target,2).mean()
    else:
        test_target = torch.randint(low=0, high=model.output_classes-1, size=(2, 32, 64), dtype=torch.int)
        loss = CrossEntropyLossFlat(axis=1)(test_out,test_target)
        loss.backward()
        loss_calc = None
    logging.debug(f'input len: {len(test_inp)}, shape: {test_inp[0].shape}, target shape: {test_target.shape}, out shape: {test_out.shape}, loss: {loss} and loss calc: {loss_calc}')
    # python models/tracknet.py  --num_inp_images 3 --output_channels 256 
