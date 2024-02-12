import argparse
from fastbook import * 
import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
from typing import Dict, Union

logging.basicConfig(level=logging.DEBUG)

def conv(ni, nf, ks=3, stride=1, act=nn.ReLU, norm=None, pool=None, bias=None):
    if bias is None: bias = not isinstance(norm, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d))
    layers = [nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2, bias=bias)]
    if norm: layers.append(norm(nf))
    if act: layers.append(act())
    if pool: layers.append(pool(2))
    return nn.Sequential(*layers)

def _conv_block(ni, nf, blocks=2, ks=3, stride=1, act=nn.ReLU, norm=nn.BatchNorm2d, pool=nn.MaxPool2d, bias=None):
    layers = [conv(ni=ni, nf=nf, stride=stride, act=act, norm=norm, pool=None, bias=bias)]
    layers += [conv(ni=nf, nf=nf, stride=stride, act=act, norm=norm, pool=None, bias=bias) for i in range(blocks-2)]
    layers += [conv(ni=nf, nf=nf, stride=stride, act=act, norm=norm, pool=pool, bias=bias)]
    return nn.Sequential(*layers)

def _deconv_block(ni, nf, blocks=2, ks=3, stride=1, act=nn.ReLU, norm=nn.BatchNorm2d, bias=None):
    layers = [nn.UpsamplingNearest2d(scale_factor=2)]
    layers += [conv(ni=ni, nf=nf, stride=stride, act=act, norm=norm, bias=bias)]
    layers += [conv(ni=nf, nf=nf, stride=stride, act=act, norm=norm, bias=bias) for i in range(blocks-1)]
    return nn.Sequential(*layers)

def get_ds(ns=(9,64,128,256,512), blocks=(2,2,3,3), act=nn.ReLU, norm=nn.BatchNorm2d, pool=nn.MaxPool2d):
    layers = [_conv_block(ni, nf, blocks=b, act=act, norm=norm, pool=pool if idx!=3 else None) for idx, (ni,nf,b) in enumerate(zip(ns[:-1], ns[1:], blocks))]
    return nn.Sequential(*layers)

def get_us(ns=(512,256,128,64), blocks=(3,2,2), n_classes=256, act=nn.ReLU, final_act=None, norm=nn.BatchNorm2d):
    layers = [_deconv_block(ni, nf, blocks=b, act=act, norm=norm) for ni,nf,b in zip(ns[:-1], ns[1:], blocks)]
    layers += [conv(ni=ns[-1], nf=n_classes, act=final_act, norm=None)]
    return nn.Sequential(*layers)


class TrackNet(nn.Module):
    """Create TrackNet Downsample/Upsample model based on input config"""
    def __init__(self, data_config: Dict, args: argparse.Namespace=None ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.num_inp_images = self.data_config.get("num_inp_images", 3)
        self.output_classes = self.data_config.get("output_classes", 1)
        self.final_act = self.args.get("final_act", None)
        self.ds = get_ds(ns=(self.num_inp_images*3,64,128,256,512))
        self.final_act = nn.Sigmoid if self.final_act=="sigmoid" else (nn.ReLU if self.final_act=="relu" else None)
        self.output_scale = 255 if self.output_classes==1 else 1  
        
        self.us = get_us(n_classes=self.output_classes, final_act=self.final_act)

    def forward(self, *args):
        x_cat = torch.cat((args), dim=1)
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
    parser = TrackNet.add_to_argparser(parser)
    args = parser.parse_args()
    data_config = {"num_inp_images": 3, "output_classes": 256}
    model = TrackNet(data_config, args)
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
