import argparse
from pathlib import Path 
from fastbook import *
import pandas as pd
import numpy as np
import logging
import re
import math
from ball_tracking.data.ball_gaussian import BallGaussianDataModule
from ball_tracking.data.data_module import BaseDataModule
from ball_tracking.models.tracknet import TrackNet

OPTIMIZER = "Adam"
LR = 1e-3
LOSS = "cross_entropy"
ONE_CYCLE_TOTAL_STEPS = 100
#default_device(False)
logging.basicConfig(level=logging.DEBUG)

class CreateLearner(object):
    """class to setup learner object from provided arguments"""
    def __init__(self, model: nn.Module, dls: DataLoaders, metrics: list, args: argparse.Namespace=None) -> None:
        super().__init__()
        self.model = model
        self.dls = dls
        self.metrics = metrics
        self.args = vars(args) if args is not None else {}
        self.data_config = self.model.data_config
        self.input_dims = self.data_config["num_inp_images"]
        self.output_classes = self.data_config["output_classes"]
        self.opt_func = Adam if self.args.get("optimizer", OPTIMIZER)=="Adam" else None
        loss = self.args.get("loss", LOSS)
        self.loss_fn = CrossEntropyLossFlat(axis=1) if loss==LOSS else MSELossFlat(axis=1)

    def get_learner(self):
        return Learner(self.dls, self.model, opt_func=self.opt_func, metrics=self.metrics, loss_func=self.loss_fn)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer name")
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS, help="")
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser

    def print_info(self):
        logging.info(f'Learner Details---------------------------------------------------------------------------------')
        logging.info(f'Model: {type(self.model).__name__} and data config: {self.data_config}')
        logging.info(f'Optimizer: {self.opt_func}') 
        logging.info(f'Metrics: {self.metrics}')
        logging.info(f'Loss Func: {self.loss_fn}')
        #logging.info(f'one_cycle_max_lr: {self.one_cycle_max_lr} and one_cycle_total_steps: {self.one_cycle_total_steps}')


if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser = BallGaussianDataModule.add_to_argparser(parser)
    parser = TrackNet.add_to_argparser(parser)
    parser = CreateLearner.add_to_argparse(parser)
    args = parser.parse_args()
    default_device(False)
    data_module = BallGaussianDataModule(args) 
    data_module.print_info()
    dls, data_config = data_module.get_dls(), data_module.config()
    model = TrackNet(data_config, args)
    model.print_info()
    setup_learner = CreateLearner(model, dls, [], args)
    learn = setup_learner.get_learner() 
    setup_learner.print_info()
    b = learn.dls.valid.one_batch()
    logging.info(f'batch datatypes: {b[0].shape}, {b[3].shape}')
    gt = b[3]
    pred = learn.model(*b[:3])
    pred =b[3]*0.1
    logging.info(f'{gt.dtype}, {pred.dtype}')
    logging.info(f'sample loss: {setup_learner.loss_fn(pred, gt)}')
    #logging.info(learner.summary())
    # print(f'{default_device(), defaults.use_cuda}')
    # learner.lr_find()
    # python ~/git/ball_tracking_3d/ball_tracking/data/data_module.py --train_data_path /Users/sanjaykarinje/Downloads/Dataset 
    #                                                                 --infer_data_path /Users/sanjaykarinje/Downloads/match_frames 
    #                                                                 --num_inp_images 3 --target_img_position 1
    #                                                                 --output_classes 1 --loss "mse" 

