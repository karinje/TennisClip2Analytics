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
        self.lr = self.args.get("lr", LR)
        loss = self.args.get("loss", LOSS)
        self.loss_fn = CrossEntropyLossFlat(axis=1) if loss==LOSS else MSELossFlat(axis=1)
        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps", ONE_CYCLE_TOTAL_STEPS)

    def get_learner(self):
        return Learner(self.dls, self.model, opt_func=self.opt_func, metrics=self.metrics, loss_func=self.loss_fn)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=OPTIMIZER, help="optimizer name")
        parser.add_argument("--lr", type=float, default=LR, help="")
        parser.add_argument("--one_cycle_max_lr", type=float, default=None, help="")
        parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS, help="")
        parser.add_argument("--loss", type=str, default=LOSS, help="loss function from torch.nn.functional")
        return parser


if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser = BallGaussianDataModule.add_to_argparser(parser)
    parser = CreateLearner.add_to_argparse(parser)
    args = parser.parse_args()
    logging.info(f'input loss: {args.loss}')
    data_module = BallGaussianDataModule(args) if args.loss==LOSS else BaseDataModule(args)  
    data_module.print_info()
    dls, data_config = data_module.get_dls(), data_module.config()
    logging.info(f'data config: {data_config}')
    model = TrackNet(data_config)
    learner = CreateLearner(model, dls, args).get_learner() 
    logging.info(learner.summary())
    #print(f'{default_device(), defaults.use_cuda}')
    learner.lr_find()
    # python ~/git/ball_tracking_3d/ball_tracking/data/data_module.py --train_data_path /Users/sanjaykarinje/Downloads/Dataset 
    #                                                                 --infer_data_path /Users/sanjaykarinje/Downloads/match_frames 
    #                                                                 --num_inp_images 3 --target_img_position 1 

