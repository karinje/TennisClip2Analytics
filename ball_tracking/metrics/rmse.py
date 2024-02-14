from fastai.metrics import Metric
from ball_tracking.data.ball_gaussian import BallGaussianDataModule
from ball_tracking.data.data_module import BaseDataModule
from ball_tracking.models.tracknet import TrackNet
from ball_tracking.learner.learner import CreateLearner
from ball_tracking.metrics.utils import mask2coord
from ball_tracking.callbacks import ShortEpochBothCallback
import argparse
import logging
import torch
import numpy as np
from fastbook import default_device 
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path

class BallPresentRMSE(Metric):
    def __init__(self):
      self.avg_dist_a, self.avg_dist_p = [], [] 
      self.below5_a, self.below5_p = 0, 0
      self.y_absent = torch.tensor([0,0]).to('cpu')

    def r2_dist(self, a, b):
      return torch.sqrt(torch.pow(a-b,2).mean(dim=-1))

    def accumulate(self, learn):
      preds,y = mask2coord(learn.pred), mask2coord(learn.y)
      #logging.info(f'preds: {preds} and y: {y}')
      dist = self.r2_dist(preds, y)
      for y_i, dist_i in zip(y, dist):
        #logging.info(f'yi: {y_i} and dist missing: {self.r2_dist(y_i,self.y_absent).item()} and dist_i: {dist_i}')
        if self.r2_dist(y_i,self.y_absent).item()<5:
            #logging.info(f'entering absent')
            self.avg_dist_a.append(dist_i.item())
            if dist_i.item()<=10: self.below5_a += 1
        else:
            self.avg_dist_p.append(dist_i.item())
            if dist_i.item()<=10: self.below5_p += 1
          

    def reset(self):
        self.avg_dist_a, self.avg_dist_p = [], []
        self.below5_a, self.below5_p = 0, 0

    @property
    def value(self): return np.mean(self.avg_dist_p)


class BallAbsentRMSE(BallPresentRMSE):
    @property
    def value(self): return np.mean(self.avg_dist_a)


class BallPresent5px(BallPresentRMSE):
    @property
    def value(self): return self.below5_p/len(self.avg_dist_p)

class BallAbsent5px(BallPresentRMSE):
    @property
    def value(self): return self.below5_a/len(self.avg_dist_a)

if __name__=="__main__":
    print(f'default device: {default_device()}')
    parser = argparse.ArgumentParser(add_help=True)
    parser = BallGaussianDataModule.add_to_argparser(parser)
    parser = TrackNet.add_to_argparser(parser)
    parser = CreateLearner.add_to_argparse(parser)
    args = parser.parse_args()
    logging.info(f'input loss: {args.loss}')
    base_d = BaseDataModule(args) 
    ballg_d = BallGaussianDataModule(args)
    dls, data_config = ballg_d.get_dls(), ballg_d.config()
    logging.info(f'data config: {data_config}')
    model = TrackNet(data_config)
    rmse_p, rmse_a, bp_pct = BallPresentRMSE(), BallAbsentRMSE(), BallPresent5px()
    learner = CreateLearner(model, dls, [rmse_p, rmse_a, bp_pct], args).get_learner()
    learner = learner.load(Path('/home/ubuntu/test-storage/ball_tracking_3d/models/noact_tracknet_ce_1e-2'))
    learner.add_cb(ShortEpochBothCallback(pct=0.1, short_valid=False))
    learner.fit(1,1e-2)
