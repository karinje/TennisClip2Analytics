from fastai.metrics import Metric
from ball_tracking.data.ball_gaussian import BallGaussianDataModule
from ball_tracking.data.data_module import BaseDataModule
from ball_tracking.models.tracknet import TrackNet
from ball_tracking.learner.learner import CreateLearner
from ball_tracking.metrics.utils import mask2coord
import argparse
import logging
import torch
import numpy as np
from fastbook import default_device 
logging.basicConfig(level=logging.DEBUG)

class BallPresentRMSE(Metric):
    def __init__(self):
      self.avg_dist_a, self.avg_dist_p = [], [] 
      self.y_absent = torch.tensor([0,0]).to(default_device())

    def r2_dist(self, a, b):
      return torch.sqrt(torch.pow(a-b,2).mean(axis=-1))

    def accumulate(self, learn):
      preds,y = mask2coord(learn.pred), mask2coord(learn.y)
      dist = self.r2_dist(preds, y)
      for y_i, dist_i in zip(y, dist):
          if torch.equal(y_i, self.y_absent):
              self.avg_dist_a.append(dist_i.item())
          else:
              self.avg_dist_p.append(dist_i.item())

    def reset(self):
        self.avg_dist_a, self.avg_dist_p = [], [] 

    @property
    def value(self): return np.mean(self.avg_dist_p)


class BallAbsentRMSE(BallPresentRMSE):
    
    @property
    def value(self): return np.mean(self.avg_dist_a)


class BallPresentPct(BallPresentRMSE):
    
    @property
    def value(self): return 1/(1+len(self.avg_dist_a)/len(self.avg_dist_p))


if __name__=="__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser = BallGaussianDataModule.add_to_argparser(parser)
    parser = CreateLearner.add_to_argparse(parser)
    args = parser.parse_args()
    logging.info(f'input loss: {args.loss}')
    base_d = BaseDataModule(args) 
    ballg_d = BallGaussianDataModule(args)
    dls, data_config = ballg_d.get_dls(), ballg_d.config()
    logging.info(f'data config: {data_config}')
    model = TrackNet(data_config)
    learner = CreateLearner(model, dls, [], args).get_learner() 
    b = learner.dls.valid.one_batch()
    test_map = ballg_d.get_y()(learner.dls.valid.items[0]).argmax()
    logging.info(f'target shape: {b[3].shape} and test_map: {test_map//1280},{test_map%1280}')
    rmse_p, rmse_a, bp_pct = BallPresentRMSE(), BallAbsentRMSE(), BallPresentPct()
    gt = torch.tensor(list(map(base_d.get_y(),learner.dls.valid.items[0:args.samples_per_batch])))
    pred = mask2coord(b[3]).cpu()*2
    logging.info(f'{learner.dls.valid.items[0]}')
    logging.info(f'gt: {gt}, pred: {pred}')
    logging.info(f'rmse: {rmse_p.r2_dist(gt,pred)}')
    # python ~/git/ball_tracking_3d/ball_tracking/data/data_module.py --train_data_path /Users/sanjaykarinje/Downloads/Dataset 
    #                                                                 --infer_data_path /Users/sanjaykarinje/Downloads/match_frames 


