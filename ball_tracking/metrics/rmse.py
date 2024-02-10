IMG_RESIZE = (360,640)
from fastai.metrics import Metric
from ball_tracking.data.ball_gaussian import BallGaussianDataModule
from ball_tracking.data.data_module import BaseDataModule
from ball_tracking.models.tracknet import TrackNet
from ball_tracking.learner.learner import CreateLearner
import argparse
import logging
import torch
logging.basicConfig(level=logging.DEBUG)

class RMSEArgmax(Metric):
    def __init__(self):
      self.r2_dist_acc = 0
      self.r2_count = 0

    def r2_dist(self, a, b):
      return torch.sqrt(torch.pow(a-b,2).mean(axis=-1))

    def mask2coord(self, batch, img_size=IMG_RESIZE):
      bs = len(batch)
      w = img_size[1]
      return torch.stack([torch.stack([x+1,y+1]) for x,y in zip(batch.view(bs,-1).argmax(dim=-1)%w, batch.view(bs,-1).argmax(dim=-1)//w)]).float()

    def accumulate(self, learn):
      preds,y = self.mask2coord(learn.pred), self.mask2coord(learn.y)
      self.r2_dist_acc += self.r2_dist(preds, y).sum()
      self.r2_count += len(preds)

    def reset(self):
        self.r2_dist_acc = 0
        self.r2_count = 0

    @property
    def value(self):
        return self.r2_dist_acc / (self.r2_count+1)
##

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
    learner = CreateLearner(model, dls, args).get_learner() 
    b = learner.dls.valid.one_batch()
    rmse = RMSEArgmax()
    gt = torch.tensor(list(map(base_d.get_y(),(learner.dls.valid.items[0],learner.dls.valid.items[1]))))
    pred = rmse.mask2coord(b[3]).cpu()*2
    logging.info(f'gt: {gt}, pred: {pred}')
    logging.info(f'rmse: {rmse.r2_dist(gt,pred)}')
    # python ~/git/ball_tracking_3d/ball_tracking/data/data_module.py --train_data_path /Users/sanjaykarinje/Downloads/Dataset 
    #                                                                 --infer_data_path /Users/sanjaykarinje/Downloads/match_frames 


