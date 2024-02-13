from fastai.metrics import Metric
from ball_tracking.data.ball_gaussian import BallGaussianDataModule
from ball_tracking.data.data_module import BaseDataModule
from ball_tracking.models.tracknet import TrackNet
from ball_tracking.learner.learner import CreateLearner
from ball_tracking.metrics.utils import mask2coord
import argparse
import logging
import torch
from fastbook import default_device
logging.basicConfig(level=logging.DEBUG)

class PredVarX(Metric):

    def __init__(self):
      self.all_preds = torch.tensor([]).to(default_device())

    def accumulate(self, learn):
      preds,y = mask2coord(learn.pred), mask2coord(learn.y)
      self.all_preds = torch.cat((self.all_preds, preds), axis=0)

    def reset(self):
        self.all_preds = torch.tensor([]).to(default_device())

    @property
    def value(self):
        return self.all_preds.std(axis=0)[0]

class PredVarY(PredVarX):
    @property
    def value(self):
        return self.all_preds.std(axis=0)[1]

if __name__=="__main__":
    print(f'{default_device()}')
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
    pred_stats = PredStats()
    gt = torch.tensor(list(map(base_d.get_y(),(learner.dls.valid.items[0],learner.dls.valid.items[1]))))
    pred = mask2coord(b[3]).cpu()*2
    logging.info(f'gt: {gt}, pred: {pred}')
    

