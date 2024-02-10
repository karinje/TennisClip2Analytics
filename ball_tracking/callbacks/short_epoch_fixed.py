
from fastbook import *

from fastai.learner import CancelFitException

class ShortEpochBothCallback(Callback):
    def __init__(self,pct=0.01,short_valid=True): self.pct,self.short_valid = pct,short_valid
    def after_batch(self):
        if self.iter/self.n_iter < self.pct: return
        if self.training:    raise CancelTrainException
        if not self.training or self.short_valid: 
            print('cancelling validation inside short epoch')
            raise CancelValidException
            

    def after_train(self):
        self.learn.recorder.cancel_train = False
        self.learn.recorder.after_train()

    def after_validate(self):
        self.learn.recorder.cancel_valid = False
        self.learn.recorder.after_validate()

class ShortEpochCallbackFixed(ShortEpochCallback):
    def after_train(self):
        self.learn.recorder.cancel_train = False
        self.learn.recorder.after_train()

