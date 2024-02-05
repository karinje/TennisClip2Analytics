
from fastbook import *


class ShortEpochCallbackFixed(ShortEpochCallback):
    def after_train(self):
        self.learn.recorder.cancel_train = False
        self.learn.recorder.after_train()
