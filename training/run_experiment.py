import argparse
from pathlib import Path
from fastbook import *
import numpy as np
import pandas as pd
import torch
from ball_tracking.learner import CreateLearner
from ball_tracking.callbacks import ShortEpochCallbackFixed, ShortEpochBothCallback 
from ball_tracking.metrics import RMSEArgmax, PredStats 
from training.util import DATA_CLASS_MODULE, MODEL_CLASS_MODULE, import_class, setup_data_and_model_from_args
from torch.profiler import profile, record_function, ProfilerActivity
import logging
from fastai.callback.tensorboard import * 
# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    learner_parser = CreateLearner.add_to_argparse(parser)
    learner_parser._action_groups[1].title = "Learner Args"
    parser = argparse.ArgumentParser(add_help=False, parents=[learner_parser])
    # Basic arguments
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Specify number of epochs"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Specify number of gpus"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Specify number of workers"
    )
    parser.add_argument(
        "--short_epoch",
        type=float,
        default=1,
        help='Specify fraction of epoch as decimal')

    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
    )
    parser.add_argument(
        "--data_class",
        type=str,
        default="BallGaussianDataModule",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="TrackNet",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--load_learner", type=str, default=None, help="If passed, loads a learner from the provided path."
    )
    parser.add_argument(
        "--save_model", type=str, default=None, help="save learner"
    )
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
        + " Default is 0.",
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparser(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparser(model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=TrackNet --data_class=BallGaussianDataModule --
    ```

    For basic help documentation, run the command
    ```
    python training/run_experiment.py --help
    ```

    The available command line args differ depending on some of the arguments, including --model_class and --data_class.

    To see which command line args are available and read their documentation, provide values for those arguments
    before invoking --help, like so:
    ```
    python training/run_experiment.py --model_class=TrackNet --data_class=BallGaussianDataModule --help
    """
    parser = _setup_parser()
    args = parser.parse_args()
    #default_device(False)
    data, model = setup_data_and_model_from_args(args)
    rmse, pred_stats = RMSEArgmax(), PredStats() 
    setup_learner = CreateLearner(model, data.get_dls(), [rmse, pred_stats], args)
    data.print_info()
    model.print_info()
    setup_learner.print_info()
    learn = setup_learner.get_learner()

    if args.load_learner is not None:
        learn = learn.load(args.load_checkpoint)

    log_dir = Path("training") / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)

    if args.short_epoch < 1:
        learn.add_cb(ShortEpochBothCallback(pct=args.short_epoch, short_valid=False))

    if args.save_model:
        learn.add_cb(SaveModelCallback(every_epoch=False, at_end=False, with_opt=True, reset_on_fit=True, fname=args.save_model))

    if args.wandb:
        learn.add_cb(WandbCallback(log_preds=False))
    else:
        learn.add_cb(TensorBoardCallback(log_dir=log_dir, trace_model=False, log_preds=False))

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True) as prof:
            lrf = learner.lr_find() 
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))
    else:
        lrf = learn.lr_find()

    lr = lrf.valley
    lr = 1e-2
    b = learn.dls.train.one_batch()
    logging.info(f'lr found through lr_find: {lr}')
    logging.info(f'batch len: {len(b[0])}, shape: {b[0].shape}')
    logging.info(f'callbacks added: {learn.cbs}')
    learn.fit(args.max_epochs, lr)

if __name__ == "__main__":
    main()
