import argparse
from pathlib import Path
from fastbook import *
import numpy as np
import pandas as pd
import torch
import wandb 
from ball_tracking.learner import CreateLearner
from ball_tracking.callbacks import ShortEpochCallbackFixed, ShortEpochBothCallback 
from ball_tracking.metrics import BallPresentRMSE, BallAbsentRMSE, BallPresent5px, BallAbsent5px, PredVarX, PredVarY 
from training.util import DATA_CLASS_MODULE, MODEL_CLASS_MODULE, import_class, setup_data_and_model_from_args
from torch.profiler import profile, record_function, ProfilerActivity
import logging
from fastai.callback.tensorboard import * 
from fastai.callback.wandb import *

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
        "--lr",
        type=float,
        default=None,
        help="Specify lr"
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
        "--half_precision",
        action="store_true",
        default=False,
        help="specify whether half precision training"
    )
    parser.add_argument(
        "--schedule_lr",
        action="store_true",
        default=False,
        help="enable reduce lr on plateau scheduler"
    )
    parser.add_argument(
        "--load_learner", type=str, default=None, help="If passed, loads a learner from the provided path."
    )
    parser.add_argument(
        "--save_model", type=str, default=None, help="save learner"
    )
    parser.add_argument(
        "--wandb_project", type=str, default='test', help="project name for wandb experiment"
    )
    parser.add_argument(
        "--num_inp_images_target_img_position", type=str, default=None, help="for wandb experiment"
    )
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
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
    python training/run_experiment.py --model_class=TrackNet --data_class=BallGaussianDataModule --help
    """
    parser = _setup_parser()
    args = parser.parse_args()
    n, t = args.num_inp_images_target_img_position.split(',')
    args.num_inp_images, args.target_img_position = n, t
    logging.info(f'checking if split works: {args.num_inp_images} and {args.target_img_position}')
    with wandb.init(project=args.wandb_project, config=args):
        args = wandb.config
        data, model = setup_data_and_model_from_args(args)
        rmse_p, rmse_a, bp_5px, ba_5px, pred_rmse_x, pred_rmse_y = BallPresentRMSE(), BallAbsentRMSE(), BallPresent5px(), \
                                                                   BallAbsent5px(), PredVarX(), PredVarY()
        setup_learner = CreateLearner(model, data.get_dls(), [rmse_p, rmse_a, bp_5px, ba_5px, pred_rmse_x, pred_rmse_y], args)
        logging.info(f'device: {default_device()}')
        data.print_info()
        model.print_info()
        setup_learner.print_info()
        learn = setup_learner.get_learner()

        if args.load_learner is not None:
            learn = learn.load(args.load_checkpoint)

            log_dir = Path("training") / "logs"
            log_dir.mkdir(exist_ok=True, parents=True)

        if args.save_model:
            learn.add_cb(SaveModelCallback(every_epoch=True, at_end=False, with_opt=True, reset_on_fit=True, fname=args.save_model))

        if args.wandb:
            learn.add_cb(WandbCallback(log_preds=False, log_model=True))
        else:
            learn.add_cb(TensorBoardCallback(log_dir=log_dir, trace_model=False, log_preds=False))

        if args.profile:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],with_stack=True) as prof:
                lrf = learner.lr_find()
                print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=2))

        if args.short_epoch < 1:
            learn.add_cb(ShortEpochBothCallback(pct=args.short_epoch, short_valid=False))

        if args.schedule_lr:
            learn.add_cb(ReduceLROnPlateau(patience=2, factor=10, min_lr=1e-6))

        if args.half_precision:
            learn = learn.to_fp16()

        lr = args.lr
        b = learn.dls.train.one_batch()
        logging.info(f'final lr: {lr}')
        logging.info(f'batch len: {len(b[0])}, shape: {b[0].shape}')
        logging.info(f'callbacks added: {learn.cbs}')
        learn.fit(args.max_epochs, args.lr)

if __name__ == "__main__":
    main()
