import argparse
from pathlib import Path
from fastbook import *
import numpy as np
import pandas as pd
import torch
from ball_tracking.learner import CreateLearner
from ball_tracking.callbacks import ShortEpochCallbackFixed, ShortEpochBothCallback 
from ball_tracking.metrics import BallPresentRMSE, BallAbsentRMSE, BallPresent5px, BallAbsent5px, PredVarX, PredVarY 
from training.util import DATA_CLASS_MODULE, MODEL_CLASS_MODULE, import_class, setup_data_and_model_from_args
from torch.profiler import profile, record_function, ProfilerActivity
import logging
from ball_tracking.metrics.utils import mask2coord
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
        "--mode", type=str, default="valid", help="valid or test"
    )

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    print(f'TESTING: {temp_args.data_class} and {temp_args.model_class}')
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparser(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparser(model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    """
    parser = _setup_parser()
    args = parser.parse_args()
    #default_device(False)
    data, model = setup_data_and_model_from_args(args)
    rmse_p, rmse_a, bp_5px, ba_5px, pred_rmse_x, pred_rmse_y = BallPresentRMSE(), BallAbsentRMSE(), BallPresent5px(), BallAbsent5px(), PredVarX(), PredVarY()
    setup_learner = CreateLearner(model, data.get_dls(), [rmse_p, rmse_a, bp_5px, ba_5px, pred_rmse_x, pred_rmse_y], args)
    data.print_info()
    model.print_info()
    setup_learner.print_info()
    learn = setup_learner.get_learner()
    def_device = default_device()
    logging.info(f'device: {def_device}')
    if args.load_learner is not None:
        learn = learn.load(args.load_learner)
    test_files = data.get_valid_files()(args.infer_data_path)
    test_files.sort()
    test_dl = learn.dls.test_dl(test_files)
    all_preds, all_ys = torch.tensor([]).to(def_device), torch.tensor([]).to(def_device)
    infer_dl = learn.dls.valid if args.mode=="valid" else test_dl
    for idx,batch in enumerate(infer_dl):
       if idx%100==0: print(f'idx: {idx}')
       pred, y = mask2coord(learn.model.to(def_device)(*batch[:args.num_inp_images])), mask2coord(batch[args.num_inp_images]) if len(batch)==4 else torch.tensor([[0,0]])
       all_preds = torch.cat((all_preds, pred.to(def_device)), axis=0)
       #logging.info(f'idx: {idx}, {pred}')
       all_ys = torch.cat((all_ys, y.to(def_device)), axis=0)
    
    test_name = Path(args.infer_data_path).name 
    results_df = pd.concat(map(pd.DataFrame, (all_preds.cpu().numpy(), all_ys.cpu().numpy())), axis=1)
    results_df.columns = ['pred_x', 'pred_y', 'gt_x', 'gt_y']
    results_df['r2'] = np.sqrt((results_df['pred_x']-results_df['gt_x'])**2+(results_df['pred_y']-results_df['gt_y'])**2).round()
    results_df.index = pd.array(infer_dl.items)
    results_df.to_csv(f'{args.mode}_{test_name}_set_results.csv')


if __name__ == "__main__":
    main()
