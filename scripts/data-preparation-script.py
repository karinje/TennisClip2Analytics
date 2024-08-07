import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import argparse
import json
import pickle
from utils import replace_df_index_prefix
import warnings
from pathlib import Path
def load_bbox_data(file_path):
    def numpy_decoder(obj):
        if isinstance(obj, list):
            try: return np.array(obj)
            except ValueError: pass
        return obj
    with open(file_path, 'r') as file:
        return json.loads(file.read(), object_hook=numpy_decoder)

def preprocess_bbox_data(bbox_data1, bbox_data2=None, local_img_folder=None):
    if bbox_data2: bbox_data1.update(bbox_data2)
    bbox_data = {}
    bbox_prefix =  Path(list(bbox_data1.keys())[0]).parent.as_posix()
    for k, v in bbox_data1.items():
        if local_img_folder: k = k.replace(bbox_prefix, local_img_folder)
        bbox_data[k] = v
    return bbox_data

def load_ball_data(file_path, local_img_folder=None):
    ball_df = pd.read_csv(file_path, index_col=0)
    ball_df.drop(['gt_x', 'gt_y', 'r2'], axis=1, inplace=True)
    ball_df['org_x'], ball_df['org_y'] = ball_df['pred_x'], ball_df['pred_y']
    ball_df.loc[(ball_df.pred_x<=5)|(ball_df.pred_y<=5),['pred_x','pred_y']] = np.nan
    ball_df = ball_df.rename(columns={'pred_x': 'x', 'pred_y': 'y'})
    ball_prefix = Path(ball_df.iloc[0].name).parent.as_posix()
    if local_img_folder: replace_df_index_prefix(ball_df, local_img_folder)
    return ball_df

def load_court_data(file_path, local_img_folder=None):
    kp_df = pd.read_csv(file_path, index_col=0)
    kp_df['kp'] = kp_df['kp'].apply(lambda x: np.array(eval(x[1:-1])))
    kp_df['hm'] = kp_df['hm'].apply(lambda x: np.array(eval(x[1:-1])))
    if local_img_folder: replace_df_index_prefix(kp_df, local_img_folder)
    return kp_df

def merge_dataframes(kp_df, ball_df, bbox_data):
    ball_kp_df = kp_df.join(ball_df)
    ball_kp_df['bbox_data'] = ball_kp_df.index.map(bbox_data)
    return ball_kp_df

def main(bbox_file1, bbox_file2, ball_file, court_file, output_file, local_img_folder):

    bbox_data1 = load_bbox_data(bbox_file1)
    bbox_data2 = load_bbox_data(bbox_file2) if bbox_file2 else None
    bbox_data = preprocess_bbox_data(bbox_data1, bbox_data2, local_img_folder)
    ball_df = load_ball_data(ball_file, local_img_folder)
    kp_df = load_court_data(court_file, local_img_folder)
    results_df = merge_dataframes(kp_df, ball_df, bbox_data)
    results_df.index = results_df.index.map(lambda x: Path(x).name)
    #results_df.columns = ['tmp', 'kp', 'valid', 'hm', 'clip_num', 'x', 'y']
    print(f'final columns: {results_df.columns}') 
    results_df.to_pickle(output_file)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess tennis match data")
    parser.add_argument("--bbox_file1", required=True, help="Path to first bbox data file")
    parser.add_argument("--bbox_file2", help="Path to second bbox data file (optional)")
    parser.add_argument("--ball_file", required=True, help="Path to ball data CSV file")
    parser.add_argument("--court_file", required=True, help="Path to court data CSV file")
    parser.add_argument("--output_file", required=True, help="Path to save preprocessed data")
    parser.add_argument("--local_img_folder", required=True, help="Path to image frames folder")
    args = parser.parse_args()
    
    main(args.bbox_file1, args.bbox_file2, args.ball_file, args.court_file, args.output_file, args.local_img_folder)
