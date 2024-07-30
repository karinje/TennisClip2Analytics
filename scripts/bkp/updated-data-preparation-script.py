import numpy as np
import pandas as pd
from pathlib import Path
import cv2
import argparse
import json

def load_bbox_data(file_path):
    def numpy_decoder(obj):
        if isinstance(obj, list):
            try:
                return np.array(obj)
            except ValueError:
                pass
        return obj
    
    with open(file_path, 'r') as file:
        return json.loads(file.read(), object_hook=numpy_decoder)

def preprocess_bbox_data(bbox_data1, bbox_data2=None):
    if bbox_data2:
        bbox_data1.update(bbox_data2)
    bbox_data = {}
    for k, v in bbox_data1.items():
        k = k.replace('/home/ubuntu/test-storage/Users/sanjaykarinje/Downloads/match_frames_v2/', '/Users/sanjaykarinje/Downloads/')
        bbox_data[k] = v
    return bbox_data

def load_ball_data(file_path):
    ball_df = pd.read_csv(file_path, index_col=0)
    ball_df.drop(['gt_x', 'gt_y', 'r2'], axis=1, inplace=True)
    return ball_df

def load_court_data(file_path):
    kp_df = pd.read_csv(file_path, index_col=0)
    kp_df['kp'] = kp_df['kp'].apply(lambda x: np.array(eval(x[1:-1])))
    kp_df['hm'] = kp_df['hm'].apply(lambda x: np.array(eval(x[1:-1])))
    return kp_df

def merge_dataframes(kp_df, ball_df, bbox_data):
    ball_kp_df = kp_df.join(ball_df)
    ball_kp_df.loc[ball_kp_df.pred_x <= 5, 'pred_x'] = np.nan
    ball_kp_df.loc[ball_kp_df.pred_y <= 5, 'pred_y'] = np.nan
    ball_kp_df.index = ball_kp_df.index.str.replace('/home/ubuntu/test-storage/Users/sanjaykarinje/Downloads/match_frames_v2', '/Users/sanjaykarinje/Downloads')
    ball_kp_df['bbox_data'] = ball_kp_df.index.map(bbox_data)
    ball_kp_df['kp'] = ball_kp_df['kp'].apply(lambda x: x.tolist())
    ball_kp_df['hm'] = ball_kp_df['hm'].apply(lambda x: x.tolist())
    return ball_kp_df

def main(bbox_file1, bbox_file2, ball_file, court_file, output_file):
    bbox_data1 = load_bbox_data(bbox_file1)
    bbox_data2 = load_bbox_data(bbox_file2) if bbox_file2 else None
    bbox_data = preprocess_bbox_data(bbox_data1, bbox_data2)
    
    ball_df = load_ball_data(ball_file)
    kp_df = load_court_data(court_file)
    
    results_df = merge_dataframes(kp_df, ball_df, bbox_data)
    
    # Save the preprocessed data
    results_df.to_csv(output_file)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess tennis match data")
    parser.add_argument("--bbox_file1", required=True, help="Path to first bbox data file")
    parser.add_argument("--bbox_file2", help="Path to second bbox data file (optional)")
    parser.add_argument("--ball_file", required=True, help="Path to ball data CSV file")
    parser.add_argument("--court_file", required=True, help="Path to court data CSV file")
    parser.add_argument("--output_file", required=True, help="Path to save preprocessed data")
    
    args = parser.parse_args()
    
    main(args.bbox_file1, args.bbox_file2, args.ball_file, args.court_file, args.output_file)
