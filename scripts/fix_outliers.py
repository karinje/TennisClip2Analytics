from utils import apply_df_overrides, read_df_from_csv, save_df_to_csv, filter_clip, calc_delta_per_frame, calc_delta_per_frame_backward, reset_logging, configure_logging
import logging
import argparse
import pandas as pd
import numpy as np
import warnings
import pickle
import os
NDARR_COLS =  ['kp','hm','bbox_data']

def fix_outliers(df):
    all_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df.loc[df.x.isna()|df.y.isna(), ['x','y']] = np.nan
        df.loc[:,'x_dp'] = calc_delta_per_frame_backward(df, 'x')
        df.loc[:,'y_dp'] = calc_delta_per_frame_backward(df, 'y')
        df.loc[:,'x_df'] = abs(calc_delta_per_frame(df, 'x'))
        df.loc[:,'y_df'] = abs(calc_delta_per_frame(df, 'y'))

        na_cond = ((df['x_dp'].fillna(50)>=20) & (df['x_df']>=20)) | ((df['x_dp']>=20) & (df['x_df'].fillna(50)>=20)) | \
                  ((df['y_dp'].fillna(50)>=30) & (df['y_df']>=30)) | ((df['y_dp']>=30) & (df['y_df'].fillna(50)>=30))
    
        df.loc[na_cond,['x','y']] = np.nan
        df.loc[:,'x_dp'] = calc_delta_per_frame_backward(df, 'x')
        df.loc[:,'y_dp'] = calc_delta_per_frame_backward(df, 'y')
        df.loc[:,'x_df'] = abs(calc_delta_per_frame(df, 'x'))
        df.loc[:,'y_df'] = abs(calc_delta_per_frame(df, 'y'))

    prev_flag = False
    for idx, row in df.iterrows():
        if (abs(row.x_dp)>=20 or abs(row.y_dp)>=30):
            if not prev_flag:
                row['flag'] = True
                prev_flag = True
                count = 1
            else:
                row['flag'] = False
                prev_flag = False
        else:
            if prev_flag and (np.isnan((row.x,row.y)).any() or count>3):
                row['flag'] = False
                prev_flag = False
                count = 0
            elif prev_flag:
                row['flag'] = True
                prev_flag = True
                count += 1
            else:
                row['flag'] = False
                prev_flag = False
        all_rows.append(row)
    out_df = pd.DataFrame(all_rows)
    out_df.loc[out_df.flag==True, ['x','y']] = np.nan
    return out_df

def main(input_file, override_file, output_file, debug):
    
    df = pd.read_pickle(input_file)
    if os.path.exists(override_file): apply_df_overrides(override_file, df)
    clean_df = pd.DataFrame([])
    for clip_num in sorted(df.clip_num.value_counts().index.values):
            clip_df = fix_outliers(fix_outliers(filter_clip(df, clip_num)))
            clean_df = pd.concat([clean_df,clip_df], axis=0)
    clean_df.to_pickle(output_file)
    reset_logging()
    configure_logging(debug)
    logging.debug(f'{df.shape}, {clean_df.shape}')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Clean ball location data")
    parser.add_argument("--override_file", required=True, help="Path to ball location override file")
    parser.add_argument("--input_file", required=True, help="Path to merged data file")
    parser.add_argument("--output_file", required=True, help="Path to output data file")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    main(args.input_file, args.override_file, args.output_file, args.debug)

