import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
from operator import itemgetter
import json

def offset_fname(f, offset, prefix='img_', suffix='.jpg'):
    img_num = int(f.replace(prefix,'').replace(suffix,'')) + offset
    return f"{prefix}{str(img_num).zfill(4)}{suffix}"

def fname2int(f, offset=0, prefix='img_', suffix='.jpg'):
    return int(f.replace(prefix,'').replace(suffix,'')) + offset

def r2_dist(a, b, axis=1, nan_dim=2):
    if not isinstance(a, np.ndarray): a = np.array(a)
    if not isinstance(b, np.ndarray): b = np.array(b)
    
    if np.isnan(a).any() or np.isnan(b).any(): 
        return np.nan if nan_dim == 1 else [np.nan, np.nan] * nan_dim
    return np.sqrt(np.sum((a-b)**2, axis=axis))

def box_area(b):
    x1, y1, x2, y2, _ = b
    return int(abs(x2-x1) * abs(y2-y1))

def select_player_box(bboxes, kps, prev_bb, player_side='n'):
    if len(bboxes) == 0: return [np.nan], [np.nan]
    if len(bboxes) == 1: return bboxes[0], kps[0]
    box_c_idx = np.argmin(bboxes[:, 3]) if player_side == 'n' else np.argmax(bboxes[:, 3])
    if np.isnan(prev_bb).any(): return bboxes[box_c_idx], kps[box_c_idx]
    box_pp_idx = np.argmin([r2_dist(bbox[2:4], prev_bb[2:4], axis=0) for bbox in bboxes])
    return bboxes[box_pp_idx], kps[box_pp_idx]

def process_player_data(df):
    prev_n_bb, prev_f_bb = [np.nan], [np.nan]
    results = []
    for idx, (f, row) in enumerate(df.iterrows()):
        bbox_data = row['bbox_data']
        get_box, get_kp = itemgetter('bbox'), itemgetter('keypoints')
        player_boxes = np.array([get_box(item) for item in bbox_data])
        player_kps = np.array([get_kp(item) for item in bbox_data])
        if player_boxes.shape[0] == 0 or max([box_area(box) for box in player_boxes]) > 20000: 
            continue 
        court_center = row.kp[12:14].mean(axis=0)
        player_f = np.array([True if item[3] < court_center[1] else False for item in player_boxes])
        row['n_bb'], row['n_kp'] = select_player_box(player_boxes[~player_f], player_kps[~player_f], prev_n_bb, 'n')
        row['f_bb'], row['f_kp'] = select_player_box(player_boxes[player_f], player_kps[player_f], prev_f_bb, 'f')
        prev_n_bb, prev_f_bb = row['n_bb'], row['f_kp']
        results.append(row)
    return pd.DataFrame(results)

def calculate_player_features(df):
    df['n_hs'] = df['n_kp'].apply(lambda x: np.array(x)[9:11][:,:2] if isinstance(x, np.ndarray) else np.array([]))
    df['f_hs'] = df['f_kp'].apply(lambda x: np.array(x)[9:11][:,:2] if isinstance(x, np.ndarray) else np.array([]))
    df['n_ls'] = df['n_kp'].apply(lambda x: np.array(x)[15:17][:,:2] if isinstance(x, np.ndarray) else np.array([]))
    df['f_ls'] = df['f_kp'].apply(lambda x: np.array(x)[15:17][:,:2] if isinstance(x, np.ndarray) else np.array([]))
    df['n_bs'] = df['n_kp'].apply(lambda x: np.array(x)[:4][:,:2] if isinstance(x, np.ndarray) else np.array([]))
    df['f_bs'] = df['f_kp'].apply(lambda x: np.array(x)[:4][:,:2] if isinstance(x, np.ndarray) else np.array([]))
    
    df['n_bdist'] = df.apply(lambda row: r2_dist(np.array([row['pred_x'],row['pred_y']])*2, row['n_hs'], axis=1)[1 if row.n_p=='red' else 0] if len(row['n_hs']) > 0 else np.nan, axis=1)
    df['f_bdist'] = df.apply(lambda row: r2_dist(np.array([row['pred_x'],row['pred_y']])*2, row['f_hs'], axis=1)[1 if row.f_p=='red' else 0] if len(row['f_hs']) > 0 else np.nan, axis=1)

    df['n_l'] = df['n_kp'].apply(lambda x: np.array(x)[15:17][:,:2].mean(axis=0) if isinstance(x, np.ndarray) else np.array([]))
    df['f_l'] = df['f_kp'].apply(lambda x: np.array(x)[15:17][:,:2].mean(axis=0) if isinstance(x, np.ndarray) else np.array([]))
    df['n_l_t1'] = df['n_l'].shift(1)
    df['f_l_t1'] = df['f_l'].shift(1)
    df['n_l_d'] = df.apply(lambda row: np.sqrt(np.sum((row['n_l_t1'] - row['n_l'])**2)) if len(row['n_l']) > 0 and len(row['n_l_t1']) > 0 else np.nan, axis=1)
    df['f_l_d'] = df.apply(lambda row: np.sqrt(np.sum((row['f_l_t1'] - row['f_l'])**2)) if len(row['f_l']) > 0 and len(row['f_l_t1']) > 0 else np.nan, axis=1)
    df['n_l_d'] = np.where(df['clip_num'] != df['clip_num'].shift(1), np.nan, df['n_l_d'])
    df['f_l_d'] = np.where(df['clip_num'] != df['clip_num'].shift(1), np.nan, df['f_l_d'])

    return df

def detect_serve_points(df):
    serve_pts = defaultdict(list)
    for clip_num in df.clip_num.unique():
        prev_f_num = 0
        clip_df = df[df.clip_num == clip_num]
        clip_df.loc[:,'raised_cumm'] = clip_df['hraised'].rolling(20).sum().shift(-19)
        clip_df.loc[:,'vec_cumm'] = abs(clip_df['vec_x1']).rolling(10, min_periods=5).mean().shift(-9)
        clip_df.loc[:,'d_cumm'] = np.where(clip_df.h_idx==0, abs(clip_df['n_l_d']).rolling(10, min_periods=5).sum(), abs(clip_df['f_l_d']).rolling(10, min_periods=5).sum())
        clip_df.loc[:,'d_cumm'] = clip_df.loc[:,'d_cumm'].shift(2)
        clip_df.loc[:,'d_cumm'] = clip_df.loc[:,'d_cumm'].fillna(method='bfill')
        for f in clip_df[((clip_df.vec_cumm<=1.5) & (clip_df.raised_cumm>5) & (clip_df.hraised==True) & (clip_df.hraised_bdist<85) & (clip_df.d_cumm<32) & (clip_df.lcdist_inc==True))].index.values:
            f_num = fname2int(f)
            if f_num > prev_f_num + 30:
                serve_pts[clip_num].append(f)
            prev_f_num = f_num
    return serve_pts

def detect_return_points(df):
    return_pts = defaultdict(list)
    for clip_num in df.clip_num.unique():
        prev_f_num, fs, final_fs = 1e6, [], []
        turn_df = df[(df.clip_num==clip_num) & (df.turn==True)]
        for f in turn_df.index.values:
            f_num = fname2int(f)
            x_turn, y_turn = ~turn_df.loc[f,'xturn'], ~turn_df.loc[f,'yturn']
            if f_num < prev_f_num + 20:
                fs.append((f, x_turn, y_turn))
            elif len(fs) > 0:
                final_fs.append(sorted(fs, key=lambda x: (x[1], x[2], x[0]))[0][0])
                fs = [(f, x_turn, y_turn)]
            else:
                fs = [(f, x_turn, y_turn)]
            prev_f_num = f_num
        if len(fs) > 0:
            final_fs.append(sorted(fs, key=lambda x: (x[1], x[2], x[0]))[0][0])
        return_pts[clip_num] = final_fs
    return return_pts

def detect_end_points(df):
    end_pts = defaultdict(list)
    cond1 = (df.n_l_d40_calc > 0.5) & (df.n_l_d40_calc < 3)
    cond2 = (df.f_l_d40_calc > 0.5) & (df.f_l_d40_calc < 3)
    cond3 = (df.n_l_f40_calc > 0.5) & (df.n_l_f40_calc < 3)
    cond4 = (df.f_l_f40_calc > 0.5) & (df.f_l_f40_calc < 3)
    cond5 = (df['n_h2bb_cumm'].rolling(50, min_periods=10).max().shift(-50) < 6)
    cond6 = (df['f_h2bb_cumm'].rolling(50, min_periods=10).max().shift(-50) < 6)
    cond7 = (df.pred_x.rolling(60