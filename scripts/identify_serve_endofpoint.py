from utils import apply_df_overrides, apply_dict_overrides, read_df_from_csv, save_df_to_csv, filter_clip, reset_logging, configure_logging, apply_func_all_clips, r2_dist, not_nan, fname2int, offset_fname
import logging
import argparse
import pandas as pd
import numpy as np
import warnings 
import cv2
from operator import itemgetter
from tqdm import tqdm
from functools import partial
from pathlib import Path
import pickle
from collections import defaultdict

NDARR_COLS =  ['kp','hm','bbox_data']
P1_PLAYER = np.array([[160, 100, 100],[180, 255, 255]]) #red mask
P2_PLAYER = np.array([[  0,   0,  50],[180,  30, 200]]) #gray mask
  
def box_area(b):
    x1,y1,x2,y2,_= b
    return int(abs(x2-x1)*abs(y2-y1))
 
def get_player_color(img_path, img_bbox):
    image = cv2.imread(img_path)
    (x1, y1, x2, y2,_) = map(int,img_bbox)
    roi = image[max(y1,0):y2, max(x1,0):x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    p1_mask = cv2.inRange(hsv, P1_PLAYER[0], P1_PLAYER[1])
    p2_mask = cv2.inRange(hsv, P2_PLAYER[0], P2_PLAYER[1])
    p1_percentage = int(cv2.countNonZero(p1_mask) / (roi.shape[0] * roi.shape[1]) * 100)
    p2_percentage = int(cv2.countNonZero(p2_mask) / (roi.shape[0] * roi.shape[1]) * 100)
    box_c = 'red' if p1_percentage > 0 else 'gray' 
    return box_c  

def select_player_box(bboxes, kps, prev_bb, player_side='n'):
    if len(bboxes)==0: return [np.nan], [np.nan]
    if len(bboxes)==1: return bboxes[0], kps[0]
    box_c_idx = np.argmin(bboxes[:,3]) if player_side=='n' else np.argmax(bboxes[:,3])
    if np.isnan(prev_bb).any(): return bboxes[box_c_idx], kps[box_c_idx]
    box_pp_idx = np.argmin([r2_dist(bbox[2:4],prev_bb[2:4],axis=0, nan_shape=(1,1)) for bbox in bboxes])
    return bboxes[box_pp_idx], kps[box_pp_idx]

def extract_player_bbox_kps(df, local_img_folder):
    results = []
    prev_n_bb, prev_f_bb = [np.nan], [np.nan]
    get_box, get_kp = itemgetter('bbox'), itemgetter('keypoints')
    for idx, (f, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="Adding player attribs"):
        img_path = str(Path(local_img_folder)/f)
        player_boxes = np.array([get_box(item) for item in row.bbox_data])
        player_kps = np.array([get_kp(item) for item in row.bbox_data])
        if player_boxes.shape[0]==0 or max([box_area(box) for box in player_boxes]) > 20000: continue 
        court_center = row.kp.squeeze()[12:14].mean(axis=0)
        player_f = np.array([True if item[3]<court_center[1] else False for item in player_boxes])
        row['n_bb'], row['n_kp'] = select_player_box(player_boxes[~player_f], player_kps[~player_f], prev_n_bb, 'n')
        row['f_bb'], row['f_kp'] = select_player_box(player_boxes[player_f], player_kps[player_f], prev_f_bb, 'f')
        prev_n_bb, prev_f_bb = row['n_bb'], row['f_bb']
        row['n_p'] = get_player_color(img_path, row['n_bb']) if not np.isnan(row['n_bb']).any() else np.nan
        row['f_p'] = get_player_color(img_path, row['f_bb']) if not np.isnan(row['f_bb']).any() else np.nan
        results.append(row)
    results_df = pd.DataFrame(results)
    for clip_num in results_df.clip_num.value_counts().index: #make players side consistent across clip
        clip_df = results_df[results_df.clip_num==clip_num]
        players, counts = np.unique(clip_df['n_p'].fillna('na'), return_counts=True)
        results_df.loc[results_df.clip_num==clip_num, 'n_p'] = players[np.argmax(counts)]
        players, counts = np.unique(clip_df['f_p'].fillna('na'), return_counts=True)
        results_df.loc[results_df.clip_num==clip_num, 'f_p'] = players[np.argmax(counts)]
    return results_df


def estimate_heel_bottom(knee_xy, ankle_xy, extrapolation_factor=1.7):
  knee_point = np.array(knee_xy)
  ankle_point = np.array(ankle_xy)
  if np.isnan(knee_point).any() or np.isnan(ankle_point).any():
      return np.array([np.nan]*4).reshape(2,2)
  leg_vector = ankle_point - knee_point
  heel_point = knee_point + leg_vector * extrapolation_factor
  return heel_point

def add_player_attribs(df):
    df['n_hs'] =  df['n_kp'].map(lambda x: np.array(x)[9:11][:,:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['f_hs'] =  df['f_kp'].map(lambda x: np.array(x)[9:11][:,:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['n_ls'] =  df['n_kp'].map(lambda x: np.array(x)[15:17][:,:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['f_ls'] =  df['f_kp'].map(lambda x: np.array(x)[15:17][:,:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['n_bs']  = df['n_kp'].map(lambda x: np.array(x)[:4][:,:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['f_bs']  = df['f_kp'].map(lambda x: np.array(x)[:4][:,:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['n_bdist'] =  df.apply(lambda row: r2_dist(np.array([row['x'],row['y']])*2,row['n_hs'], axis=1, nan_shape=(2,1))[1 if row.n_p=='red' else 0], axis=1)
    df['n_bdist'] = df['n_bdist'].map(lambda x: x.item() if isinstance(x, np.ndarray) else x)
    df['f_bdist'] =  df.apply(lambda row: r2_dist(np.array([row['x'],row['y']])*2,row['f_hs'], axis=1, nan_shape=(2,1))[1 if row.f_p=='red' else 0], axis=1)
    df['f_bdist'] = df['f_bdist'].map(lambda x: x.item() if isinstance(x, np.ndarray) else x)
    df['n_knees'] =  df.apply(lambda row: row.n_kp[13:15,:2] if not np.isnan(row.n_kp).any() else np.array([np.nan]), axis=1)
    df['n_ankles'] = df.apply(lambda row: row.n_kp[15:17,:2] if not np.isnan(row.n_kp).any() else np.array([np.nan]), axis=1)
    df['f_knees'] =  df.apply(lambda row: row.f_kp[13:15,:2] if not np.isnan(row.f_kp).any() else np.array([np.nan]), axis=1)
    df['f_ankles'] = df.apply(lambda row: row.f_kp[15:17,:2] if not np.isnan(row.f_kp).any() else np.array([np.nan]), axis=1)
    df['n_heels'] =  df.apply(lambda row: estimate_heel_bottom(row.n_knees, row.n_ankles), axis=1)
    df['f_heels'] =  df.apply(lambda row: estimate_heel_bottom(row.f_knees, row.f_ankles, extrapolation_factor=1.2), axis=1)
    df['n_l'] = df['n_kp'].map(lambda x: np.array(x)[15:17][:,:2].mean(axis=0) if not np.isnan(x).any() else np.array([np.nan]))
    df['f_l'] = df['f_kp'].map(lambda x: np.array(x)[15:17][:,:2].mean(axis=0) if not np.isnan(x).any() else np.array([np.nan]))
    df['n_l_t1'] = df['n_l'].shift(1)
    df['f_l_t1'] = df['f_l'].shift(1)
    df['n_l_d'] = ((df['n_l_t1'] - df['n_l'])**2).map(lambda x: np.sqrt(np.sum(x)))
    df['f_l_d'] = ((df['f_l_t1'] - df['f_l'])**2).map(lambda x: np.sqrt(np.sum(x)))
    df['n_l_d'] = np.where(df['clip_num']!=df['clip_num'].shift(1), np.nan, df['n_l_d'])
    df['f_l_d'] = np.where(df['clip_num']!=df['clip_num'].shift(1), np.nan, df['f_l_d'])
    df['n_b'] = df['n_kp'].map(lambda x: np.array(x)[:4].mean(axis=0)[:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['f_b'] = df['f_kp'].map(lambda x: np.array(x)[:4].mean(axis=0)[:2] if not np.isnan(x).any() else np.array([np.nan]))
    df['vec_x1'] = df['x'] - df['x'].shift(1)
    df['vec_y1'] = df['y'] - df['y'].shift(1)
    df['vec_mod'] = np.sqrt(df['vec_x1']**2+df['vec_y1']**2)
    df['vec_mod'] = df['vec_mod'].shift(-1).fillna(method='bfill')
    df['vec_mod_delta'] = df['vec_mod'] - df['vec_mod'].shift(1) 
    return df

def add_serve_attribs(df):
    results = []
    for f, row in df.iterrows():
        hraised, hraised_side, ball_c, bdist, h_idx, lcdist = False, np.nan, np.array([0,0]), None, None, None
        for idx, kp in enumerate([row.n_kp, row.f_kp]):
            if not not_nan(kp): continue
            if not_nan([row.x, row.y]): ball_c = np.array([row.x*2, row.y*2])
            try: 
                head_y, hand_y, leg_y = max(kp[1:3][:,1]), min(kp[9:11][:,1]), np.mean(kp[15:16][:,1])
                dist = min(r2_dist(kp[1:3][:,:2], ball_c))
            except:
                print(f,kp,type(kp),ball_c,np.isnan(ball_c).any(),r2_dist(kp[1:3][:,:2], ball_c))
            if hand_y <= head_y and dist is not None and dist<200: 
                hraised = True
                bdist=dist
                h_idx = idx
                lcdist = None
                hraised_side = 'F' if leg_y<300 else 'N' if leg_y>=300 else np.nan
                if not (row.kp==None).any(): 
                    lcdist = r2_dist(row.kp.squeeze()[[2,3,5,7]].mean(axis=0), row.n_ls.mean(axis=0), axis=0, nan_shape=(1,1)) if idx==0 else r2_dist(row.kp.squeeze()[[0,1,4,6]].mean(axis=0), row.f_ls.mean(axis=0), axis=0)
        row['hraised'] = hraised
        row['hraised_side'] = hraised_side
        row['hraised_bdist'] = bdist
        row['h_idx'] = h_idx
        row['lcdist'] = lcdist
        row['lcdist_inc'] = lcdist<45 if h_idx==1 else lcdist<100 if h_idx==0 else None
        results.append(row)
    df = pd.DataFrame(results)
    return df

def clip_serve_fs(clip_df):
    prev_f_num = 0
    serve_fs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip_df.loc[:,'raised_cumm'] = clip_df['hraised'].rolling(20).sum().shift(-19)
        clip_df.loc[:,'vec_cumm'] = abs(clip_df['vec_x1']).rolling(10, min_periods=5).mean().shift(-9)
        clip_df.loc[:,'d_cumm'] = np.where(clip_df.h_idx==0, abs(clip_df['n_l_d']).rolling(10, min_periods=5).sum(), abs(clip_df['f_l_d']).rolling(10, min_periods=5).sum())
        clip_df.loc[:,'d_cumm'] = clip_df.loc[:,'d_cumm'].shift(2)
        clip_df.loc[:,'d_cumm'] = clip_df.loc[:,'d_cumm'].fillna(method='bfill')
    cond1 = (clip_df.vec_cumm<=1.5) 
    cond2 = (clip_df.raised_cumm>5) 
    cond3 = (clip_df.hraised==True) 
    cond4 = (clip_df.hraised_bdist<85) 
    cond5 = (clip_df.d_cumm<32) 
    cond6 = (clip_df.lcdist_inc==True)
    for f in clip_df[cond1 & cond2 & cond3 & cond4 & cond5 & cond6 ].index.values:
        f_num = fname2int(f)
        if f_num > prev_f_num+30: 
            serve_fs.append(f)
        prev_f_num = f_num
    return serve_fs

def clip_serve_fs_refined(serve_pts, clip_df):
    serve_refined_fs = []
    clip_num = clip_df.iloc[0].clip_num
    fs = serve_pts.get(clip_num)
    for f in fs:
        f_40 = offset_fname(f,40)
        tmp_df = clip_df.loc[f:f_40]
        cond1 = (tmp_df.vec_mod>=8.9) & (~tmp_df.x.isna())
        tmp_df = tmp_df[cond1]
        f_h, vec_mod, vec_mod_delta = tmp_df.index.values[0] if tmp_df.shape[0]>0 else None, \
                                      round(tmp_df.iloc[0]['vec_mod'],1) if tmp_df.shape[0]>0 else None, \
                                      round(tmp_df.iloc[0]['vec_mod_delta'],1) if tmp_df.shape[0]>0 else None
        f_p1, vec_mod_p1 = offset_fname(f_h,-1) if tmp_df.shape[0]>0 else None, \
                           round(clip_df.loc[offset_fname(f_h,-1)]['vec_mod'],1) if tmp_df.shape[0]>0 else None
        if vec_mod_p1 is not None and vec_mod_p1>=5.0: 
            f_h = f_p1
            vec_mod = vec_mod_p1
            vec_mod_p1 = round(clip_df.loc[offset_fname(f_h,-1)]['vec_mod'],1) if tmp_df.shape[0]>0 else None
        if f_h is None: f_h = f
        serve_refined_fs.append(f_h) 
    return serve_refined_fs

def hand2box_dist(bbox, hands):
    if np.isnan(hands).any() or np.isnan(bbox).any():
        return np.nan
    return r2_dist(bbox[:4].reshape(2,2).mean(axis=0),hands)

def calc_hs_dist_cumm(df, column_name):
    result = pd.DataFrame(index=df.index, columns=['A'])
    for f, row in df.iterrows():
        i = df.index.get_loc(f)
        row_slice = slice(max(0,i-10),min(len(df),i))
        result.iloc[i,0] = np.max(df[row_slice][column_name].mean(axis=0))
    return result 

def add_point_end_attribs(df):
    df['n_h2bb'] = df.apply(lambda row: hand2box_dist(row.n_bb, row.n_hs), axis=1)
    df['f_h2bb'] = df.apply(lambda row: hand2box_dist(row.f_bb, row.f_hs), axis=1)
    df['n_h2bb_delta'] = abs(df['n_h2bb'] - df['n_h2bb'].shift(1))
    df['f_h2bb_delta'] = abs(df['f_h2bb'] - df['f_h2bb'].shift(1))
    df['n_h2bb_cumm'] = calc_hs_dist_cumm(df, 'n_h2bb_delta')
    df['f_h2bb_cumm'] = calc_hs_dist_cumm(df, 'f_h2bb_delta')
    df['n_l_d40_calc'] = df['n_l_d'].rolling(40, min_periods=30).mean()
    df['f_l_d40_calc'] = df['f_l_d'].rolling(40, min_periods=30).mean()
    df['n_l_f40_calc'] = df['n_l_d40_calc'].shift(-40)
    df['f_l_f40_calc'] = df['f_l_d40_calc'].shift(-40)
    return df

def clip_end_fs(serve_pts, clip_df):
    cond1 = (clip_df.n_l_d40_calc>0.5) & (clip_df.n_l_d40_calc<3)
    cond2 = (clip_df.f_l_d40_calc>0.5) & (clip_df.f_l_d40_calc<3)
    cond3 = (clip_df.n_l_f40_calc>0.5) & (clip_df.n_l_f40_calc<3)
    cond4 = (clip_df.f_l_f40_calc>0.5) & (clip_df.f_l_f40_calc<3)
    cond5 = (clip_df['n_h2bb_cumm'].rolling(50, min_periods=10).max().shift(-50)<6)
    cond6 = (clip_df['f_h2bb_cumm'].rolling(50, min_periods=10).max().shift(-50)<6)
    cond7 = (clip_df.x.rolling(60, min_periods=0).apply(lambda x: x.isna().sum()).shift(-59)>50) & (clip_df.x.isna())
    prev_f_num = 0 
    end_fs = []
    for f in clip_df.loc[((cond1 & cond2 & cond3 & cond4 & cond5 & cond6) | (cond7))].index.values:
        f_num = fname2int(f)
        if f_num>prev_f_num+120:
            end_fs.append(f)
            prev_f_num = f_num
    clip_num = clip_df.iloc[0].clip_num   
    serve_fs = serve_pts.get(clip_num)  
    end_del_fs = []
    for end_f in end_fs:    
            if len(serve_fs)==0: serve_fs = [clip_df.index.values[0]]
            #print(f'clip {clip_num} {serve_fs}')
            for idx,serve_f in enumerate(serve_fs):
                s_num, e_num = fname2int(serve_f), fname2int(end_f)
                if (((idx==0) & (e_num<s_num)) | ((e_num>s_num-100) & (e_num<s_num+100))):
                    end_del_fs.append(end_f)
    return list(set(end_fs)-set(end_del_fs))

def main(input_file, local_img_folder, serve_override_file, end_override_file, output_df_file, output_dict_file, debug):
    df = read_df_from_csv(input_file, ndarr_cols=NDARR_COLS)
    results_df = extract_player_bbox_kps(df, local_img_folder)
    results_df = add_player_attribs(results_df)
    results_df = add_serve_attribs(results_df)
    serve_pre_pts = apply_func_all_clips(results_df, clip_serve_fs)
    serve_pts = apply_func_all_clips(results_df, partial(clip_serve_fs_refined, serve_pre_pts))
    results_df = add_point_end_attribs(results_df) 
    results_df.to_pickle(output_df_file)
    end_pts = apply_func_all_clips(results_df, partial(clip_end_fs, serve_pre_pts), clip_end_offset=50) 
    apply_dict_overrides(serve_override_file, serve_pts)
    apply_dict_overrides(end_override_file, end_pts)
    with open(output_dict_file, 'wb') as file:
        pickle.dump([dict(serve_pts), dict(end_pts)], file)
    reset_logging()
    configure_logging(debug)
    logging.debug(f'input data shape{df.shape},output data shape {results_df.shape}')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--serve_override_file", required=False, default=None, help="serve override file")
    parser.add_argument("--end_override_file", required=False, default=None, help="end point override file")
    parser.add_argument("--input_file", required=True, help="Path to cleaned merged data file")
    parser.add_argument("--local_img_folder", required=True, help="Path to local image folder")
    parser.add_argument("--output_df_file", required=True, help="Path to output df with attributes added")
    parser.add_argument("--output_dict_file", required=True, help="Path to output dicts for serve and end points")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    main(args.input_file, args.local_img_folder, args.serve_override_file,args.end_override_file, args.output_df_file, args.output_dict_file, args.debug)

