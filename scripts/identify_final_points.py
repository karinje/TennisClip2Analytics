from utils import apply_df_overrides, apply_dict_overrides, filter_clip, reset_logging, configure_logging, apply_func_all_clips, r2_dist, not_nan, check_nan, fname2int, offset_fname, drop_df_overrides
import math
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
from itertools import zip_longest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import types
from functools import partial
from itertools import starmap
import numpy.linalg as LA
import os
col_x_v1_p1, col_x_v1_p2, col_x_v2_p1, col_x_v2_p2 = 10, 13, 8, 12
col_y_v1_p1, col_y_v1_p2, col_y_v2_p1, col_y_v2_p2 = 10, 8, 13, 12
col_z_v1_p1, col_z_v1_p2, col_z_v2_p1, col_z_v2_p2 = 14, 15, 16, 17 #'LeftB', 'LeftTop', 'RightB', 'RightTop'

def tuple_shot(s_t, df, plot_chart=False, local_img_folder=None):
    f, side, ty = s_t
    if side in ('N','F'):
        f_idx = df.index.get_loc(f)
        sel_cols = ['n_hs','f_hs','n_p','f_p','n_b','f_b','x','y']
        row = df.loc[f,sel_cols]
        shot = None
        player = row.n_p if side[-1]=='N' else row.f_p
        body_pt = row.n_b if side[-1]=='N' else row.f_b
        hands = row.n_hs.mean(axis=0) if ((not np.isnan(row.n_hs).any()) and (side[-1]=='N')) else \
                (row.f_hs.mean(axis=0) if ((not np.isnan(row.f_hs).any()) and (side[-1]=='F')) else [np.nan, np.nan])
        ball = np.array([row.x,row.y])*2
        compare_pt = hands if np.isnan(ball).any() else ball
        try:
            shot = 'BH' if ((side=='F') ^ (player=='gray') ^ (compare_pt[0]>=body_pt[0]))==0 else 'FH'
        except:
            print(body_pt)
        if ty =='S': shot='S'
        if plot_chart:
            image = cv2.imread(str(local_img_folder/f))
            for idx, (x,y) in enumerate(df.iloc[f_idx-10:f_idx+10][['x','y']].values*2):
                if not (np.isnan(x) or np.isnan(y)): 
                    color = (0, 255, 0) if idx==10 else (255, 0, 0)
                    cv2.circle(image, (int(x), int(y)), 3, color , -1)
            plt.figure(figsize=(6,4))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"{f}_{side}_{shot}")
        return shot
    else:
        return None


def calculate_P(df, f, local_img_folder, court_kp_file, plot_chart=False, figsize=(7,4)):
    mod_kp = pd.read_csv(court_kp_file, header=None).values
    ref_net_kp = mod_kp[14:][:,None,:]
    img_path = local_img_folder/f
    hm = df.loc[f,['hm']][0]
    court_kp = df.loc[f,['kp']][0].squeeze()
    net_kp = cv2.perspectiveTransform(ref_net_kp.reshape(-1,1,2), hm).squeeze()
    court_net_kp = np.concatenate((court_kp, net_kp), axis=0)
    for name in sorted(globals()):
      if name.startswith('col_') and not callable(globals()[name]):
        globals()[name[4:]] = court_net_kp[globals()[name]]
    van_pts = []
    plines = [np.cross((*eval(ax+'_'+line+'_p1'), 1), (*eval(ax+'_'+line+'_p2'), 1)) for ax in ('x','y','z') for line in ('v1','v2')]
    for i in range(0, len(plines), 2):
      van_pts.append(np.cross(*plines[i:i+2]))
    van_pts = np.array(van_pts)
    if np.isnan(van_pts).any(): return np.nan
    van_pts = van_pts / van_pts[:,-1].reshape(-1,1)
    srvcbox_ntpost = np.cross(  np.cross((*eval('z_v1_p1'), 1),  van_pts[1]), np.cross((*eval('x_v1_p1'), 1), (*eval('x_v1_p2'), 1))  )
    srvcbox_ntpost = srvcbox_ntpost / srvcbox_ntpost[-1]
    wo, ref_x, ref_y, ref_z = map(np.array, ((*eval('z_v1_p1'), 1), (*eval('z_v2_p1'), 1), srvcbox_ntpost, (*eval('z_v1_p2'), 1)))
    ref_mat = np.array([ref_x, ref_y, ref_z])
    real_lens = np.array([33*12, -21.5*12, 42])
    a_mat, b_mat = van_pts-ref_mat, ref_mat-wo
    if plot_chart:
        def draw_line(img, p1, p2, ax, thickness=2):
              cv2.line(img, tuple(map(int,p1)), tuple(map(int,p2)), colors.get(ax), thickness)
        colors = {'x': (255,0,0), 'y': (0,255,0), 'z': (0,0,255)} #red, green, blue
        plt.figure(figsize=figsize)
        img = cv2.imread(str(img_path))
        list(starmap(partial(draw_line,img),[(eval(ax+'_'+line+'_p1'), eval(ax+'_'+line+'_p2'), ax) for ax in ('x','y','z') for line in ('v1','v2')]))
        tmp_p = (eval('x_v1_p1')-eval('x_v1_p2'))+eval('x_v1_p2')
        logging.debug(f"line 1 end_pts: {tmp_p} {eval('x_v1_p2')}")
        draw_line(img, tmp_p, eval('x_v1_p2'), 'x')
        logging.debug(f"line 2 end_pts: {wo[:-1]} {ref_x[:-1]}")
        draw_line(img, wo[:-1], ref_x[:-1], 'x')
        logging.debug(f"line 3 end_pts: {wo[:-1]} {ref_y[:-1]}")
        draw_line(img, wo[:-1], ref_y[:-1], 'y')
        logging.debug(f"line 4 end_pts: {wo[:-1]} {ref_z[:-1]}")
        draw_line(img, wo[:-1], ref_z[:-1], 'z')
        ball_c = df.loc[f][['x','y']].values*2
        heels = max(df.loc[f].f_heels, key=lambda x: x[1]) if df.loc[f].side=='F' else min(df.loc[f].n_heels, key=lambda x: x[1])
        cv2.circle(img, (int(ball_c[0]), int(ball_c[1])), 4, colors['x'], -1)
        cv2.circle(img, (int(heels[0]), int(heels[1])), 2, colors['x'], -1)
        plt.title(f'{f}')
        plt.imshow(img)
    try:
        scale_factors = np.array([np.linalg.lstsq(a.reshape(1,-1).T , b.T , rcond=None)[0]/r_len for a,b,r_len in zip(a_mat, b_mat, real_lens)])
    except:
         return np.nan
    return np.concatenate(((van_pts*scale_factors).T, wo.reshape(-1,1)), axis=1)

def compute_2d_3d_with_plane(pt_2d, serve_plane, P, ret_ft=True):
    P_updated = np.concatenate([P.T, serve_plane.reshape(-1,1)], axis=1)
    P_updated_inv = np.linalg.inv(P_updated)
    pt_h_0 = np.array((*pt_2d,1,0)) 
    vals = pt_h_0 @ P_updated_inv
    vals = vals/vals[-1]
    return vals[:-1] if not ret_ft else np.around(vals[:-1]/12, 3)

def compute_2d_3d_with_y(pt_2d, y_coord, P):
    PT_inv = np.linalg.inv(P.T[:-1])
    pt_h = np.array((*pt_2d,1))
    z = (P.T[-1] @ PT_inv[:,-2] + y_coord) / (pt_h @ PT_inv[:,-2])
    pt_h_scaled = z * pt_h
    vals = pt_h_scaled - P.T[-1]
    pt_3d = vals @ PT_inv
    return pt_3d 

def compute_plane(df, f1, f2, P1, P2, f2_type='B'):
    row1, row2 = df.loc[f1], df.loc[f2]
    try:
        ank1, ball1, side1 = max(row1.f_heels, key=lambda x: x[1]) if row1.side=='F' else min(row1.n_heels, key=lambda x: x[1]), row1[['x','y']].values*2, row1.side
        ank2, ball2, side2 = max(row2.f_heels, key=lambda x: x[1]) if row2.side=='F' else min(row2.n_heels, key=lambda x: x[1]), row2[['x','y']].values*2, row2.side
    except:
        return np.nan, (np.nan, np.nan)
    z_plane = np.array([0,0,1,0])
    ank1_3d = compute_2d_3d_with_plane(ank1, z_plane, P1, ret_ft=False)
    logging.debug(f'ank1: {ank1_3d/12}')
    ball1_3d = compute_2d_3d_with_y(ball1, ank1_3d[-2], P1) #compute ball 3d assuming y_coord is same as ankle
    logging.debug(f'ball1 3d: {ball1_3d/12}')
    if f2_type=='B':
        ball2_3d = compute_2d_3d_with_plane(ball2, z_plane, P2, ret_ft=False)
    else: 
        ank2_3d = compute_2d_3d_with_plane(ank2, z_plane, P2, ret_ft=False)
        ball2_3d = compute_2d_3d_with_y(ball2, 0 if f2_type=='T' else ank2_3d[-2] , P2) 
    logging.debug(f'ball2 3d: {ball2_3d/12}')
    abc = np.cross(ball1_3d-ball2_3d, np.array([0,0,1]))
    serve_plane = np.concatenate([abc, np.array(-np.dot(abc, ball1_3d).reshape(1))])
    logging.debug(f'pt1_3d: {ball1_3d/12} pt2_3d: {ball2_3d/12}')
    return serve_plane, (ball1_3d/12, ball2_3d/12)

def compute_2d_3d(df, img_name, serve_plane, P, t=None):
  row = df.loc[img_name]
  P_updated = np.concatenate([P.T, serve_plane.reshape(-1,1)], axis=1)
  P_updated_inv = np.linalg.inv(P_updated)
  ball_h_0 = np.array((*row[['x','y']].values*2,1,0)) if t is None else np.array(((*t,1,0)))
  vals = ball_h_0 @ P_updated_inv
  vals = vals/vals[-1]
  return np.around(vals[:-1]/12,3)  

def y_dist(a,b):
  return np.abs(a[1]-b[1])

def x_dist(a,b):
  return np.abs(a[0]-b[0])

def xy_dist(a,b):
    return r2_dist(a,b,axis=0,nan_dim=1)

def ft_miles_hour(x, frames=1, fps=30):
  return x*0.000189394*fps*60*60/frames

def neg_side(side):
    return 'N' if side[-1]=='F' else 'F' 

def compute_speed(df, img_name1, img_name2, P1, P2, dist_func, sp, fps=30, frames=None):
    if not not_nan(sp): return np.nan
    if frames is None: frames = fname2int(img_name2) - fname2int(img_name1)
    pt1_2d, pt2_2d = df.loc[img_name1, ['x','y']].values*2, df.loc[img_name2, ['x','y']].values*2
    pt1, pt2 = compute_2d_3d_with_plane(pt1_2d, sp, P1), compute_2d_3d_with_plane(pt2_2d, sp, P2)
    logging.debug(f'{img_name1} {img_name2} {frames} {pt1} {pt2} {np.around(ft_miles_hour(dist_func(pt1, pt2), fps=fps, frames=frames),0)}')
    return np.around(ft_miles_hour(dist_func(pt1, pt2), fps=fps, frames=frames),0)

def compute_max_speed(df, img_name1, img_name2, f1_offsets=4, dist_func=y_dist, plot_chart=False, f2_type='B', ret_ends=False, local_img_folder=None, court_kp_file=None):
    P1, P2 = calculate_P(df, img_name1, local_img_folder, court_kp_file, plot_chart=plot_chart), calculate_P(df, img_name2, local_img_folder, court_kp_file, plot_chart=plot_chart)
    serve_plane, end_pts = compute_plane(df, img_name1, img_name2, P1, P2, f2_type=f2_type)
    if np.isnan(serve_plane).any(): return (np.nan,np.nan) if not ret_ends else (np.nan, np.nan),(np.nan, np.nan)
    serve_s = compute_speed(df, img_name1, img_name2, P1, P2, dist_func=dist_func, sp=serve_plane)
    all_s = []
    slice_df = df.loc[img_name1:img_name2]
    all_fs = slice_df[(slice_df.x.notna()) & (slice_df.y.notna())].index.values[:f1_offsets]
    logging.debug(f'all_fs: {all_fs}')
    speed_dict = {}
    for f1 in all_fs:
        f2 = img_name2
        P1, P2 = calculate_P(df, f1, local_img_folder, court_kp_file, plot_chart=plot_chart), calculate_P(df, f2, local_img_folder, court_kp_file, plot_chart=False)
        s_avg = compute_speed(df, f1, f2, P1, P2, dist_func=dist_func, sp=serve_plane)
        n_frames = fname2int(f2)-fname2int(f1)
        s_ini = s_avg + n_frames*(n_frames-1)/2 * 0.2
        s_ini = s_avg + (n_frames-1)/2 * (s_avg**2) * 3.5e-4
        speed_dict[(f1,f2)] = s_ini
    logging.debug(f'speed_dict: {sorted(speed_dict.items(), key=lambda item: item[1], reverse=True)}')
    first_item, second_item =  sorted(speed_dict.items(), key=lambda item: item[1], reverse=True)[0],  sorted(speed_dict.items(), key=lambda item: item[1], reverse=True)[1]
    if first_item[1]>140:
        return second_item if (not ret_ends) else (second_item, end_pts)
    else:
        return first_item if (not ret_ends) else (first_item, end_pts)

def gen_points_df(final_pts_dict, df, local_img_folder):
    rows = []
    for clip_num, pt_tups in final_pts_dict.items():
        for pt_idx, pt_tup in enumerate(pt_tups):
            for s_t in pt_tup:
                shot = tuple_shot(s_t, df, plot_chart=False, local_img_folder=local_img_folder)
                rows.append([clip_num, pt_idx, *s_t, shot])
    return pd.DataFrame(rows, columns=['clip_num', 'pt_idx', 'img_name', 'side', 'return_type', 'shot_type'])

def validate_points_df(points_df):
    for clip_num in sorted(points_df.clip_num.value_counts().index.values):
        points_clip_df = filter_clip(points_df, clip_num).reset_index(drop=True)
        for idx, row in points_clip_df.iterrows():
            if idx==len(points_clip_df)-1: break
            next_row = points_clip_df.iloc[idx+1]
            if (row.side in ('N','F')) and not ((next_row.side in ('B')) or next_row.return_type in ('T') or next_row.shot_type in ('S')):
                raise Exception(f"Points df validation failed - clip:{clip_num} {idx} {len(points_clip_df)} {row.img_name,row.side} {next_row.img_name,next_row.side}")

def compute_heels_3d(row, df, local_img_folder, court_kp_file, col='n_heels'):
    f = row.img_name
    if check_nan(df.loc[f,col]): return np.array([np.nan, np.nan, np.nan])
    heels_2d = df.loc[f,col].mean(axis=0)
    heels_2d = max(df.loc[f,col], key=lambda x: x[1]) if col=='f_heels' else (min(df.loc[f,col], key=lambda x: x[1]) if col=='n_heels' else np.nan)
    P = calculate_P(df, f, local_img_folder, court_kp_file)
    heels_3d = np.array([*heels_2d,1]) @ LA.inv(P.T[[0,1,3]])
    heels_3d = np.around(np.array([*heels_3d[:2]/heels_3d[-1],0])/12,1)
    logging.debug(f'f: {f} heels_2d: {heels_2d} heels_3d: {heels_3d}')
    return heels_3d

def compute_hit_angle(pt1, pt2, side):
    if check_nan(pt1) or check_nan(pt2): return np.nan
    hit_v = pt2 - pt1
    y_axis = np.array([0, 1, 0])
    theta = np.degrees( np.arccos( np.dot(hit_v, y_axis) / np.sqrt(np.sum(hit_v**2)) ) )
    theta = 180-theta if theta>90 else theta
    theta = theta if ((hit_v[0]>0 and side=='N')|(hit_v[0]<0 and side=='F')) else -theta 
    return theta 

def assign_results(df):
    for idx,row in df.iterrows():
        prev_idx = df.index[df.index.get_loc(idx)-1]
        if row.return_type in ('T'):
            df.loc[idx,'result'] = 'L'
            df.loc[prev_idx,'result'] = 'W'
        elif row.return_type in ('E'):
            if row.shot_type=='S':
                cond1 = (row.ball_3d[0]<=16.5 and row.ball_3d_f1[0]>=16.5 and row.ball_3d_f1[0]<=30)
                cond2 = (row.ball_3d[0]>16.5 and row.ball_3d_f1[0]<=16.5 and row.ball_3d_f1[0]>=3)
                cond3 = row.b2bl>=18
                logging.debug(f'cond1 {cond1}  cond2 {cond2} cond3 {cond3}')
                if (cond1 or cond2) and cond3:
                    logging.debug(f'entered S and inside 1')
                    df.loc[idx,'result'] = 'W'
                    df.loc[prev_idx,'result'] = 'L'
                else:
                    logging.debug(f'entered S and inside 2')
                    df.loc[idx,'result'] = 'L'
                    df.loc[prev_idx,'result'] = 'W'
            elif (row.shot_type!='S') and (row.b2sl<0 or row.b2bl<0):
                logging.debug(f'entered S and inside 3')
                df.loc[idx,'result'] = 'L'
                df.loc[prev_idx,'result'] = 'W'
            elif (row.shot_type!='S') and (row.b2sl>=0 and row.b2bl>=0):
                logging.debug(f'not S and inside 3')
                df.loc[idx,'result'] = 'W'
                df.loc[prev_idx,'result'] = 'L'
        elif row.return_type in ('R'):
            df.loc[idx,'result'] = 'R'
        else:
            df.loc[idx,'result'] = 'O'

def gen_point_attribs(points_df, clip_num, df, local_img_folder, court_kp_file):
    point_clip_df = filter_clip(points_df, clip_num).reset_index(drop=True)
    point_clip_df['f_heels'] = point_clip_df.apply(lambda row: compute_heels_3d(row, df, local_img_folder, court_kp_file, col='f_heels'), axis=1)
    point_clip_df['n_heels'] = point_clip_df.apply(lambda row: compute_heels_3d(row, df, local_img_folder, court_kp_file, col='n_heels'), axis=1)
    all_rows = []
    for idx, row in point_clip_df.iterrows():
        row['player'] = df.loc[row.img_name].f_p if row.side=='F' else (df.loc[row.img_name].n_p if row.side=='N' else np.nan)
        next_row = point_clip_df.iloc[idx+1] if idx<len(point_clip_df)-1 else pd.Series(np.nan, index=row.index)
        logging.debug(f'next row: {row.img_name} {next_row.img_name}')
        if row.side in ('N','F') and type(next_row.img_name) is str:
            f2_type = 'T' if next_row.return_type=='T' else next_row.side
            (_, x_max_s), _ = compute_max_speed(df, row.img_name, next_row.img_name, f1_offsets=6, dist_func=x_dist, plot_chart=False, f2_type=f2_type, ret_ends=True, local_img_folder=local_img_folder, court_kp_file=court_kp_file)
            (_, y_max_s), end_pts = compute_max_speed(df, row.img_name, next_row.img_name, f1_offsets=6, dist_func=y_dist, plot_chart=False, f2_type=f2_type, ret_ends=True, local_img_folder=local_img_folder, court_kp_file=court_kp_file)
            if x_max_s>40 or y_max_s>150: print('Out of range speed')
            row['x_speed'] = round(x_max_s) if not np.isnan(x_max_s) and x_max_s<40 else np.nan
            row['y_speed'] = round(y_max_s) if not np.isnan(y_max_s) and y_max_s<150 else np.nan
            end_pts = np.around(end_pts,1) if (not check_nan(end_pts[0]) and not check_nan(end_pts[1])) else (np.nan, np.nan)
            row['ball_3d'], next_pt = end_pts
            row['ball_3d_f1'] = next_pt
            cur_heels, opp_heels = (row.n_heels, row.f_heels) if row.side=='N' else (row.f_heels, row.n_heels)     
            row['hh2c'], row['hh2n'] = (16.5 - cur_heels[0],  -cur_heels[1]) if row.side=='N' else (-16.5 + cur_heels[0],  cur_heels[1]) #hit heels to center line
            row['oh2c'], row['oh2n'] = (16.5 - opp_heels[0],  -opp_heels[1]) if neg_side(row.side)=='N' else (-16.5 + opp_heels[0],  opp_heels[1]) #oppsite heels to center line
            row['r2b_theta'] = np.around(compute_hit_angle(row['ball_3d'], next_pt, row.side),1) #return to bounce theta 
            row['r2h_theta'] = np.around(compute_hit_angle(row['ball_3d'], opp_heels, row.side),1)
            if next_row.side=='B' and not check_nan(next_pt):
                logging.debug(f'next_pt: {next_pt}, opp_heels: {opp_heels}')
                row['b2sl'], row['b2bl'] = min(next_pt[0]-3, 30-next_pt[0]), 39-(-next_pt[1] if neg_side(row.side)=='N' else next_pt[1])
                row['b2h_x'], row['b2h_y'], _ = (next_pt - opp_heels) if neg_side(row.side)=='N' else -(next_pt - opp_heels) # bounce in current frame to opposite player heels in current frame
                if row.img_name in ('img_8348.jpg'): row.b2sl = -1*row.b2sl
                prev_row = point_clip_df.iloc[max(idx-2,0):idx][point_clip_df.side==neg_side(row.side)]
                if prev_row.shape[0]>0:
                    prev_opp_heels = prev_row.n_heels.values[0] if neg_side(row.side)=='N' else prev_row.f_heels.values[0]
                    row['r2ph_theta'] = np.around(compute_hit_angle(row['ball_3d'], prev_opp_heels, row.side),1)
                    row['b2ph_x'], row['b2ph_y'], _ = (next_pt - prev_opp_heels)  if neg_side(row.side)=='N' else -(next_pt - prev_opp_heels) # bounce in current frame to opposite player heels in previous return frame 
                else:
                    row['b2ph_x'], row['b2ph_y'], row['r2ph_theta'] = [np.nan]*3 
            else:
                row['b2sl'], row['b2bl'], row['b2h_x'], row['b2h_y'], row['b2ph_x'], row['b2ph_y'], row['r2ph_theta']= [np.nan]*7
        else:
            if row.side in ('B'):
                row['ball_3d'] = next_pt
        all_rows.append(row)
    point_clip_df = pd.DataFrame(all_rows)
    point_clip_df = point_clip_df.loc[ (point_clip_df.side=='F') | (point_clip_df.side=='N')]
    point_clip_df['r2r_theta'] = point_clip_df['r2b_theta'] - point_clip_df['r2b_theta'].shift(1)
    point_clip_df['h2h_theta'] = point_clip_df['r2h_theta'] - point_clip_df['r2ph_theta']
    point_clip_df.loc[point_clip_df.shot_type=='S',['r2r_theta', 'h2h_theta','b2ph_x','b2ph_y']] = np.nan
    point_clip_df['hdist'] = np.around(np.sqrt(np.sum((point_clip_df[['hh2c','hh2n']].values - point_clip_df.shift(2)[['hh2c','hh2n']].values)**2, axis=1)))
    point_clip_df['return_type'] = point_clip_df['return_type'].shift(-1).fillna('O')
    assign_results(point_clip_df)
    point_clip_df[['hdist_avg','x_speed_avg','y_speed_avg']] = np.around(point_clip_df.apply(lambda row: point_clip_df.loc[((point_clip_df.index<=row.name) & (point_clip_df.side==row.side) & (point_clip_df.shot_type!='S') & (point_clip_df.result!='O')),\
                                                                                             ['hdist','x_speed','y_speed']].mean(), axis=1),1)
    sel_cols = ['clip_num','img_name','return_type','side','shot_type','player','x_speed','y_speed','r2r_theta','h2h_theta','b2ph_x','b2h_x','b2ph_y','b2h_y','hh2c','hh2n','oh2c','oh2n','b2sl','b2bl','hdist','hdist_avg','x_speed_avg','y_speed_avg','result']
    #display(point_clip_df[sel_cols])
    return point_clip_df[sel_cols].round(2)

def main(input_dict_file, input_df_file, points_override_file, points_drop_file, output_df_file, local_img_folder, court_kp_file, debug):
     with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_img_folder = Path(local_img_folder)
        df = pd.read_pickle(input_df_file)
        print(f'input dict file: {input_dict_file}')
        with open(input_dict_file, 'rb') as fp:
            hits_and_bounce_dict = pickle.load(fp)[1]
        points_df = gen_points_df(hits_and_bounce_dict, df, local_img_folder=local_img_folder) 
        if os.path.exists(points_override_file): apply_df_overrides(points_override_file, points_df)
        if os.path.exists(points_drop_file): drop_df_overrides(points_drop_file, points_df)
        all_points_df = pd.DataFrame([])
        for clip_num in np.array(sorted(points_df.clip_num.value_counts().index)):
            print(f'processing clip: {clip_num}')
            point_clip_df = gen_point_attribs(points_df, clip_num, df, local_img_folder, court_kp_file)
            all_points_df = pd.concat([all_points_df,point_clip_df], ignore_index=True)
        all_points_df.to_csv(output_df_file) 
        logging.debug(f'final shape: {all_points_df.shape}')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_dict_file", required=True, help="Path to hit and bounce dict file")
    parser.add_argument("--input_df_file", required=True, default=None, help="Path to dataframe with all attributes added after hits/bounce")
    parser.add_argument("--points_override_file", required=False, default=None, help="File to override points df")
    parser.add_argument("--points_drop_file", required=False, default=None, help="Points to be dropped from points_gf")
    parser.add_argument("--court_kp_file", required=False, default=None, help="Court key point file")
    parser.add_argument("--output_df_file", required=True, help="Path to output df after serve and end points")
    parser.add_argument("--local_img_folder", required=False, default=None, help="Local image folder")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    main(args.input_dict_file, args.input_df_file, args.points_override_file, args.points_drop_file, args.output_df_file, args.local_img_folder, args.court_kp_file, args.debug)

