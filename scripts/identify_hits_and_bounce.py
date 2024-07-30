from utils import apply_df_overrides, apply_dict_overrides, read_df_from_csv, save_df_to_csv, filter_clip, reset_logging, configure_logging, apply_func_all_clips, r2_dist, not_nan, fname2int, offset_fname, calc_delta_per_frame, calc_delta_per_frame_backward, argmin
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

HYPERS = {'d1':3, 'd2':2, 'a1':0.1, 'a2':0.1}

def add_turn_attribs(df):
    window_size = 29
    min_window = 1
    df['x_minf'] = df['x'].rolling(window_size, min_periods=min_window).min().shift(-(window_size-1))
    df['x_minp'] = df['x'].rolling(window_size, min_periods=min_window).min()
    df['x_maxf'] = df['x'].rolling(window_size, min_periods=min_window).max().shift(-(window_size-1))
    df['x_maxp'] = df['x'].rolling(window_size, min_periods=min_window).max()
    df['y_minf'] = df['y'].rolling(window_size, min_periods=min_window).min().shift(-(window_size-1))
    df['y_minp'] = df['y'].rolling(window_size, min_periods=min_window).min()
    df['y_maxf'] = df['y'].rolling(window_size, min_periods=min_window).max().shift(-(window_size-1))
    df['y_maxp'] = df['y'].rolling(window_size, min_periods=min_window).max()
    f_cond, n_cond = ((df.y <= df.y_minp) & (df.y <= df.y_minf)), \
                     ((df.y >= df.y_maxp) & (df.y >= df.y_maxf) & (df.y >= 100)) 
    x_cond = ((df.x <= df.x_minp) & (df.x <= df.x_minf)) | \
             ((df.x >= df.x_maxp) & (df.x >= df.x_maxf))
    bdist_cond = ((df.n_bdist <= 130)|(df.f_bdist <= 130))
    df['xturn'] = (x_cond) 
    df['yturn'] = (f_cond | n_cond )
    df['only_turn'] = (f_cond | n_cond | x_cond) 
    df['turn'] = (f_cond | n_cond | x_cond) & (bdist_cond)
    df['side'] = np.where(f_cond, 'F', np.where(n_cond, 'N', np.where(df.f_bdist<120, 'F', np.where(df.n_bdist, 'N', np.nan))))
    df['side'] = np.where(f_cond, 'F', np.where(n_cond, 'N', \
                         np.where(df.f_bdist<120, 'F', \
                         np.where(df.n_bdist<120, 'N', \
                         np.where(df.f_bdist<df.n_bdist, 'F', \
                         np.where(df.n_bdist<df.f_bdist, 'N',np.nan))))))
    return df

def add_vel_acc(df):
    df['ball_c'] = df[['x', 'y']].to_numpy().tolist()
    df.loc[:,'x_vel'] = calc_delta_per_frame(df, 'x')
    df.loc[:,'y_vel'] = calc_delta_per_frame(df, 'y')
    df.loc[:,'ball_vel'] = calc_delta_per_frame(df, 'ball_c')
    df.loc[:,'x_acc'] = calc_delta_per_frame(df, 'x_vel').shift(1)
    df.loc[:,'y_acc'] = calc_delta_per_frame(df, 'y_vel').shift(1)
    df.loc[:,'ball_acc'] = calc_delta_per_frame(df, 'ball_vel').shift(1)

def add_theta(df, t=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['x_p'+str(t)] = df['x'].rolling(t, min_periods=max(1,t-2)).mean().shift(1).round(1)
        df['y_p'+str(t)] = df['y'].rolling(t, min_periods=max(1,t-2)).mean().shift(1).round(1)
        df['x_f'+str(t)] = df['x'].rolling(t, min_periods=max(1,t-2)).mean().shift(-t).round(1)
        df['y_f'+str(t)] = df['y'].rolling(t, min_periods=max(1,t-2)).mean().shift(-t).round(1)
        df['vec_x'+str(t)] = df['x_f'+str(t)] - df['x']
        df['vec_y'+str(t)] = df['y_f'+str(t)] - df['y']
        df['vec_mod'+str(t)] = np.sqrt(df['vec_x'+str(t)]**2+df['vec_y'+str(t)]**2).round(1)
        df['vec_mod'+str(t)] = df['vec_mod'+str(t)].fillna(method='bfill')
        df['theta'+str(t)] = (np.arctan(np.abs(df['vec_y'+str(t)]/df['vec_x'+str(t)]))*180/math.pi).round()
        df['theta'+str(t)] = np.where(df['vec_mod'+str(t)]>55,np.nan,df['theta'+str(t)])    
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]>0) & (df['vec_y'+str(t)]>0)), df['theta'+str(t)], df['theta'+str(t)]) 
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]>0) & (df['vec_y'+str(t)]<0)),  360-df['theta'+str(t)], df['theta'+str(t)]) 
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]<0) & (df['vec_y'+str(t)]<0)), 180+df['theta'+str(t)], df['theta'+str(t)]) 
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]<0) & (df['vec_y'+str(t)]>0)), 180-df['theta'+str(t)], df['theta'+str(t)])
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]==0) & (df['vec_y'+str(t)]>0)), 90, df['theta'+str(t)]) 
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]==0) & (df['vec_y'+str(t)]<0)), 270, df['theta'+str(t)]) 
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]>0) & (df['vec_y'+str(t)]==0)), 360, df['theta'+str(t)]) 
        df['theta'+str(t)] = np.where(((df['vec_x'+str(t)]<0) & (df['vec_y'+str(t)]==0)), 180, df['theta'+str(t)]) 
        df['theta'+str(t)] = df['theta'+str(t)].fillna(method='bfill')
        df['theta'+str(t)+'_delta'] = abs(df['theta'+str(t)] - df['theta'+str(t)].shift(1))
        df['theta'+str(t)+'_delta'] = np.where(df['theta'+str(t)+'_delta'] > 180, 360-df['theta'+str(t)+'_delta'], df['theta'+str(t)+'_delta']) 
        cond1 = df['theta'+str(t)]>df['theta'+str(t)].shift(1)
        cond2 = (df['theta'+str(t)]>=0) & (df['theta'+str(t)]<=90) & (df['theta'+str(t)].shift(1)>=270) & (df['theta'+str(t)].shift(1)<=360)
        df['theta'+str(t)+'_delta']  = np.where(cond1|cond2 , df['theta'+str(t)+'_delta'], -df['theta'+str(t)+'_delta'])
        df['theta'+str(t)+'_delta_cumm'] = df['theta'+str(t)+'_delta'].rolling(3).sum().shift(-2).round(1)
    

def neg_side(side):
    return 'N' if side[-1]=='F' else 'F' 

def pick_best_hit(df,turn_only=False):
    logging.debug(f'df : {df.shape}')
    if len(np.unique(df.side))>1:
       logging.debug('inside pick best')
       logging.debug(df[['side']])
       raise ValueError(f"Multiple sides found in turn_df")
    df['abs_theta1_delta'] = df['theta1_delta'].abs()
    df['theta_dist'] = abs(df['abs_theta1_delta']) + abs(df['y_f25']) 
    sort_cols = ['theta_dist']
    sel_row = df.sort_values(by=sort_cols, ascending=False).iloc[0]
    sel_row = df.iloc[0] if turn_only else sel_row
    cond_moving = abs(sel_row.y_f25)>50
    cond_moving = abs(sel_row.y_f25)>27
    cond_jump = (sel_row.y_m25>30) | (sel_row.y_mf10>167)
    cond_net = abs(sel_row.y*2-np.average(sel_row.kp.squeeze()[12:14], axis=0, weights=np.array([0.6,0.4]))[1]) < 60
    cond_net = abs(sel_row.y*2-np.average(sel_row.kp.squeeze()[12:14], axis=0, weights=np.array([0.6,0.4]))[1]) < 100
    shot_type = 'R' if (cond_moving and not cond_jump) else ('T' if cond_net else 'E')
    shot_type = ('T' if cond_net else 'E') if turn_only else shot_type
    logging.debug(f'cond_moving: {cond_moving, abs(sel_row.y_f25)}  cond_jump:{cond_jump} cond_net: {cond_net}')
    return sel_row.name, sel_row.side, shot_type


def find_pt_hits(serve_hit_pts, end_pts_mod, results_df, clip_df):
    clip_pts = []
    clip_num = clip_df.iloc[0].clip_num
    c_serve_pts = serve_hit_pts.get(clip_num)
    add_vel_acc(clip_df)
    add_theta(clip_df)
    cond7 = (results_df.x.rolling(60, min_periods=0).apply(lambda x: x.isna().sum()).shift(-59)>51) & (results_df.x.isna())
    zero_out = results_df.loc[cond7 & (results_df.clip_num==clip_num)].index.values
    zero_out = [z for z in zero_out if fname2int(z)>fname2int(c_serve_pts[0])] if c_serve_pts else zero_out
    logging.debug(f'zero out: {zero_out}')
    if len(zero_out)>2: 
        clip_df.loc[zero_out[2]:None if ((not c_serve_pts) or (len(c_serve_pts)==1)) else c_serve_pts[1],['x','y']] = [np.nan, np.nan]
    clip_df['y_d1'] = calc_delta_per_frame(clip_df, 'y')
    clip_df['y_d1'] = clip_df['y_d1'].fillna(method='ffill', limit=7)
    clip_df['y_m25'] = clip_df['y_d1'].abs().rolling(25, min_periods=10).max().shift(-25)
    clip_df['y_f10'] = clip_df['y_d1'].rolling(10, min_periods=5).sum().shift(-10)
    clip_df['y_mf10'] = clip_df['y_f10'].abs().rolling(30, min_periods=10).max().shift(-25)
    clip_df['y_f25'] = clip_df['y_d1'].rolling(25, min_periods=10).sum().shift(-25)
    c_serve_pts = [clip_df.index.values[0]] if not c_serve_pts else c_serve_pts
    temp_df = pd.DataFrame(columns=clip_df.columns) 
    first_s = clip_df[clip_df.turn==True].iloc[0]['side']
    logging.debug(f'start frame: {clip_df[clip_df.turn==True].iloc[0].name} end frame: {clip_df[clip_df.turn==True].iloc[-1].name}')
    for s1,s2 in zip_longest(c_serve_pts, c_serve_pts[1:]):
        logging.debug(f'entering for: {s1} {s2}')
        pt_hits, best_t, best_f = [], None, None
        end_pts = sorted(end_pts_mod.get(clip_num)) if end_pts_mod.get(clip_num) else None
        pt_end_fs = [ep for ep in end_pts if (fname2int(ep)>fname2int(s1)) and (fname2int(ep)<(1e6 if s2 is None else fname2int(s2)))] if end_pts else None
        pt_end_f = pt_end_fs[0] if (pt_end_fs and len(pt_end_fs)>0) else None
        logging.debug(f'end pts: {pt_end_f} {s2} {(pt_end_f if pt_end_f else (s2 if s2 else True))}')
        #display(clip_df[sel_cols])
        turn_df = clip_df.loc[(clip_df.index>s1) & (clip_df.index<(pt_end_f if pt_end_f else (s2 if s2 else clip_df.index.values[-1]))) & (clip_df.turn==True)]
        logging.debug(f'turn_df: {turn_df.shape}')
        if serve_hit_pts.get(clip_num):
            s_t = [s1,'F','S'] if (clip_df['hraised_side'].fillna(method='ffill').loc[s1][0]=='F') else \
                   [s1,'N','S'] if (clip_df['hraised_side'].fillna(method='ffill').loc[s1][0]=='N') else (s1,'S')
            pt_hits.append(s_t)
            best_f, best_s, best_t = s1, s_t[1], s_t[1][-1]
            logging.debug(f'serve appended to pt hits: {pt_hits}')
        while len(turn_df)>0:
            logging.debug(f'while loop turn df: {turn_df.shape}')
            top_row = turn_df.iloc[0]
            find_s = neg_side(pt_hits[-1][1]) if len(pt_hits)>0 else first_s
            logging.debug(f'turn_df shape: {turn_df.shape} find_s: {find_s} {top_row.side} {top_row.name}')
            if fname2int(top_row.name)<fname2int(s1):
                turn_df = turn_df.iloc[1:] 
                print(f'top row: {top_row.name} is less than {s1}')
            elif top_row.side==find_s:
                logging.debug('entering elif')
                pop_row = turn_df.iloc[0]
                turn_df = turn_df.iloc[1:] 
                temp_df = temp_df.append(pop_row, ignore_index=False)
                logging.debug(f'Inserting to temp: {find_s} {top_row.name} {pop_row.name} {pop_row.side} turn_df shape: {turn_df.shape}')
            else:
                if len(temp_df)==0: 
                    turn_df = turn_df.iloc[1:] 
                    continue
                best_f, best_s, best_t = pick_best_hit(temp_df)
                logging.debug(f'found best 1: {best_f} {best_s} {best_t}')
                pt_hits.append([best_f, best_s, best_t])
                temp_df = pd.DataFrame(columns=clip_df.columns)
                if best_t in ('T','E'):
                    logging.debug(f'clip: {clip_num} final_pt1 {pt_hits}')
                    break
        if len(temp_df)>0:
            best_f, best_s, best_t = pick_best_hit(temp_df)
            logging.debug(f'found best 2: {best_f} {best_s} {best_t}')
            temp_df = pd.DataFrame(columns=clip_df.columns)
            pt_hits.append([best_f, best_s, best_t])
        if best_t not in ('T','E'):
            pt_end_f = results_df.iloc[results_df.index.get_loc(pt_end_f)+41].name if pt_end_f else None
            logging.debug(f'pt_end_f: {pt_end_f} {best_f} {best_s}')
            only_turn_df = clip_df.loc[(clip_df.index>s1) & (clip_df.index<=(pt_end_f if pt_end_f else (s2 if s2 else clip_df.index.values[-1]))) & (clip_df.only_turn==True)]
            logging.debug(f'only turn df: {only_turn_df.index.values}')
            temp_df = only_turn_df[(only_turn_df.index>best_f)&(only_turn_df.side==neg_side(best_s))]
            if temp_df.shape[0]==0: 
                logging.debug(f'Skipping final point. final_pt2 {pt_hits}')
            else:
                logging.debug(f"best_f: {best_f} best_s: {best_s} pt hits: {pt_hits} final temp_df: {temp_df['turn']}")
                fin_f, fin_s, fin_t =  pick_best_hit(temp_df, turn_only=True)
                temp_df = pd.DataFrame(columns=clip_df.columns)
                logging.debug(f'final pick: {fin_f} {fin_s} {fin_t}')
                pt_hits.append([fin_f, fin_s, fin_t])
                logging.debug(f'clip: {clip_num} final_pt2 {pt_hits}')
        clip_pts.append(pt_hits)
    return clip_pts

def find_last_true(bool_list):
    try:
        return len(bool_list)-1-list(reversed(bool_list)).index(True)
    except ValueError:
        return 0

def insert_correction(pts_lists, ins_tup):
    pt_idx = find_last_true([ins_tup[0]>pts_list[0][0] for pts_list in pts_lists])
    pts_list = pts_lists[pt_idx]
    for idx, (pt1_tup, pt2_tup) in enumerate(zip(pts_list, pts_list[1:])):
        if ins_tup[0]==pt1_tup[0]: pts_list[idx] = ins_tup; return
        elif ins_tup[0]>pt1_tup[0] and ins_tup[0]<pt2_tup[0]: pts_list.insert(idx+1, ins_tup); return 
        elif idx==0 and ins_tup[0]<pt1_tup[0]: pts_list.insert(idx, ins_tup); return 
        elif idx==len(pts_list)-2 and ins_tup[0]==pt2_tup[0]: pts_list[idx+1] = ins_tup; return
        elif idx==len(pts_list)-2 and ins_tup[0]>pt2_tup[0]: pts_list.insert(idx+2, ins_tup); return 

def remove_correction(pts_lists, rem_tup):
    for pts_list in pts_lists:
        for idx, pt_tup in enumerate(pts_list):
            if rem_tup[0]==pt_tup[0]: pts_list.pop(idx); return

def apply_corrections(final_pts, corrections_dict):
    for clip_num, corrections in corrections_dict.items():
        logging.debug(f'processing clip: {clip_num}')
        insert_pts, remove_pts = corrections.get('insert'), corrections.get('remove')
        for ins_tup in insert_pts: insert_correction(final_pts.get(clip_num), ins_tup)
        for rem_tup in remove_pts: remove_correction(final_pts.get(clip_num), rem_tup)

def l2_error(y, y_pred):
    return np.sum((y-y_pred)**2)


def fit_model(x, y, degree=3, alpha=0.1):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)  
    x_p = poly_features.fit_transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        model = Ridge(alpha=alpha if alpha else 0)
        model.fit(x_p, y)
        y_pred = model.predict(x_p)
        error = l2_error(y, y_pred)
        return model, y_pred, error
    
def find_curves_w_error(x, y, cut_idx, d1=4, d2=3, a1=0.1, a2=0.1):
    x, y = x.reshape(-1,1), y.reshape(-1,1)
    x1, x2 = x[:cut_idx], x[cut_idx:]
    y1, y2 = y[:cut_idx], y[cut_idx:]
    if x1.shape[0]==0 or x2.shape[0]==0: return None, None, None, None, None
    m1, y1_pred, e1 = fit_model(x1, y1, degree=d1, alpha=a1)
    m2, y2_pred, e2 = fit_model(x2, y2, degree=d2, alpha=a2)
    return m1, m2, y1_pred, y2_pred, e1+e2
    
def find_farthest_point(line_point1, line_point2, points, side_point):
    x1, y1 = line_point1
    x2, y2 = line_point2
    if np.isnan([x1, y1, x2, y2]).any(): raise AttributeError(f'Error in find farthest point {x1} {y1} {x2} {y2}')
    cp = None
    if x1 == x2:
        slope = None
        y_intercept = None
    else:
        slope = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - slope * x1
    def is_same_side(point, ref_point):
        def cross_product(a,b): return a[0]*b[1]-a[1]*b[0]
        x, y = point
        x_ref, y_ref = ref_point
        if slope is None:
             return True if (x-x1)*(x_ref-x1)>=0 else False, None, None
        else:
            line_vec = (x2 - x1, y2 - y1)
            point_vec = (x - x1, y - y1)
            ref_point_vec = (x_ref-x1, y_ref-y1)
            cp_pt = cross_product(line_vec, point_vec)
            cp_ref_pt = cross_product(line_vec, ref_point_vec)
            return True if (cp_pt * cp_ref_pt >= 0) else False, cp_pt, cp_ref_pt
            
    def distance_from_line(point):
        x, y = point
        if slope is None:
            return abs(x - x1)
        else:
            return abs((y - y_intercept) - slope * x) / math.sqrt(slope**2 + 1)

    max_distance = -1
    farthest_point = None
    for idx, point in enumerate(points):
        distance = distance_from_line(point)
        if distance > max_distance and is_same_side(point, side_point)[0]:
            logging.debug(f'idx: {idx} same_side:{is_same_side(point, side_point)} point: {point} sp: {side_point} x1,y1:{x1,y1} dist: {distance:0.0f}')
            max_distance = distance
            farthest_point = (idx,point)
    return farthest_point if farthest_point is not None else (idx,point)


def find_intersection(m1, m2, x_min, x_max, d=3):
    def equations(x):
        poly_features = PolynomialFeatures(degree=d, include_bias=True)  
        x_p = poly_features.fit_transform(x.reshape(-1,1))
        y1 = m1.predict(x_p)
        y2 = m2.predict(x_p)
        return y1[0] - y2[0]

    initial_guesses = np.linspace(x_min, x_max, 21) 
    roots = []
    for x0 in initial_guesses:
        solution_x = fsolve(equations, x0)
        if abs(equations(solution_x)) < 1e-6:  # Check for valid roots
            roots.append(solution_x.round(1))
    return np.unique(np.array(roots))

def closer_neighbour(x, y, cut_idx):
    logging.debug(f'closer neighbour - cut_idx: {cut_idx}, x: {x}')
    if cut_idx==0 or cut_idx==len(x)-1: return 0 
    d1 = abs(x[cut_idx]-x[cut_idx-1])
    d2 = abs(x[cut_idx+1]-x[cut_idx])
    return 0 if d1<=d2 else 1

def fit_curves(df, img_name1, img_name2, hypers, f_offset=None, debug=False, plot=True, local_offset=6):
    if img_name2 is None and f_offset is None: print('specify offset'); return
    if img_name2 is None: img_name1, img_name2 =  foffset_fname(df, img_name1, offset=-offset),  \
                                                  foffset_fname(df, img_name1, offset=offset)
    xy_df = df.loc[img_name1:img_name2,['x','y']]*2
    x_data, y_data, fs = xy_df.values[:,0], -xy_df.values[:,1], xy_df.index.values
    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x_data, y_data, fs = x_data[mask], y_data[mask], fs[mask]
    ms_ys_es = [find_curves_w_error(x_data, y_data, cand_idx, **hypers) for cand_idx in range(1,len(x_data)-2)]
    opt_idx1, (e, m1, m2, y1_pred, y2_pred) = argmin(ms_ys_es, key=lambda x: (x[4],*x[:4]))
    opt_idx1 += 1
    pt1 = (x_data[opt_idx1], -y_data[opt_idx1])
    if debug:
        for idx,(m1,m2,_,_,e) in enumerate(ms_ys_es):
            print(f'idx: {idx+1} f: {fs[idx+1]} error: {e}')
        logging.debug(f'm1 coeff: {m1.coef_}, m2 coeff: {m2.coef_}')    
    opt_idx2, pt2 = find_farthest_point((x_data[0],y_data[0]),(x_data[-1],y_data[-1]),list(zip(x_data,y_data)), (pt1[0],-pt1[1]))
    clip_num = df.loc[img_name1].clip_num
    if plot:
        x1, x2, y1, y2 = x_data[:opt_idx1], x_data[opt_idx1:], y_data[:opt_idx1], y_data[opt_idx1:]
        if debug: print(f'{x1.shape}, {y1.shape}, {x2.shape}, {y2.shape} {y1_pred.shape} {y2_pred.shape}')
        plt.figure(figsize=(7,4))
        plt.plot(x1, y1, 'o', label='Data1')
        plt.plot(x1, y1_pred, label='Curve1')
        plt.plot(x2, y2, 'o', label='Data2')
        plt.plot(x2, y2_pred, label='Curve2')
        plt.scatter(x2[0], y2[0], color='r', s=100)
        plt.scatter(pt2[0], pt2[1], color='y', s=200)
        plt.plot((x_data[0],x_data[-1]),(y_data[0],y_data[-1]), color='y')
        plt.legend()
        plt.title(f'clip{clip_num}_{opt_idx1}_{fs[opt_idx1]}_{x_data[opt_idx1],y_data[opt_idx1]}')
    return pt1, fs[opt_idx1], (pt2[0],-pt2[1]), fs[opt_idx2]

def add_bounce_to_point_hits(final_pts, clip_df, hypers=HYPERS):
    clip_num = clip_df.iloc[0].clip_num
    pt_tups = final_pts.get(clip_num)
    pt_results = [[] for _ in range(len(pt_tups))]
    for pt_idx, pt_tup in enumerate(pt_tups):
        if len(pt_tup)==1: pt_results[pt_idx].append(pt_tup[0]); continue
        for s1_t, s2_t in zip(pt_tup, pt_tup[1:]):
            (s1_f, s1_s, s1_ty), (s2_f, s2_s, s2_ty) = s1_t, s2_t
            start_offset = 4 if s1_ty=='S' else 9 
            s1_f_mod, s2_f_mod = clip_df.iloc[clip_df.index.get_loc(s1_f)+start_offset].name, s2_f
            if s1_s=='N': s2_f_mod = clip_df.iloc[clip_df.index.get_loc(s2_f)-3].name
            if (fname2int(s2_f)-fname2int(s1_f))>65 and s2_ty in ('E','T'): 
                s2_f_mod = clip_df.iloc[clip_df.index.get_loc(s1_f)+40].name
            add_vel_acc(clip_df)
            if s2_ty=='T': 
                pt_results[pt_idx].append(s1_t)
            else:
                pt1, b_f1, pt2, b_f2 = fit_curves(clip_df, s1_f_mod, s2_f_mod, hypers, debug=False, plot=False, local_offset=6)
                pt_results[pt_idx].append(s1_t)
                pt_results[pt_idx].append([b_f2,'B','B'])
        pt_results[pt_idx].append(s2_t)
    return pt_results

def main(input_file, serve_endofpoint_dict_file, turn_override_file, hit_override_file, output_dict_file, output_df_file, debug):
     with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_pickle(input_file)
        df = add_turn_attribs(df)
        apply_df_overrides(turn_override_file, df)
        with open(serve_endofpoint_dict_file, 'rb') as fp:
            serve_dict, end_dict = pickle.load(fp)
        hit_pts = apply_func_all_clips(df, partial(find_pt_hits, serve_dict, end_dict, df))
        corrections_dict = {}
        apply_dict_overrides(hit_override_file, corrections_dict)
        apply_corrections(hit_pts, corrections_dict)
        hit_bounce_pts = apply_func_all_clips(df, partial(add_bounce_to_point_hits, hit_pts))
        df.to_pickle(output_df_file)
        with open(output_file, 'wb') as file:
            pickle.dump([dict(hit_pts), dict(hit_bounce_pts)], file)
        reset_logging()
        configure_logging(debug)
        logging.debug(f'hit_pts: {hit_bounce_pts}')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_file", required=True, help="Path to cleaned merged data file")
    parser.add_argument("--serve_endofpoint_dict_file", required=False, default=None, help="serve and end of point dicts file")
    parser.add_argument("--turn_override_file", required=False, default=None, help="turn point df override file")
    parser.add_argument("--hit_override_file", required=False, default=None, help="hit point dict override file")
    #parser.add_argument("--local_img_folder", required=True, help="Path to local image folder")
    parser.add_argument("--output_dict_file", required=True, help="Path to output dicts for serve and end points")
    parser.add_argument("--output_df_file", required=True, help="Path to output df after serve and end points")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging")
    args = parser.parse_args()
    main(args.input_file, args.serve_endofpoint_dict_file, args.turn_override_file, args.hit_override_file, args.output_dict_file, args.output_df_file, args.debug)

