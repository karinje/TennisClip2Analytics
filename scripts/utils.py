from collections import defaultdict
import pandas as pd 
import numpy as np 
import numbers
import logging
import warnings
from pathlib import Path

def dict_representer(dumper, data):
    return dumper.represent_dict(dict(data))

def dict_constructor(loader, node):
    return defaultdict(int, loader.construct_pairs(node))

def configure_logging(debug):
    logging_level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s'
    )

def reset_logging():
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


def not_nan(value):
    if isinstance(value, (list,np.ndarray)): return not np.isnan(value).any()
    elif isinstance(value, numbers.Number): return not np.isnan(value)
    else: return True

def check_nan(value):
    return not not_nan(value)

def save_df_to_csv(df, filename):
    df_copy = df.copy()
    ndarr_cols = []
    for col in df_copy.columns:
        logging.debug(f'processing col: {col}')
        if isinstance(df_copy[col].iloc[0], np.ndarray):
            ndarr_cols.append(col)
            df_copy[col] = df_copy[col].map(lambda x: x.tolist() if not pd.isna(x).all() else x)
    df_copy.to_csv(filename, index=True)
    del df_copy
    return ndarr_cols

def read_df_from_csv(filename, index_col=0, ndarr_cols=None):
    df = pd.read_csv(filename, index_col=index_col)
    for col in ndarr_cols:
            df[col] = df[col].map(lambda x: np.array(eval(x)) if not_nan(x) else x)
    return df

def apply_df_overrides(override_file, df):
    if not override_file: return 
    with open(override_file, 'r') as file:
        for line in file:
            line = line.strip()
            line = line.split('=')
            logging.debug(f'{line}')
            df.loc[eval(line[0])] = eval(line[1])

def apply_dict_overrides(override_file, input_dict):
    if not override_file: return
    with open(override_file, 'r') as file:
            for line in file:
                line = line.strip()
                line = line.split('=')
                input_dict[eval(line[0])] = eval(line[1])

def drop_df_overrides(override_file, df):
    with open(override_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('clip'): drop_idx = df[df.clip_num==int(line.split('_')[1])].index
            else: drop_idx = df[df.img_name==str(line)].index
            df.drop(drop_idx, inplace=True)

def replace_df_index_prefix(df, new_prefix):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        cur_prefix = Path(df.iloc[0].name).parent.as_posix()
        print(f'Replacing index {cur_prefix}:{new_prefix}')
        df.index = df.index.str.replace(cur_prefix, new_prefix)
    
def filter_clip(df, clip_num, clip_end_offset=None):
        clip_offset = 0 if not clip_end_offset else clip_end_offset
        return df.loc[(df.clip_num==clip_num)|(df.clip_num.shift(clip_offset)==clip_num)]

def get_clip_nums(df):
    return df.clip_num.value_counts().sort_index().index.values

def offset_fname(f, offset, prefix='img_', suffix='.jpg'):
  img_num = int(f.replace(prefix,'').replace(suffix,''))+offset
  img_f = prefix+str(img_num).zfill(4)+suffix
  return img_f

def foffset_fname(df, f, offset=15):
    return df.iloc[df.index.get_loc(f)+offset].name

def fname2int(f, offset=0, prefix='img_', suffix='.jpg'):
    return int(f.replace(prefix,'').replace(suffix,''))+offset


def argmin(iterable, key=None):
    if key is None:
        key = lambda x: x
    min_val, min_idx = min((key(x), i) for i, x in enumerate(iterable))
    return min_idx, min_val

def argmax(iterable, key=None):
    if key is None:
        key = lambda x: x
    max_val, max_idx = max((key(x), i) for i, x in enumerate(iterable))
    return max_idx, max_val

def r2_dist(a,b,axis=1,nan_shape=(2,2)):
    if not isinstance(a, np.ndarray): a = np.array(a)
    if not isinstance(b, np.ndarray): b = np.array(b)    
    
    if np.isnan(a).any() or np.isnan(b).any(): 
        return np.full(nan_shape, np.nan)
    return np.sqrt(np.sum((a-b)**2, axis=axis))

def apply_func_all_clips(df, func, clip_end_offset=None):
    clip2dict = defaultdict(list)
    for clip_num in get_clip_nums(df): 
        clip_df = filter_clip(df, clip_num, clip_end_offset=clip_end_offset)
        logging.info(f'processing clip: {clip_num} shape: {clip_df.shape}')
        clip2dict[clip_num] = func(clip_df)
    return clip2dict

def calc_delta_per_frame_backward(df, column_name):
        prev_value = np.nan
        result = pd.Series(index=df.index, dtype=float)
        for f, row in df.iterrows():
            current_value = row[column_name]
            i = df.index.get_loc(f)
            for j in reversed(range(max(i-11,0), max(0,i))):
                prev_value = df.iloc[j][column_name]
                if not_nan(prev_value):
                    break
            if not_nan(prev_value):
                n_frames = fname2int(f) - fname2int(df.iloc[j].name)
                if isinstance(current_value, (int, float)) and isinstance(prev_value, (int, float)):
                      result.iloc[i] = (prev_value-current_value)/max((j - i),n_frames)
                else:
                    result.iloc[i] = r2_dist(current_value,prev_value, axis=0, nan_shape=(1,1))/max((j - i),n_frames)
            else:
                result.iloc[i] = np.nan
        return result 


def calc_delta_per_frame(df, column_name):
    result = pd.Series(index=df.index, dtype=float)
    for f, row in df.iterrows():
        current_value = row[column_name]
        i = df.index.get_loc(f)
        if i > len(df)-10:
            result.iloc[i] = np.nan
            continue 
        for j in range(i+1, i+11):
            if j<len(df): next_value = df.iloc[j][column_name]
            if not_nan(next_value):
                break
        if not_nan(next_value):
            n_frames = fname2int(df.iloc[j].name) - fname2int(f)
            if isinstance(current_value, (int, float)) and isinstance(next_value, (int, float)):
                 #if f=='img_25012.jpg': print(f'f {f} {df.iloc[j].name} j-i {(j - i)} n_frames {n_frames} {(next_value-current_value)/max((j - i),n_frames)}')
                 result.iloc[i] = (next_value-current_value)/max((j - i),n_frames)
            else:
                result.iloc[i] = r2_dist(current_value,next_value, axis=0, nan_shape=(1,1))/max((j - i),n_frames)
    return result 
