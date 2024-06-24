from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
app_dir = pathlib.Path(__file__).parent
submodule_dir = app_dir / 'ViTPose'
sys.path.insert(0, submodule_dir.as_posix())
import time 
import json
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)


DET_MODEL_DICT = {
        'YOLOX-tiny': {
            'config':
            'mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        },
        'YOLOX-s': {
            'config':
            'mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        },
        'YOLOX-l': {
            'config':
            'mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        },
        'YOLOX-x': {
            'config':
            'mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        },
    }

POSE_MODEL_DICT = {
        'ViTPose-L': {
            'config':
            'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': 'models/ViTPose-L.pth',
        },
        'ViTPose-H': {
            'config':
            'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py',
            'model': 'models/ViTPose-H.pth',
        },
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Run the AppPoseModel')

    parser.add_argument('--det_model_name', type=str, required=True,
                        help='Name of the model to use')
    parser.add_argument('--pose_model_name', type=str, required=True,
                        help='Name of the model to use')
    parser.add_argument('--clip_csv', type=str, required=True,
                        help='Path to the input images in file')
    parser.add_argument('--start_row', type=int, default=0,
                        help='specify start row of dataframe')
    parser.add_argument('--end_row', type=int, default=-1,
                        help='specify start row of dataframe')
    parser.add_argument('--step', type=int, default=1,
                        help='specify step size of dataframe')
    parser.add_argument('--box_score_threshold', type=float, default=0.2,
                        help='Threshold for box scores (default: 0.2)')
    parser.add_argument('--kpt_score_threshold', type=float, default=0.2,
                        help='Threshold for keypoint scores (default: 0.2)')
    parser.add_argument('--vis_dot_radius', type=int, default=4,
                        help='Radius of the visualization dots (default: 4)')
    parser.add_argument('--vis_line_thickness', type=int, default=2,
                        help='Thickness of the visualization lines (default: 2)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output file for the visualization image')
    parser.add_argument('--show_viz', action='store_true', default=False,
                        help='optional to save visualization')
    args = parser.parse_args()

    return args.det_model_name, args.pose_model_name, args.clip_csv, args.start_row, args.end_row, args.step, \
           args.box_score_threshold, args.kpt_score_threshold, args.vis_dot_radius, \
           args.vis_line_thickness, args.output_dir, args.show_viz

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def numpy_decoder(obj):
    if isinstance(obj, list):
        try:
            return np.array(obj)
        except ValueError:
            pass
    return obj

def gen_mask():
    mask = np.ones((720,1280),dtype='uint8')
    mask[100:360,180:330] = 0
    mask[:112,:500] = 0
    mask[:120,500:] = 0
    for idx in range(400):
        mask[idx,range(400-idx)] = 0
    for idx in range(260):
        mask[range(200+idx), 1279-260+idx] = 0
    return mask

def filter_boxes(boxes, mask):
    first_boxes = boxes[0]
    mod_first_boxes = np.array([box.tolist() for box in first_boxes if mask[min(int(box[3]),719), min(int(box[2]),1279)]==1])
    boxes[0] = mod_first_boxes
    #for box in first_boxes:
    #    print(f'box: {int(box[3])-1, int(box[2])-1}')
    #    if mask[int(box[4]),int(box[3])]==0:
    #        print(f'selected: {box}')
    #print(f'inside func call {first_boxes.shape}')
    #print(f'{mod_first_boxes.shape}')
    #print(f'{len(boxes)}')
    return boxes 


def main():
    device='cuda:0'
    results_json = {}
    det_name, pose_name, clip_csv, start_row, end_row, step, box_score_threshold, kpt_score_threshold, vis_dot_radius, vis_line_thickness, output_dir, show_viz = parse_args()
    det_dict, pose_dict = DET_MODEL_DICT[det_name], POSE_MODEL_DICT[pose_name]
    det_model = init_detector(det_dict['config'], det_dict['model'], device=device)
    pose_model = init_pose_model(pose_dict['config'], pose_dict['model'], device=device)
    poseout_json = str(Path(output_dir)/(Path(clip_csv).stem+'_res.json'))
    print(f'output json: {poseout_json}')
    mask = gen_mask()
    clip_df = pd.read_csv(clip_csv,index_col=0)[start_row:end_row:step]
    clip_df['kp'] = clip_df['kp'].apply(lambda x: np.array(eval(x[1:-1])))
    clip_df['hm'] = clip_df['hm'].apply(lambda x: np.array(eval(x[1:-1])))
    print(f'df shape: {clip_df.shape} row range: {start_row}:{end_row}')
    start_time = time.time()
    for img_path,row in clip_df.iterrows():
        hm = row.hm
        mask_mod = cv2.warpPerspective(mask, hm, (mask.shape[1], mask.shape[0]))
        poseout_file = str(Path(output_dir)/(Path(img_path).stem+'_vis.jpg'))
        detout_file = str(Path(output_dir)/(Path(img_path).stem+'_det.jpg'))
        image = cv2.imread(f'{img_path.rstrip()}')
        image = image[:,:,::-1]
        print(f'img_path: {img_path}' )
        det_out = inference_detector(det_model, image)
        #print(f'shape before filter: {det_out[0].shape}')
        det_out = filter_boxes(det_out, mask_mod)
        #print(f'shape after filter: {det_out[0].shape}')
        person_det = [det_out[0]] + [np.array([]).reshape(0, 5)] * 79
        if show_viz: 
            vis = det_model.show_result(image, person_det, score_thr=box_score_threshold, bbox_color=None, text_color=(200, 200, 200), mask_color=None)
            cv2.imwrite(detout_file, vis[:, :, ::-1])
        person_results = process_mmdet_results(det_out, 1)
        pose_out, _ = inference_top_down_pose_model(pose_model, image, person_results=person_results, bbox_thr=box_score_threshold, format='xyxy')
        results_json[img_path.rstrip()] = pose_out
        if show_viz:
            vis = vis_pose_result(pose_model, image, pose_out, kpt_score_thr=kpt_score_threshold, radius=vis_dot_radius, thickness=vis_line_thickness)
            cv2.imwrite(poseout_file, vis[:, :, ::-1])

    end_time = time.time()
    with open(poseout_json, 'w') as f: f.write(json.dumps(results_json, cls=NumpyEncoder))
    print(f'total time: {end_time-start_time: 0.1f}')

if __name__ == "__main__":
    main()
