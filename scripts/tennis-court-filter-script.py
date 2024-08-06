import argparse
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from natsort import natsorted

def get_sorted_files(path, prefix="img_"):
    files = []
    for file_name in os.listdir(path):
        if file_name.startswith(prefix) and file_name[-4:] in [".jpg", ".png"]:
            files.append(os.path.join(path, file_name))
    return natsorted(files)

def add_size_to_kp(kp, size=10):
    return [cv2.KeyPoint(x, y, size) for x, y in kp.squeeze()]

def r2_dist(a, b):
    return np.sqrt(np.sum((a-b)**2))

def resize_image(img,scale=2):
    height, width = img.shape[:2]
    new_height, new_width = int(height*scale), int(width*scale)
    #print(f'dims: {height} {width} {new_height} {new_width}')
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def valid_kp(kp, img_size=(720, 1280)):
    cond1 = np.all(kp[:, 0] < img_size[1]) and np.all(kp[:, 1] < img_size[0])
    cond2 = sum([r2_dist(kp[i,:], kp[j,:]) for i,j in ((0,1), (0,2), (1,3), (2,3))])
    cond3 = min([r2_dist(kp[i,:], kp[j,:]) for i,j in ((0,1), (0,2), (1,3), (2,3))])
    cond4 = r2_dist(kp[12:14,:].mean(axis=0), np.array([[640,360]]))
    return (cond1) & (cond2>1600) & (cond3>300) & (cond4<110)

def court_kp_and_validity(args):
    f1, f2, img1_kp, plot_imgs = args
    img1, img2 = cv2.imread(str(f1)), cv2.imread(str(f2))
    if img2.shape[0] != img1.shape[0]: img2 = resize_image(img2,img1.shape[0]/img2.shape[0])
    img1[600:700, 200:400] = np.zeros((100, 200, 3), dtype=np.uint8)
    sift = cv2.SIFT_create()
    kp1, d1 = sift.detectAndCompute(img1, None)
    kp2, d2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(d1, d2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    img1_kp = np.loadtxt(img1_kp, delimiter=',')
    transformation_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    try:
        img2_kp = cv2.perspectiveTransform(img1_kp.reshape(-1,1,2), transformation_matrix)
    except:
        return f2, [[[0,0,0]]], False, [[0,0]]
    if plot_imgs:
        img1_w_kp = cv2.drawKeypoints(img1, add_size_to_kp(img1_kp), outImage=np.array([]), color=(0, 0, 255), flags=0)
        img2_w_kp = cv2.drawKeypoints(img2, add_size_to_kp(img2_kp), outImage=np.array([]), color=(0, 0, 255), flags=0)
        _, axes = plt.subplots(2,1, figsize=(20,20))
        axes[0].imshow(img1_w_kp)
        axes[1].imshow(img2_w_kp)
    return f2, img2_kp.tolist(), valid_kp(img2_kp.squeeze()), transformation_matrix.tolist()

def process_images(ref_img_f, sorted_files, court_kp_file, num_cores, part_size, output_dir):
    import multiprocessing
    n_parts = len(sorted_files) // part_size + 1
    for part in range(n_parts):
        start_time = time.time()
        pool = multiprocessing.Pool(processes=num_cores)
        results = pool.map(court_kp_and_validity, [(ref_img_f, f2, court_kp_file, False) for f2 in sorted_files[part*part_size:(part+1)*part_size]])
        pool.close()
        pool.join()
        pool_end_time = time.time()
        print(f'Part: {part} Pool Time: {pool_end_time - start_time:.6f}')
        results_df = pd.DataFrame(results, columns=['f','kp','valid','hm'])
        results_df.to_csv(os.path.join(output_dir, f'part{part}_{n_parts}_processed.csv'))
        write_end_time = time.time()
        print(f'Part: {part} Total Time: {write_end_time - start_time:.6f}')

def main():
    parser = argparse.ArgumentParser(description='Process tennis court images.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--ref_img', type=str, required=True, help='Reference image file')
    parser.add_argument('--court_kp_file', type=str, required=True, help='Court keypoints file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed files')
    parser.add_argument('--num_cores', type=int, default=os.cpu_count(), help='Number of CPU cores to use')
    parser.add_argument('--part_size', type=int, default=5000, help='Number of images to process in each part')
    args = parser.parse_args()

    path = Path(args.input_dir)
    sorted_files = get_sorted_files(path)
    ref_img_f = Path(args.ref_img)

    os.makedirs(args.output_dir, exist_ok=True)

    process_images(ref_img_f, sorted_files, args.court_kp_file, args.num_cores, args.part_size, args.output_dir)

    # Post-processing
    all_results_df = pd.DataFrame()
    n_parts = len(sorted_files) // args.part_size + 1
    for part in range(n_parts):
        processed_f = os.path.join(args.output_dir, f'part{part}_{n_parts}_processed.csv')
        results_df = pd.read_csv(processed_f, index_col=1)
        results_df['kp'] = results_df['kp'].apply(lambda x: np.array(eval(x[1:-1])))
        results_df['hm'] = results_df['hm'].apply(lambda x: np.array(eval(x[1:-1])))
        all_results_df = pd.concat([all_results_df, results_df], ignore_index=False)
        print(f'Processed part{part} with range {Path(results_df.index.values[0]).name}:{Path(results_df.index.values[-1]).name} and results shape: {all_results_df.shape}')

    all_results_df['kp'] = all_results_df['kp'].apply(lambda x: x.tolist())
    all_results_df['hm'] = all_results_df['hm'].apply(lambda x: x.tolist())
    all_results_df.to_csv(os.path.join(args.output_dir, f'allparts_{n_parts}_processed.csv'))

    # Generate valid image list
    prev_valid, all_clips, clip_num = None, [], 0
    for idx, row in all_results_df.iterrows():
        if row.valid == True:
            row['clip_num'] = clip_num
            all_clips.append(row)
            prev_valid = True
        else:
            if prev_valid == True:
                clip_num += 1
            prev_valid = False

    all_clips_df = pd.DataFrame(all_clips)
    all_clips_grp = all_clips_df.reset_index().groupby('clip_num').apply(lambda x: pd.Series([x['kp'].count()]))
    all_clips_grp.columns = ['f_count']
    all_clips_df[all_clips_df.clip_num.isin(all_clips_grp[all_clips_grp.f_count>100].index.values)].to_csv(os.path.join(args.output_dir, f'clips_filtered.csv'))
    all_valid_arr = all_clips_df[all_clips_df.clip_num.isin(all_clips_grp[all_clips_grp.f_count>100].index.values)].index.values
    np.savetxt(os.path.join(args.output_dir, f'imglist_filtered.csv'), all_valid_arr, fmt='%s')

if __name__ == "__main__":
    main()
