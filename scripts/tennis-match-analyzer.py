import argparse
import subprocess
import os
import yaml
import time
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager


@contextmanager
def change_dir(new_dir):
    old_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(old_dir)

def run_command(command, step_name, conda_envs=None, working_dir=None, env_vars=None, check_packages=None):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting {step_name}...")
    conda_path = os.path.expanduser('~/test-storage/miniconda3')  # Adjust this path if necessary
    conda_sh_path = os.path.join(conda_path, 'etc', 'profile.d', 'conda.sh')
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    package_check_code = """

import importlib
import sys

def check_package(package_name):
    try:
        package = importlib.import_module(package_name)
        print(f'Package {{package_name}} is installed at: {{package.__file__}}')
    except ImportError:
        print(f'Package {{package_name}} is not installed or cannot be imported.')

for package in {0}:
    check_package(package)

print(f'Python path: {{sys.path}}')
"""
    # Start building the command
    full_command = f"""
source {conda_sh_path}
conda init zsh
source ~/.zshrc
"""
    if conda_envs:
        for env_name in conda_envs:
            full_command += f"conda activate {env_name} && "
    if working_dir:
        full_command += f"export PYTHONPATH={working_dir}:$PYTHONPATH && "
    full_command += "echo \"Current Conda environment: $CONDA_DEFAULT_ENV\" && which python && python --version"
    if check_packages:
        package_list = str(check_packages)
        full_command += f' && python -c "{package_check_code.format(package_list)}"'
    full_command += f" && {command}"
    if conda_envs:
        full_command += " && conda deactivate" * len(conda_envs)

    # Use a context manager to change directory if working_dir is specified
    with change_dir(working_dir) if working_dir else contextmanager(lambda: (yield))():
        # Run the command
        process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, executable='/bin/bash', env=env)
        
        output = []
        for line in iter(process.stdout.readline, ''):
            print(line.rstrip(), flush=True)
            output.append(line.rstrip())
        
        process.stdout.close()
        return_code = process.wait()
    
    if return_code != 0:
        print(f"Error executing command: {full_command}")
        print("\n".join(output))
        sys.exit(1)
    
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {step_name}")
    return "\n".join(output)



def should_run_step(step_name, steps_to_run):
    return "all" in steps_to_run or step_name in steps_to_run

def main(config_file):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting tennis match analysis")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    video_name = config['video_name']
    base_dir = config['base_dir']
    shared_input_dir = config['shared_input_dir']
    steps_to_run = config.get('steps_to_run', ['all'])

    print(f"Processing video: {video_name}")

    # Create a directory for this specific video/show_viz
    video_dir = os.path.join(base_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)

    # Derive file names based on video_name
    output_file = os.path.join(video_dir, f"{video_name}.mp4")
    frames_dir = str( Path(output_file).parent / (Path(output_file).stem + "_frames"))
    filtered_frames_dir = os.path.join(video_dir, "filtered_frames_files")
    ref_dir = frames_dir if config['ref_image_dir'] is None else config['ref_image_dir']
    ref_image = os.path.join(ref_dir, config['ref_image'])
    print(f'ref_image_path: {ref_image}')
    filtered_img_list = os.path.join(filtered_frames_dir, 'imglist_filtered.csv')
    filtered_clip_csv = os.path.join(filtered_frames_dir, 'clips_filtered.csv')
    print(f'filtered_img_list: {filtered_img_list}')
    court_kp_file = os.path.join(video_dir, f"{video_name}_court_kp.txt")
    data_prep_output = os.path.join(video_dir, f"{video_name}_data_prep_results.pkl")
    fixed_output = os.path.join(video_dir, f"{video_name}_fixed.csv")
    serve_end_pts_file = os.path.join(video_dir, f"{video_name}_serve_end_pts.pkl")
    serve_endofpoint_df_file = os.path.join(video_dir, f"{video_name}_serve_endofpoint_df_out.csv")
    hit_pts_file = os.path.join(video_dir, f"{video_name}_hit_pts.pkl")
    final_df_file = os.path.join(video_dir, f"{video_name}_final_df.csv")
    vitpose_output_dir = os.path.join(video_dir, "vitpose_output")

    # Derive shared input file names
    bbox_file1 = os.path.join(shared_input_dir, f"{video_name}_bbox1.json")
    bbox_file2 = os.path.join(shared_input_dir, f"{video_name}_bbox2.json")
    ball_file = os.path.join(shared_input_dir, f"{video_name}_ball.csv")
    court_file = os.path.join(filtered_frames_dir, f"clips_filtered.csv")
    override_file = os.path.join(shared_input_dir, f"{video_name}_override.csv")
    hit_override_file = os.path.join(shared_input_dir, f"{video_name}_hit_override.csv")
    turn_override_file = os.path.join(shared_input_dir, f"{video_name}_turn_override.csv")
    points_override_file = os.path.join(shared_input_dir, f"{video_name}_points_override.csv")
    points_drop_file = os.path.join(shared_input_dir, f"{video_name}_points_drop.csv")

    # YouTube video download and frame extraction
    if should_run_step("download_and_extract", steps_to_run):
        run_command(f"python scripts/youtube-downloader-ffmpeg-subclip.py --url {config['match_url']} --start_time {config['start_time']} --end_time {config['end_time']} --output_file {output_file} --extract_frames",
                    "YouTube video download and frame extraction")

    # Tennis court detection
    if should_run_step("court_detection", steps_to_run):
        run_command(f"python TennisCourtDetector/infer_in_image.py --input_path {ref_image} --model_path {config['model_path']} --output_path {court_kp_file} --use_refine_kps --use_homography",
                    "Tennis court detection")

    # Frame filtering
    if should_run_step("frame_filtering", steps_to_run):
        run_command(f"python scripts/tennis-court-filter-script.py --input_dir {frames_dir} --ref_img {ref_image} --court_kp_file {court_kp_file} --output_dir {filtered_frames_dir} --num_cores {config['num_cores']} --part_size {config['part_size']}",
                    "Frame filtering")

    # Run inference
    if should_run_step("run_inference", steps_to_run):
        run_command(f"python training/run_inference.py --infer_data_path {filtered_img_list} --train_data_path {config['train_data_path']} --mode test --samples_per_batch {config['samples_per_batch']} --load_learner {config['load_learner']} --num_inp_images {config['num_inp_images']} --target_img_position {config['target_img_position']} --final_act {config['final_act']} --results_file {ball_file}",
                    "Running inference")

    # ViTPose model

    if should_run_step("vitpose_processing", steps_to_run):
        run_command(f"python model_v3.py --det_model_name '{config['det_model_name']}' --pose_model_name '{config['pose_model_name']}' --clip_csv {filtered_clip_csv} --output_dir {vitpose_output_dir}  --output_json {bbox_file1} --start_row {config['start_row']} --end_row {config['end_row']} --step {config['step']}"
        ,"ViTPose model processing"
        , conda_envs=["openmmlabv2"]
        , working_dir="/home/ubuntu/test-storage/ViTPose"
        , env_vars={"PYTHONPATH": "/home/ubuntu/test-storage/ViTPose:$PYTHONPATH"}
        , check_packages=["mmdet"]
)
    # Data preparation
    if should_run_step("data_preparation", steps_to_run):
        run_command(f"python scripts/data-preparation-script.py --bbox_file1 {bbox_file1}  --ball_file {ball_file} --court_file {court_file} --output_file {data_prep_output} --local_img_folder {frames_dir}",
                    "Data preparation")

    # Fix outliers
    if should_run_step("fix_outliers", steps_to_run):
        run_command(f"python scripts/fix_outliers.py --override_file {override_file} --input_file {data_prep_output} --output_file {fixed_output} --debug",
                    "Fixing outliers")

    # Identify serve and end of points
    if should_run_step("identify_serve_end", steps_to_run):
        run_command(f"python scripts/identify_serve_endofpoint.py --input_file {fixed_output} --output_dict_file {serve_end_pts_file} --output_df_file {serve_endofpoint_df_file} --local_img_folder {frames_dir} --debug",
                    "Identifying serve and end of points")

    # Identify hits and bounces
    if should_run_step("identify_hits_bounces", steps_to_run):
        run_command(f"python scripts/identify_hits_and_bounce.py --input_file {serve_endofpoint_df_file} --serve_endofpoint_dict_file {serve_end_pts_file} --turn_override_file {turn_override_file} --hit_override_file {hit_override_file} --output_dict_file {serve_end_pts_file} --output_df_file {serve_endofpoint_df_file} --debug",
                    "Identifying hits and bounces")

    # Identify final points
    if should_run_step("identify_final_points", steps_to_run):
        run_command(f"python scripts/identify_final_points.py --input_df_file {hit_pts_file} --input_dict_file {hit_pts_file} --points_override_file {points_override_file} --points_drop_file {points_drop_file} --local_img_folder {frames_dir} --court_kp_file {court_kp_file} --output_df_file {final_df_file}",
                    "Identifying final points")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Tennis match analysis for {video_name} completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tennis Match Analysis Script")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()
    main(args.config)
