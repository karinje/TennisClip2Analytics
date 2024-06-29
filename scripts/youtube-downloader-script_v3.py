import argparse
from pytube import YouTube
import moviepy.editor as mp
from pathlib import Path
import subprocess
from tqdm import tqdm
import time

def on_progress(stream, chunk, bytes_remaining):
    """Callback function for download progress"""
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    if time.time() - on_progress.last_update > 0.5:  # Update every 0.5 seconds
        download_pbar.update(bytes_downloaded - download_pbar.n)
        on_progress.last_update = time.time()

on_progress.last_update = 0

def download_video(url, start_time, end_time, output_file):
    """
    Download a segment of a YouTube video and save it to the specified output file.
    """
    global download_pbar
    yt = YouTube(url)
    yt.register_on_progress_callback(on_progress)
    stream = yt.streams.filter(res='720p', file_extension='mp4').first()
    output_path = str(output_file.parent)
    print("Downloading full video...")
    download_pbar = tqdm(total=stream.filesize, unit='B', unit_scale=True, desc="Downloading")
    stream.download(output_path=output_path)
    download_pbar.close()
    video_path = f"{output_path}/{stream.default_filename}"
    print(f'Full video saved at: {video_path}')
    print("Subclipping video segment...")
    video = mp.VideoFileClip(video_path)
    video_segment = video.subclip(start_time, end_time)
    video_segment.write_videofile(str(output_file), logger='bar')
    video.close()

def extract_frames(input_file, output_folder):
    """
    Extract frames from a video file using FFmpeg and save them as images.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get video duration
    duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(input_file)]
    duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
    
    # Extract frames using FFmpeg
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', str(input_file),
        '-vf', 'fps=30',  # Extract 1 frame per second, adjust as needed
        '-q:v', '2',  # High quality (2-31, lower is better)
        f'{output_folder}/img_%04d.jpg'
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    
    with tqdm(total=int(duration), unit='s', desc="Extracting frames") as extract_pbar:
        for line in process.stdout:
            if "time=" in line:
                time_str = line.split("time=")[1].split()[0]
                current_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(time_str.split(":"))))
                extract_pbar.update(int(current_time) - extract_pbar.n)
    
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, ffmpeg_cmd)

def main():
    """
    Main function to parse command-line arguments and execute the video download and frame extraction.
    """
    parser = argparse.ArgumentParser(description='Download YouTube video segment and extract frames.')
    parser.add_argument('--url', required=True, help='YouTube video URL')
    parser.add_argument('--start_time', required=True, help='Start time of the segment (HH:MM:SS)')
    parser.add_argument('--end_time', required=True, help='End time of the segment (HH:MM:SS)')
    parser.add_argument('--output_file', required=True, help='Output video file path')
    parser.add_argument('--extract_frames', action='store_true', help='Extract frames after downloading')
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    download_video(args.url, args.start_time, args.end_time, output_file)
    
    if args.extract_frames:
        print("Extracting frames...")
        frames_folder = output_file.parent / (output_file.stem + "_frames")
        extract_frames(output_file, frames_folder)
        print(f"Frames extracted to: {frames_folder}")

if __name__ == "__main__":
    main()
