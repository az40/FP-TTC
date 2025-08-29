from PIL import Image
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import datetime
from glob import glob
from fpttc.fp_ttc import FpTTC
from utils.trainer import TTCTrainer
from utils.draw import disp2rgb, flow_uv_to_colors
from tqdm import tqdm
import subprocess
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # inference on images or videos
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--frame_rate', default=None, type=int)
    args = parser.parse_args()
    subdir_names = [subdir for subdir in os.listdir(args.inference_dir) if os.path.isdir(os.path.join(args.inference_dir, subdir))]
    subdir_paths = [os.path.join(args.inference_dir, subdir) for subdir in subdir_names]

    for subdir_name, subdir_path in zip(subdir_names, subdir_paths):
        subprocess.run(["ffmpeg", 
                        "-framerate",
                        f"{args.frame_rate}",
                        "-i",
                        f"{os.path.join(subdir_path, f'scale%04d.png')}",
                        # f"{os.path.join(subdir_path, f'%06d.png')}",
                        "-c:v",
                        "libx264", 
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        f"{os.path.join(args.inference_dir, subdir_name + '_ttc.mp4')}"])
