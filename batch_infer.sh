#!/bin/bash
set -e

python batch_infer.py \ 
--inference_dir "/home/viplab/tartanair_dataset/mono/" \
--out_dir "log/25_07_08-01_06_42_selfcon_ttc/tartanair_test_inference" \ 
--resume "pretrained/fpttc_kitti29y.pth.tar" \
--padding_factor 32 \ 
--upsample_factor 4 \ 
--num_scales 2 \ 
--num_head 1 \ 
--attn_splits_list 2 8 \ 
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \ 
--image_size 640 480 \ 
--frame_diff 1
