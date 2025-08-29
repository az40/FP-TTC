#!/bin/bash
set -e

FOLDER="ME007"
OUT_DIR="ttc_output/tartanair_output_all_frames/${FOLDER}_5_frame_diff"
PATTERN="*.png"

python test_tartanair.py --inference_dir "/home/andrew/tartanair_tools/tartanair-test-mono-release/mono/$FOLDER/" \
--out_dir $OUT_DIR --resume pretrained/fpttc_kitti.pth.tar \
--padding_factor 32 --upsample_factor 4 --num_scales 2 --num_head 1 --attn_splits_list 2 8 --corr_radius_list -1 4 \
--prop_radius_list -1 1