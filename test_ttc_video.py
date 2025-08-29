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


def main(args):
    model_loaded = FpTTC(num_scales=args.num_scales,
                         feature_channels=args.feature_channels,
                         upsample_factor=args.upsample_factor,
                         num_head=args.num_head,
                         ffn_dim_expansion=args.ffn_dim_expansion,
                         num_transformer_layers=args.num_transformer_layers,
                         reg_refine=args.reg_refine,
                         train=False).to(device)
    num_params = sum(p.numel() for p in model_loaded.parameters())
    print('Number of params:', num_params)

    if args.resume is not None:
        loc = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(args.resume, map_location=loc)
        if 'net' in checkpoint:
            model_loaded.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['net'].items()})
        elif 'state_dict' in checkpoint:
            model_loaded.load_state_dict(checkpoint['state_dict'])
        else:
            model_loaded.load_state_dict(checkpoint)


    time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")

    model_loaded.eval()
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=False)

    # path1, path2 = 'test_img/2341.jpg', 'test_img/2344.jpg'
    inference_dir = args.inference_dir
    filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    # print(filenames)
    print('%d images found' % len(filenames))

    w, h = 640, 480
    output_filename = os.path.join(args.out_dir, args.filename)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    vid_writer = cv2.VideoWriter(output_filename, fourcc, 15, (w, h))
    # w, h = 320, 320
    total = 0
    with torch.no_grad():
        for test_id in tqdm(range(0, len(filenames))):
            # if test_id % 5 != 0:
            #     continue
            
            file_1 = filenames[test_id]
            try:
                file_2 = filenames[test_id + 5]
            except IndexError:
                print("Out of files")
            # file_1 = inference_dir + 'image_' + str(test_id).zfill(4) + '.png'
            # file_2 = inference_dir + 'image_' + str(test_id + 15).zfill(4) + '.png'

            image1 = Image.open(file_1).convert('RGB')
            image2 = Image.open(file_2).convert('RGB')
            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)
            image1 = cv2.resize(image1, (w,h))
            image2 = cv2.resize(image2, (w,h))
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

            padding_factor = 32
            inference_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                            int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

            ori_size = image1.shape[-2:]
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                                        align_corners=True)
                image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                                        align_corners=True)

            start = time.time()
            scale, flow, _ = model_loaded(image1, image2,
                        attn_type=args.attn_type,
                        attn_splits_list=args.attn_splits_list,
                        corr_radius_list=args.corr_radius_list,
                        prop_radius_list=args.prop_radius_list,
                        num_reg_refine=args.num_reg_refine,
                        testing=False)
            if test_id>3:
                total = total+time.time()-start

            # resize back
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                scale = F.interpolate(scale, size=ori_size, mode='bilinear',
                                        align_corners=True)

            ttc_warp_image2 = (scale[0].transpose(0,1).transpose(1,2) - 0.5) / (1.0) # [H, W, 2]
            ttc_warp_image2 = disp2rgb(np.clip(ttc_warp_image2.detach().cpu().numpy(), 0.0, 1.0))
            ttc_warp_image2 = (ttc_warp_image2 * 255.0).astype(np.uint16)
            ttc_warp_image2 = cv2.cvtColor(ttc_warp_image2, cv2.COLOR_RGB2BGR)
            vid_writer.write(ttc_warp_image2)
            # cv2.imwrite(os.path.join(out_dir, 'scale'+str(test_id)+'.png'), ttc_warp_image2)
            # txt_result = scale[0,0].detach().cpu().numpy()
            # np.save(os.path.join(out_dir, str(test_id)+'_eta_out.npy'), txt_result)

    vid_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--out_dir', default=None, type=str)
    parser.add_argument('--filename', default=None, type=str)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')

    # dataset
    parser.add_argument('--padding_factor', default=16, type=int, 
                        help='the input should be divisible by padding_factor, otherwise do padding or resizing')

    # model: parameter-free
    parser.add_argument('--attn_type', default='swin', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    # model: learnable parameters
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')

    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    main(args)
