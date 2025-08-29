import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
import argparse
import datetime
import os
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg
from glob import glob
from PIL import Image
from tqdm import tqdm


def preprocess(batch):
    transforms = T.Compose(
        [
            # T.ConvertImageDtype(torch.float32),
            T.Lambda(lambda x: x / 255.0),  # manually scale to [0, 1]
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            # T.Resize(size=(520, 960))
            # T.Resize(size=(1080, 1920)),
        ]
    )
    batch = transforms(batch)
    return batch


def main(args, inference_dir, out_dir):
    torch.cuda.set_device(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    time_stamp = datetime.datetime.now().strftime("%y_%m_%d-%H_%M_%S")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=False)

    filenames = sorted(glob(inference_dir + '/*.png') + glob(inference_dir + '/*.jpg'))
    # print(filenames)
    print('%d images found' % len(filenames))

    with torch.no_grad():
        for test_id in tqdm(range(0, len(filenames) - 5)):
            file_1 = filenames[test_id]
            file_2 = filenames[test_id + 5]

            image1 = Image.open(file_1).convert('RGB')
            image2 = Image.open(file_2).convert('RGB')
            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)
            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)
            
            image1 = preprocess(image1).to(device)
            image2 = preprocess(image2).to(device)
            flows = model(image1.to(device), image2.to(device))
            predicted_flow = flows[-1][0]
            flow_img = flow_to_image(predicted_flow).to("cpu")
            write_jpeg(flow_img, out_dir + f"/flow_{test_id:04d}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--out_dir', default=None, type=str)
    args = parser.parse_args()

    inference_subdirs = [os.path.join(args.inference_dir, subdir) for subdir in os.listdir(args.inference_dir) if os.path.isdir(os.path.join(args.inference_dir, subdir))]
    subdir_names = [subdir for subdir in os.listdir(args.inference_dir) if os.path.isdir(os.path.join(args.inference_dir, subdir))]
    out_subdirs = [os.path.join(args.out_dir, subdir_name) for subdir_name in subdir_names]

    for inference_subdir, out_subdir in zip(inference_subdirs, out_subdirs):
        main(args, inference_subdir, out_subdir)
