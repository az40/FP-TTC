import cv2
import os
import argparse


def video_to_frames(input_loc, output_loc):
    os.makedirs(output_loc, exist_ok=False)
        
    cap = cv2.VideoCapture(input_loc)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_loc, f"frame_{frame_count:04d}.png"), frame)
        frame_count += 1
    cap.release()


def main(args):
    directory = args.video_dir
    filenames = os.listdir(directory)
    filepaths = [os.path.join(directory, f) for f in filenames if os.path.isfile(os.path.join(directory, f))]
    for path, name in zip(filepaths, filenames):
        video_to_frames(path, os.path.join(directory, name.split(".")[0]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default=None, type=str)
    args = parser.parse_args()
    main(args)
