'''
Example of how to run script:

1) python data_processing.py

this will run with default arguments of : 
batch=rloct_batch_072, volume=00fb08b780c7, batch_size=50, L=200 these are the default values if a certain one is missing

the path gets built as datasets/batch/volume later in the code

where batch is the folder name in datasets, volume (of scans) is the folder name inside batch, batch_size is the number of middle scans to extract per batch, and L is the crop length in y direction (number of pixels to crop)

2) Example of how to run with all arguments specified:

python data_processing.py --batch rloct_batch_072 --volume 0d6f65b02955 --batch_size 30 --L 250

'''

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random as random

NUM_SCANS = 250

def load_volume(img):
    frames_l = []
    for i in range(img.n_frames):
        img.seek(i)
        frames_l.append(np.array(img))
    volume = np.stack(frames_l)
    #print(f'shape of volume: {volume.shape}')  # (num_scans, height, width)
    #print(volume.dtype)
    #print(volume.min(), volume.max())
    '''
    (250, 500, 1928)
    uint8
    0 255
    There are 250 scans per volume and each is 500 x 1928. Every volume is 250.
    '''
    return volume

def extract_middle(volume, num_middle=50):
    #  This exracts the middle n scans from the volume. For example, if num_middle=50, it will extract the middle 50 scans (ie 100, 150).
    mid = NUM_SCANS // 2
    return volume[mid - num_middle // 2 : mid + num_middle // 2]

def sample_crop_start(height=500, crop_l=200):
    # randomly sample a y start such that the window fits within the scan
    return np.random.randint(0, height - crop_l)

def crop_all(middle_scans, crop_x_start, crop_x_end, crop_y_start=0, crop_y_end=None):
    # this will apply crop to all N scans
    cropped = middle_scans[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    print(f"Cropped all scans — shape: {cropped.shape}")  
    # (50, H, W)
    return cropped

def visualize_all_crops(cropped_scans, crop_y_start, crop_y_end, batch_size, L, out_path="crop_check.png"):
    n = len(cropped_scans)
    ncols = 10
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 2 * nrows))
    for i, ax in enumerate(axes.flat):
        if i < n:
            ax.imshow(cropped_scans[i], cmap="gray")
            ax.set_title(str(i), fontsize=6)
        ax.axis("off")
    plt.suptitle(f"All {n} cropped scans, shape per scan: {cropped_scans[0].shape}, batch_size: {batch_size}, L: {L}, y: start: {crop_y_start} end: {crop_y_end}", fontsize=10)
    plt.tight_layout()
    out_path = f"crop_check_{batch_size}batch_L{L}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="rloct_batch_072", help="batch folder name datasets/")
    parser.add_argument("--volume", type=str, default="00fb08b780c7", help="volume of scan's folder name inside  batch")
    parser.add_argument("--batch_size", type=int, default=50, help="number of middle scans to extract per batch")
    parser.add_argument("--L", type=int, default=200, help="crop length in y direction (number of pixels to crop)")
    args = parser.parse_args()

    # build path from args
    scan_path = Path(f"datasets/{args.batch}/{args.volume}")
    tif_files = sorted(scan_path.glob("*_bscans.tif"))
    img = Image.open(tif_files[0])

    volume = load_volume(img)
    middle_scans = extract_middle(volume, args.batch_size) 
    # here batch_size is the number of middle scans to extract
    #print(middle_scans.shape)
    # get (50, 500, 1928) as expected

    crop_y_start = sample_crop_start(height=500, crop_l=args.L)
    print(f"Randomly sampled crop_y_start: {crop_y_start}")
    crop_y_end = crop_y_start + args.L
    print(f"crop_y_end: {crop_y_end}")
    print(f"Crop window: y: {crop_y_start}:{crop_y_end}, difference {crop_y_end - crop_y_start}")

    # x stays fixed at full 1928 only y is cropped
    cropped = crop_all(middle_scans, crop_x_start=0, crop_x_end=1928, crop_y_start=crop_y_start, crop_y_end=crop_y_end)
    
    out_path = f"crop_check_{args.batch_size}batch_L{args.L}.png"
    visualize_all_crops(cropped, crop_y_start, crop_y_end, args.batch_size, args.L, out_path=out_path)

if __name__ == "__main__":
    main()