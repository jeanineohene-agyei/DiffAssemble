# Example usage:
#   python data_processing.py --batch rloct_batch_072 --volume 00fb08b780c7
#   python data_processing.py --batch rloct_batch_072 --volume 0d6f65b02955

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

NUM_SCANS = 250

def load_volume(img):
    frames_l = []
    for i in range(img.n_frames):
        img.seek(i)
        frames_l.append(np.array(img))
    volume = np.stack(frames_l)
    print(f'shape of volume: {volume.shape}')  # (num_scans, height, width)
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
    #  from slack: dont need all, can focus on middle 50 and create batches using those 
    mid = NUM_SCANS // 2
    return volume[mid - num_middle // 2 : mid + num_middle // 2]

def crop_all(middle_scans, crop_x_start, crop_x_end, crop_y_start=0, crop_y_end=None):
    # this will apply crop to all 50 scans
    cropped = middle_scans[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    print(f"Cropped all scans — shape: {cropped.shape}")  
    # (50, H, W)
    return cropped

def visualize_all_crops(cropped_scans, out_path="crop_check.png"):
    n = len(cropped_scans)
    ncols = 10
    nrows = int(np.ceil(n / ncols))


    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 2 * nrows))
    for i, ax in enumerate(axes.flat):
        if i < n:

            ax.imshow(cropped_scans[i], cmap="gray")
            ax.set_title(str(i), fontsize=6)

        ax.axis("off")

    plt.suptitle(f"All {n} cropped B-scans — shape per scan: {cropped_scans[0].shape}", fontsize=10)
    plt.tight_layout()

    plt.savefig(out_path, dpi=150)

    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="rloct_batch_072",
                        help="batch folder name datasets/")
    parser.add_argument("--volume", type=str, default="00fb08b780c7",
                        help="volume folder name inside  batch")
    args = parser.parse_args()

    # build path from args
    scan_path = Path(f"datasets/{args.batch}/{args.volume}")
    tif_files = sorted(scan_path.glob("*_bscans.tif"))
    
    img = Image.open(tif_files[0])

    volume = load_volume(img)
    middle_scans = extract_middle(volume, 50) # here 50 is the number of middle scans to extract

    #print(middle_scans.shape)
    # get (50, 500, 1928) as expected
    # now need to play around with crop sizes

    # over all, 1000 seems to be best (tested 800-1200)
    cropped = crop_all(middle_scans, crop_x_start=1000, crop_x_end=1928,
                       crop_y_start=0, crop_y_end=500)
    # the length of L we crop is end - start = 1928 - 1000 = 928, which is the width of the cropped image
    # like wise for the height crop 
    visualize_all_crops(cropped, out_path="crop_check.png")

if __name__ == "__main__":
    main()