from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff


ROOT = Path("/data/jeanine/processed_data")

NUM_SCANS = 250
DEPTH_PIXELS = 100
OUTPUT_PATH = "enface_test-250.png"


def rotate_volume(vol, bscan_path):
    participant = Path(bscan_path).parent.name

    rotation_overrides = {"a947ccaffbe1": 1}
    default_rotation = 3

    k = rotation_overrides.get(participant, default_rotation)

    return np.rot90(vol, k=k, axes=(1, 2)).copy()


def find_random_volume():
    bscan_paths = sorted(ROOT.glob("batch_*/*/*_bscans.tif"))
    seg_paths = sorted(ROOT.glob("batch_*/*/*_depth_segmentation.tif"))

    if not bscan_paths:
        raise RuntimeError(f"No B-scan volumes found under {ROOT}")

    if not seg_paths:
        raise RuntimeError(f"No segmentation volumes found under {ROOT}")

    seg_lookup = {}

    for seg_path in seg_paths:
        volume_name = seg_path.name.removesuffix("_depth_segmentation.tif")
        key = (seg_path.parent.parent.name, seg_path.parent.name, volume_name)
        seg_lookup[key] = seg_path

    candidates = []

    for bscan_path in bscan_paths:
        volume_name = bscan_path.name.removesuffix("_bscans.tif")
        key = (bscan_path.parent.parent.name, bscan_path.parent.name, volume_name)

        if key in seg_lookup:
            candidates.append((bscan_path, seg_lookup[key]))

    if not candidates:
        raise RuntimeError("Could not find any matched B-scan/segmentation pairs.")

    return random.choice(candidates)


def make_enface(volume, seg, depth_pixels=50):
    num_scans, H, W = volume.shape

    enface = np.zeros((num_scans, W), dtype=np.float32)

    for scan_idx in range(num_scans):
        for x in range(W):
            mask_column = seg[scan_idx, :, x] > 0
            rows = np.where(mask_column)[0]

            if len(rows) == 0:
                continue

            surface_row = rows[0]
            depth_end = min(surface_row + depth_pixels, H)

            values = volume[scan_idx, surface_row:depth_end, x]

            if len(values) > 0:
                enface[scan_idx, x] = values.mean()

    return enface


def normalize_image(img):
    valid = img[img > 0]

    if len(valid) == 0:
        return img

    lo, hi = np.percentile(valid, [1, 99])

    img = (img - lo) / (hi - lo + 1e-8)

    return np.clip(img, 0, 1)


def main():
    bscan_path, seg_path = find_random_volume()

    print("B-scan:")
    print(bscan_path)

    print("\nSegmentation:")
    print(seg_path)

    volume = tiff.imread(bscan_path)
    seg = tiff.imread(seg_path)

    print("\nOriginal volume shape:", volume.shape)
    print("Original seg shape:", seg.shape)

    volume = rotate_volume(volume, bscan_path)
    seg = rotate_volume(seg, bscan_path)

    print("Rotated volume shape:", volume.shape)
    print("Rotated seg shape:", seg.shape)

    total_scans = volume.shape[0]

    if total_scans < NUM_SCANS:
        raise RuntimeError(f"Volume only has {total_scans} scans, but NUM_SCANS={NUM_SCANS}")

    center = total_scans // 2
    start = center - NUM_SCANS // 2
    end = start + NUM_SCANS

    volume_middle = volume[start:end]
    seg_middle = seg[start:end]

    print(f"\nUsing middle scans {start}:{end}")
    print("Subset shape:", volume_middle.shape)

    enface = make_enface(volume_middle, seg_middle, depth_pixels=DEPTH_PIXELS)
    enface = normalize_image(enface)
    
    print("En-face shape:", enface.shape)

    
    plt.figure(figsize=(16, 6))

    plt.imshow(
        enface,
        cmap="gray",
        aspect="auto",
        origin="upper",
    )

    plt.xlabel("Lateral position within B-scan")
    plt.ylabel("B-scan")
    plt.title(
        f"En-face projection | scans {start}-{end - 1} | "
        f"{DEPTH_PIXELS} pixels below segmented surface"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()
    
    

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()