# visualize_anterior_bscan.py

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

bscan_path = Path(
    "rl_whole_eye_oct/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_bscans.tif"
)

seg_path = Path(
    "rl_whole_eye_oct/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_depth_segmentation.tif"
)

stack = tiff.imread(bscan_path)
seg_stack = tiff.imread(seg_path)

print("B-scan shape:", stack.shape)
print("Seg shape:", seg_stack.shape)
print("Seg unique values:", np.unique(seg_stack))

# single crop box covering all anatomy in the volume
volume_mask = seg_stack > 0

rows = np.where(volume_mask.any(axis=(0, 2)))[0]
cols = np.where(volume_mask.any(axis=(0, 1)))[0]

rmin, rmax = rows[0], rows[-1]
cmin, cmax = cols[0], cols[-1]

margin_y = 30
margin_x = 180

rmin = max(0, rmin - margin_y)
rmax = min(stack.shape[1], rmax + margin_y)

cmin = max(0, cmin - margin_x)
cmax = min(stack.shape[2], cmax + margin_x)

print(f"Crop box: rows {rmin}:{rmax}, cols {cmin}:{cmax}")

# comparison figure
indices = [0, 50, 100, 125, 150, 200, 249]

fig, axes = plt.subplots(2, len(indices), figsize=(20, 8))

for j, idx in enumerate(indices):

    axes[0, j].imshow(stack[idx], cmap="gray", aspect="auto")
    axes[0, j].set_title(f"Orig {idx}")
    axes[0, j].axis("off")

    crop = stack[idx, rmin:rmax, cmin:cmax]

    axes[1, j].imshow(crop, cmap="gray", aspect="auto")
    axes[1, j].set_title(f"Crop {idx}")
    axes[1, j].axis("off")

plt.tight_layout()
plt.savefig("crop_comparison.png", dpi=300)
plt.close()

print("Saved crop_comparison.png")

# segmentation overlay sanity check
idx = 0

plt.figure(figsize=(8, 6))
plt.imshow(stack[idx], cmap="gray")
plt.imshow(seg_stack[idx], alpha=0.4)
plt.axis("off")
plt.tight_layout()
plt.savefig("segmentation_overlay.png", dpi=300)
plt.close()

print("Saved segmentation_overlay.png")
