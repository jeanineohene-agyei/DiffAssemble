import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

bscan_path = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_bscans.tif")
seg_path = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_depth_segmentation.tif")

stack = tiff.imread(bscan_path)
seg = tiff.imread(seg_path)

print("stack:", stack.shape)
print("seg:", seg.shape)
print("labels:", np.unique(seg))

mask = seg > 0
rows = np.where(mask.any(axis=(0, 2)))[0]
cols = np.where(mask.any(axis=(0, 1)))[0]

margin_y = 40
margin_x_left = 260
margin_x_right = 260

r0 = max(0, rows[0] - margin_y)
r1 = min(stack.shape[1], rows[-1] + margin_y)
c0 = max(0, cols[0] - margin_x_left)
c1 = min(stack.shape[2], cols[-1] + margin_x_right)

stack_crop = stack[:, r0:r1, c0:c1]

print("crop:", stack_crop.shape)

def norm_img(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

def add_img(ax, img, x, y, height=0.75):
    img = norm_img(img)

    # rotate only for visualization so it appears tall
    img = np.rot90(img, k=3)

    H, W = img.shape
    width = height * (W / H)

    ax.imshow(
        img,
        cmap="gray",
        extent=[
            x - width / 2,
            x + width / 2,
            y - height / 2,
            y + height / 2,
        ],
        aspect="auto",
        zorder=2,
    )

indices = list(range(75, 97))

# place scans left-to-right on a 2D grid
xs = np.linspace(-1.0, 1.0, len(indices))
ys = np.zeros(len(indices))

fig, ax = plt.subplots(figsize=(18, 6))

for idx, x, y in zip(indices, xs, ys):
    add_img(ax, stack_crop[idx], x, y, height=1.25)
    ax.text(x, -0.72, str(idx), ha="center", fontsize=10)

ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-0.75, 0.75)
ax.set_aspect("auto")
ax.scatter(xs, ys, s=30)
ax.set_title("Cropped B-scans placed on 2D grid")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("bscan_grid_plot.png", dpi=300)
plt.close()

print("saved bscan_grid_plot.png")