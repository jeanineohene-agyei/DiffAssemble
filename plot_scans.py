import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

bscan_path = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_bscans.tif")
seg_path = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_depth_segmentation.tif")

OUT_PATH = "air_removed_check.png"

MARGIN_R = 80
MARGIN_C = 300
DISPLAY_WIDTH = 2.6

indices = [75, 76]

stack = tiff.imread(bscan_path)
seg = tiff.imread(seg_path)

print("stack:", stack.shape)
print("seg:", seg.shape)
print("labels:", np.unique(seg))


def norm_img(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def add_img(ax, img, x, y, width=2.6):
    img = norm_img(img)
    H, W = img.shape
    height = width * (H / W)

    ax.imshow(
        img,
        cmap="gray",
        extent=[x - width / 2, x + width / 2, y - height / 2, y + height / 2],
        aspect="auto",
        zorder=2,
    )


def remove_air_above_scanline(stack, seg, margin_c=250):
    """
    Keeps all 500 scanline rows.
    Removes only leading air along the depth axis.
    Uses volume-wide segmentation when available.
    """
    mask = seg > 0

    cols = np.where(mask.any(axis=(0, 1)))[0]

    if len(cols) == 0:
        print("No segmentation found. Keeping original.")
        return stack, seg, (0, stack.shape[1], 0, stack.shape[2]), "failed"

    c0 = max(0, cols[0] - margin_c)
    c1 = stack.shape[2]

    r0 = 0
    r1 = stack.shape[1]

    return stack[:, r0:r1, c0:c1], seg[:, r0:r1, c0:c1], (r0, r1, c0, c1), "leading_depth_air_only"

stack_crop, seg_crop, crop_box, source = remove_air_above_scanline(
    stack,
    seg,
    margin_c=250,
)

print("crop source:", source)
print("crop box:", crop_box)
print("air-removed stack:", stack_crop.shape)

xs = np.arange(len(indices)) * 3.0

fig, axes = plt.subplots(2, 1, figsize=(18, 8))

for ax, data, title in [
    (axes[0], stack, "Original B-scans"),
    (axes[1], stack_crop, "After removing air/free space"),
]:
    for idx, x in zip(indices, xs):
        add_img(ax, data[idx], x, 0, width=DISPLAY_WIDTH)
        ax.text(x, -0.45, str(idx), ha="center", fontsize=10)

    ax.set_xlim(xs.min() - 1.8, xs.max() + 1.8)
    ax.set_ylim(-0.65, 0.65)
    ax.set_title(title)
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print("saved", OUT_PATH)