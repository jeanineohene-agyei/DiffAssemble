import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BSCAN_PATH = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_bscans.tif")
SEG_PATH = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_depth_segmentation.tif")

SEED = 0

DENSE_SIZE = 30

MIN_NUM_BATCHES = 2
MAX_NUM_BATCHES = 8

MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 10

SCAN_SPACING = 0.08

CROP_L = 220

VALID_SCAN_START = 50
VALID_SCAN_END = 200

MARGIN_Y = 40
MARGIN_X_LEFT = 260
MARGIN_X_RIGHT = 260

MIN_MASK_PIXELS = 500

OUT_PATH = "random_batch_crop_test.png"


def norm_img(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def add_img(ax, img, x, y, height=0.75):
    img = norm_img(img)
    img = np.rot90(img, k=3)

    H, W = img.shape
    width = height * (W / H)

    ax.imshow(
        img,
        cmap="gray",
        extent=[x - width / 2, x + width / 2, y - height / 2, y + height / 2],
        aspect="auto",
        zorder=2,
    )


def global_anatomy_crop(stack, seg):
    mask = seg > 0

    rows = np.where(mask.any(axis=(0, 2)))[0]
    cols = np.where(mask.any(axis=(0, 1)))[0]

    r0 = max(0, rows[0] - MARGIN_Y)
    r1 = min(stack.shape[1], rows[-1] + MARGIN_Y)

    c0 = max(0, cols[0] - MARGIN_X_LEFT)
    c1 = min(stack.shape[2], cols[-1] + MARGIN_X_RIGHT)

    return stack[:, r0:r1, c0:c1], seg[:, r0:r1, c0:c1], (r0, r1, c0, c1)


def sample_dense_indices(rng, n_scans):
    start_min = VALID_SCAN_START
    start_max = min(VALID_SCAN_END, n_scans) - DENSE_SIZE

    start = rng.integers(start_min, start_max + 1)
    return list(range(start, start + DENSE_SIZE))


def sample_one_batch(rng, dense_indices):
    max_size = min(MAX_BATCH_SIZE, len(dense_indices))
    min_size = min(MIN_BATCH_SIZE, max_size)

    batch_size = rng.integers(min_size, max_size + 1)
    start = rng.integers(0, len(dense_indices) - batch_size + 1)

    return dense_indices[start:start + batch_size]


def sample_batches(rng, dense_indices):
    num_batches = rng.integers(MIN_NUM_BATCHES, MAX_NUM_BATCHES + 1)

    batches = [
        sample_one_batch(rng, dense_indices)
        for _ in range(num_batches)
    ]

    return batches


def sample_crop_rows(rng, H):
    center = rng.integers(CROP_L // 2, H - CROP_L // 2 + 1)
    r0 = center - CROP_L // 2
    r1 = center + CROP_L // 2
    return r0, r1


def anatomy_xy_from_crop(img_crop, seg_crop):
    # first try segmentation
    m = seg_crop > 0

    # if segmentation is too sparse, use bright OCT signal instead
    if m.sum() < MIN_MASK_PIXELS:
        img = img_crop.astype(np.float32)
        threshold = np.percentile(img, 95)
        m = img > threshold

    rr, cc = np.where(m)

    if len(rr) == 0:
        return None

    center_r = rr.mean()
    center_c = cc.mean()

    H, W = seg_crop.shape

    x = 2 * center_c / (W - 1) - 1
    y = 2 * center_r / (H - 1) - 1

    return x, y


def crop_xy_to_grid(x_scan, crop_x_norm, crop_y_norm, displayed_height, crop_shape):
    H, W = crop_shape
    display_width = displayed_height * (H / W)

    grid_x = x_scan + crop_y_norm * (display_width / 2)
    grid_y = -crop_x_norm * (displayed_height / 2)

    return grid_x, grid_y


def make_debug_plot(stack_vis, seg_vis, dense_indices, batches, crop_rows, out_path):
    center = (len(dense_indices) - 1) / 2
    xs = (np.arange(len(dense_indices)) - center) * SCAN_SPACING
    idx_to_x = dict(zip(dense_indices, xs))

    cr0, cr1 = crop_rows

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    ax = axes[0]

    for idx, x in zip(dense_indices, xs):
        add_img(ax, stack_vis[idx], x, 0, height=1.15)
        ax.text(x, -0.72, str(idx), ha="center", fontsize=8)

    ax.scatter(xs, np.zeros(len(xs)), s=20)
    x_pad = SCAN_SPACING * 2
    xlim = (xs.min() - x_pad, xs.max() + x_pad)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.75, 0.75)
    ax.set_title("Full dense volume locked on grid")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)

    ax = axes[1]
    crop_display_height = 1.15
    crop_targets = []

    for b, batch_indices in enumerate(batches):
        # y_offset = 0.18 * (b - (len(batches) - 1) / 2)
        y_offset = 0

        for idx in batch_indices:
            x_scan = idx_to_x[idx]

            img_crop = stack_vis[idx, cr0:cr1, :]
            seg_crop = seg_vis[idx, cr0:cr1, :]

            add_img(ax, img_crop, x_scan, y_offset, height=crop_display_height)
            ax.text(x_scan, y_offset - 0.72, str(idx), ha="center", fontsize=8)

            xy = anatomy_xy_from_crop(img_crop, seg_crop)
            if xy is None:
                print(f"batch {b}, scan {idx}: skipped xy, not enough anatomy pixels")
                continue

            crop_x_norm, crop_y_norm = xy

            gx, gy = crop_xy_to_grid(
                x_scan=x_scan,
                crop_x_norm=crop_x_norm,
                crop_y_norm=crop_y_norm,
                displayed_height=crop_display_height,
                crop_shape=seg_crop.shape,
            )

            gy += y_offset

            crop_targets.append((b, idx, gx, gy))

            ax.scatter([gx], [gy], s=50, marker="x", zorder=5)
            ax.text(gx, gy + 0.04, f"b{b}", ha="center", fontsize=7)

    for b, batch_indices in enumerate(batches):
        batch_xs = [idx_to_x[i] for i in batch_indices]
        ax.scatter(batch_xs, [y_offset] * len(batch_xs), s=20)

    x_pad = SCAN_SPACING * 2
    xlim = (xs.min() - x_pad, xs.max() + x_pad)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.9, 0.9)
    ax.set_title("Random batches with same crop row window")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return crop_targets


rng = np.random.default_rng(SEED)

stack = tiff.imread(BSCAN_PATH)
seg = tiff.imread(SEG_PATH)

print("stack:", stack.shape)
print("seg:", seg.shape)
print("labels:", np.unique(seg))

stack_vis, seg_vis, crop_box = global_anatomy_crop(stack, seg)

print("global crop:", stack_vis.shape)
print("global crop box:", crop_box)

dense_indices = sample_dense_indices(rng, stack_vis.shape[0])
batches = sample_batches(rng, dense_indices)
crop_rows = sample_crop_rows(rng, stack_vis.shape[1])

print("dense indices:", dense_indices)
print("batches:", batches)
print("crop rows:", crop_rows)

crop_targets = make_debug_plot(
    stack_vis=stack_vis,
    seg_vis=seg_vis,
    dense_indices=dense_indices,
    batches=batches,
    crop_rows=crop_rows,
    out_path=OUT_PATH,
)

print("saved", OUT_PATH)
print("crop target xy:")
for b, idx, gx, gy in crop_targets:
    print(f"batch {b}, scan {idx}: {gx:.4f}, {gy:.4f}")