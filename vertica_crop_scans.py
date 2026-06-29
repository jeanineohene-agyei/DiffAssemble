import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BSCAN_PATH = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_bscans.tif")
SEG_PATH = Path("datasets/rl-whole-eye/processed_data/batch_000/881ca00e6c71/881ca00e6c71_001_depth_segmentation.tif")

SEED = 0

DENSE_SIZE = 30

MIN_NUM_BATCHES = 2
MAX_NUM_BATCHES = 4
MIN_BATCH_SIZE = 2
MAX_BATCH_SIZE = 10

CROP_L = 220

VALID_SCAN_START = 50
VALID_SCAN_END = 200

MARGIN_Y = 40
MARGIN_X_LEFT = 260
MARGIN_X_RIGHT = 260

MIN_MASK_PIXELS = 500

DISPLAY_HEIGHT = 0.55
SCAN_SPACING = 0.10
WIDTH_STRETCH = 1.8

OUT_PATH = "random_batch_crop_test.png"


def norm_img(img):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


def add_img_with_air_transparent(ax, img, extent, alpha_floor=0.08):
    img = norm_img(img)

    # low-intensity air/background becomes mostly transparent
    alpha = np.clip((img - alpha_floor) / (1.0 - alpha_floor), 0, 1)

    ax.imshow(
        img,
        cmap="gray",
        extent=extent,
        aspect="auto",
        zorder=2,
        alpha=alpha,
    )
    
    
def rotate_volume(vol):
    # Rotate every B-scan first. After this, all crop logic uses rotated coords.
    return np.rot90(vol, k=3, axes=(1, 2))


def full_display_width(full_shape):
    H, W = full_shape
    return DISPLAY_HEIGHT * (W / H) * WIDTH_STRETCH


def add_full_img(ax, img, x_center, y_center, full_width):
    imshow_transparent(
        ax=ax,
        img=img,
        extent=[
            x_center - full_width / 2,
            x_center + full_width / 2,
            y_center - DISPLAY_HEIGHT / 2,
            y_center + DISPLAY_HEIGHT / 2,
        ],
    )
    
def imshow_transparent(ax, img, extent, alpha_floor=0.10):
    img = norm_img(img)

    alpha = np.clip((img - alpha_floor) / (1.0 - alpha_floor), 0, 1)

    ax.imshow(
        img,
        cmap="gray",
        extent=extent,
        aspect="auto",
        alpha=alpha,
        zorder=2,
    )


def add_crop_img(ax, img, c0, c1, y_center, full_shape):
    img = norm_img(img)
    img = img[:, c0:c1]

    x0 = col_to_full_x(c0, full_shape)
    x1 = col_to_full_x(c1, full_shape)

    add_img_with_air_transparent(
        ax=ax,
        img=img,
        extent=[
            x0,
            x1,
            y_center - DISPLAY_HEIGHT / 2,
            y_center + DISPLAY_HEIGHT / 2,
        ],
        alpha_floor=0.10,
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
    return [sample_one_batch(rng, dense_indices) for _ in range(num_batches)]


def sample_crop_cols(rng, W):
    center = rng.integers(CROP_L // 2, W - CROP_L // 2 + 1)
    return center - CROP_L // 2, center + CROP_L // 2


def sample_batch_crop_windows(rng, batches, W):
    return [sample_crop_cols(rng, W) for _ in batches]


def col_to_full_x(col, full_shape):
    H, W = full_shape
    width = full_display_width(full_shape)
    return ((col / (W - 1)) - 0.5) * width


def anatomy_center_pixels(img, seg):
    m = seg > 0

    if m.sum() < MIN_MASK_PIXELS:
        img = img.astype(np.float32)
        threshold = np.percentile(img, 95)
        m = img > threshold

    rr, cc = np.where(m)

    if len(rr) == 0:
        return None

    return rr.mean(), cc.mean()


def pixel_to_display_xy(row, col, y_scan, full_shape):
    H, W = full_shape

    x = ((col / (W - 1)) - 0.5) * full_display_width(full_shape)
    y = y_scan + (0.5 - (row / (H - 1))) * DISPLAY_HEIGHT

    return x, y


def make_debug_plot(stack_vis, seg_vis, dense_indices, batches, crop_windows, out_path):
    full_shape = stack_vis[0].shape
    full_width = full_display_width(full_shape)

    center = (len(dense_indices) - 1) / 2
    ys = (center - np.arange(len(dense_indices))) * SCAN_SPACING
    idx_to_y = dict(zip(dense_indices, ys))

    x_pad = 0.05 * full_width
    xlim = (-full_width / 2 - x_pad, full_width / 2 + x_pad)

    y_pad = SCAN_SPACING * 2
    ylim = (ys.min() - y_pad, ys.max() + y_pad)

    fig, axes = plt.subplots(1, 2, figsize=(14, 18))

    dense_targets = []
    crop_targets = []

    ax = axes[0]

    for idx in dense_indices:
        y_scan = idx_to_y[idx]

        img = stack_vis[idx]
        seg = seg_vis[idx]

        add_full_img(
            ax=ax,
            img=img,
            x_center=0,
            y_center=y_scan,
            full_width=full_width,
        )

        ax.text(xlim[0], y_scan, str(idx), ha="right", va="center", fontsize=8)

        center_px = anatomy_center_pixels(img, seg)
        if center_px is None:
            print("scan {}: skipped dense anatomy center".format(idx))
            continue

        row, col = center_px
        gx, gy = pixel_to_display_xy(row, col, y_scan, full_shape)

        dense_targets.append((idx, gx, gy))
        ax.scatter([gx], [gy], s=60, marker="x", zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Dense volume locked vertically")
    ax.set_xlabel("Full scan width")
    ax.set_ylabel("Locked scan position")
    ax.grid(alpha=0.3)

    ax = axes[1]

    for b, batch_indices in enumerate(batches):
        c0, c1 = crop_windows[b]

        for idx in batch_indices:
            y_scan = idx_to_y[idx]

            img_crop = stack_vis[idx, :, c0:c1]
            seg_crop = seg_vis[idx, :, c0:c1]

            add_crop_img(
                ax=ax,
                img=stack_vis[idx],
                c0=c0,
                c1=c1,
                y_center=y_scan,
                full_shape=full_shape,
            )

            ax.text(xlim[0], y_scan, str(idx), ha="right", va="center", fontsize=8)

            center_px = anatomy_center_pixels(img_crop, seg_crop)
            if center_px is None:
                print("batch {}, scan {}: skipped crop anatomy center".format(b, idx))
                continue

            row, local_col = center_px
            full_col = c0 + local_col

            gx, gy = pixel_to_display_xy(row, full_col, y_scan, full_shape)

            crop_targets.append((b, idx, gx, gy, c0, c1))

            ax.scatter([gx], [gy], s=50, marker="x", zorder=5)
            ax.text(gx + 0.03, gy, "b{}".format(b), va="center", fontsize=7)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Random batches cropped, placed on same dense x-axis")
    ax.set_xlabel("Full scan width")
    ax.set_ylabel("Locked scan position")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return dense_targets, crop_targets


rng = np.random.default_rng(SEED)

stack = tiff.imread(BSCAN_PATH)
seg = tiff.imread(SEG_PATH)

print("stack:", stack.shape)
print("seg:", seg.shape)
print("labels:", np.unique(seg))

stack_vis, seg_vis, crop_box = global_anatomy_crop(stack, seg)

print("global crop before rotation:", stack_vis.shape)
print("global crop box:", crop_box)

stack_vis = rotate_volume(stack_vis)
seg_vis = rotate_volume(seg_vis)

print("global crop after rotation:", stack_vis.shape)

dense_indices = sample_dense_indices(rng, stack_vis.shape[0])
batches = sample_batches(rng, dense_indices)
crop_windows = sample_batch_crop_windows(rng, batches, stack_vis.shape[2])

print("dense indices:", dense_indices)
print("batches:", batches)
print("crop windows:", crop_windows)

dense_targets, crop_targets = make_debug_plot(
    stack_vis=stack_vis,
    seg_vis=seg_vis,
    dense_indices=dense_indices,
    batches=batches,
    crop_windows=crop_windows,
    out_path=OUT_PATH,
)

print("saved", OUT_PATH)

print("dense anatomy centers:")
for idx, gx, gy in dense_targets:
    print("scan {}: {:.4f}, {:.4f}".format(idx, gx, gy))

print("crop target xy:")
for b, idx, gx, gy, c0, c1 in crop_targets:
    print(
        "batch {}, scan {}: {:.4f}, {:.4f}, crop cols {}:{}".format(
            b, idx, gx, gy, c0, c1
        )
    )