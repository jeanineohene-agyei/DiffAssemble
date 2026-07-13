from pathlib import Path
import matplotlib.pyplot as plt
import torch
from puzzle_dataset import OCTPuzzleDataset

ROOT = Path("/home/jeanine/DiffAssemble/datasets/rl-whole-eye/processed_data")
OUT = Path("debug_oct_dataset")
OUT.mkdir(exist_ok=True)

bscan_paths = sorted(ROOT.glob("batch_*/*/*_bscans.tif"))
seg_paths = sorted(ROOT.glob("batch_*/*/*_depth_segmentation.tif"))

print(f"Found {len(bscan_paths)} bscan files and {len(seg_paths)} segmentation files.")

dt = OCTPuzzleDataset(
    bscan_paths=bscan_paths[:1],
    seg_paths=seg_paths[:1],
    dense_size=30,
    min_num_batches=4,
    max_num_batches=4,
    min_batch_size=3,
    max_batch_size=3,
    crop_l=220,
    seed=0,
)

sample = dt[0]

print(sample)
print("x:", sample.x.shape)
print("patches:", sample.patches.shape)
print("scan_indices:", sample.scan_indices.shape)
print("scan_offsets:", sample.scan_offsets.shape)
print("batch_scan_xy:", sample.batch_scan_xy.shape)
print("crop_windows:", sample.crop_windows)

for node_i in range(sample.patches.shape[0]):
    fig, axes = plt.subplots(1, sample.patches.shape[1], figsize=(12, 4))

    for s in range(sample.patches.shape[1]):
        ax = axes[s]
        img = sample.patches[node_i, s, 0].numpy()

        scan_id = int(sample.scan_indices[node_i, s])
        xy = sample.batch_scan_xy[node_i, s].tolist()
        offset = sample.scan_offsets[node_i, s].tolist()

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"node {node_i}, scan {scan_id}\n"
            f"xy=({xy[0]:.3f},{xy[1]:.3f})\n"
            f"off=({offset[0]:.3f},{offset[1]:.3f})"
        )
        ax.axis("off")

    fig.suptitle(f"Node {node_i}: anchor xy = {sample.x[node_i].tolist()}")
    plt.tight_layout()
    plt.savefig(OUT / f"node_{node_i}.png", dpi=200)
    plt.close(fig)

print(f"Saved debug images to {OUT.resolve()}")