from pathlib import Path

from .puzzle_dataset import (OCTPuzzleDataset)

ALLOWED_DT = ["oct"]

from pathlib import Path
from .puzzle_dataset import OCTPuzzleDataset


def get_dataset(cfg):
    root = Path(cfg["oct_root"])

    bscan_paths = sorted(root.glob("batch_*/*/*_bscans.tif"))
    seg_paths = sorted(root.glob("batch_*/*/*_depth_segmentation.tif"))

    assert len(bscan_paths) == len(seg_paths), (len(bscan_paths), len(seg_paths))
    assert len(bscan_paths) > 0, f"No OCT files found under {root}"

    n = cfg.get("train_num_volumes", None)
    if n is not None:
        bscan_paths = bscan_paths[:n]
        seg_paths = seg_paths[:n]
        
    print("Found {} volumes under {}".format(len(bscan_paths), root))

    split = max(1, int(0.9 * len(bscan_paths)))

    train_dt = OCTPuzzleDataset(
        bscan_paths=bscan_paths[:split],
        seg_paths=seg_paths[:split],
        dense_size=cfg["dense_size"],
        crop_l=cfg["crop_l"],
        min_num_batches=cfg["min_num_batches"],
        max_num_batches=cfg["max_num_batches"],
        min_batch_size=cfg["min_batch_size"],
        max_batch_size=cfg["max_batch_size"],
        degree=cfg["degree"],
        seed=42,
    )

    test_dt = OCTPuzzleDataset(
        bscan_paths=bscan_paths[split:],
        seg_paths=seg_paths[split:],
        dense_size=cfg["dense_size"],
        crop_l=cfg["crop_l"],
        min_num_batches=cfg["min_num_batches"],
        max_num_batches=cfg["max_num_batches"],
        min_batch_size=cfg["min_batch_size"],
        max_batch_size=cfg["max_batch_size"],
        degree=cfg["degree"],
        seed=100000,
    )

    return train_dt, test_dt
