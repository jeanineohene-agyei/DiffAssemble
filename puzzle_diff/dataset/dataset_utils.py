from collections import defaultdict
from pathlib import Path
import random
import numpy as np

from .puzzle_dataset import OCTPuzzleDataset


ALLOWED_DT = ["oct"]


def get_participant_id(path: Path) -> str:
    """
    Expected structure:
        root/batch_XXX/PARTICIPANT_ID/volume_bscans.tif
    """
    return path.parent.name


def get_volume_key(path: Path) -> str:
    """return the filename prefix (participant id) shared by the B-scan and segmentation files"""
    suffix = "_bscans.tif"

    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected B-scan filename: {path}")

    return path.name.removesuffix(suffix)


import numpy as np
import torch


def tensor_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def axis_stats(name, arr):
    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim == 1:
        arr = arr[:, None]

    lines = []
    lines.append(f"\n{name}")
    lines.append("-" * len(name))

    axis_names = ["x", "y"]

    for d in range(arr.shape[1]):
        values = arr[:, d]
        label = axis_names[d] if d < 2 else str(d)

        lines.append(
            f"{label}: "
            f"min={values.min():.6f}, "
            f"p01={np.percentile(values, 1):.6f}, "
            f"p05={np.percentile(values, 5):.6f}, "
            f"median={np.median(values):.6f}, "
            f"mean={values.mean():.6f}, "
            f"p95={np.percentile(values, 95):.6f}, "
            f"p99={np.percentile(values, 99):.6f}, "
            f"max={values.max():.6f}, "
            f"std={values.std():.6f}, "
            f"mean_abs={np.abs(values).mean():.6f}"
        )

    return lines


def audit_coordinate_scales(dataset, output_path="coordinate_audit.txt", num_samples=200):
    num_samples = min(num_samples, len(dataset))

    all_raw_xy = []
    all_gt_delta = []
    all_gt_delta_model = []

    all_rough_delta = []
    all_rough_delta_model = []

    all_correction = []
    all_correction_model = []

    all_rough_radius = []
    all_rough_radius_model = []

    delta_scales = []
    correction_scales = []

    node_counts = []
    batch_counts = []

    raw_x_ranges = []
    raw_y_ranges = []

    gt_x_ranges = []
    gt_y_ranges = []

    lines = []

    lines.append("OCT COORDINATE / ROUGH POSITION AUDIT")
    lines.append("=" * 80)
    lines.append(f"Samples audited: {num_samples}")

    if hasattr(dataset, "x_spacing_mm"):
        lines.append(f"x_spacing_mm: {dataset.x_spacing_mm:.6f}")

    if hasattr(dataset, "y_spacing_mm"):
        lines.append(f"y_spacing_mm: {dataset.y_spacing_mm:.6f}")

    if hasattr(dataset, "dense_size"):
        lines.append(f"dense_size: {dataset.dense_size}")

    if hasattr(dataset, "crop_l"):
        lines.append(f"crop_l: {dataset.crop_l}")

    if hasattr(dataset, "rough_radius_x"):
        lines.append(f"configured rough_radius_x: {dataset.rough_radius_x:.6f}")

    if hasattr(dataset, "rough_radius_y"):
        lines.append(f"configured rough_radius_y: {dataset.rough_radius_y:.6f}")

    for i in range(num_samples):
        data = dataset[i]

        raw_xy = tensor_np(data.raw_xy).reshape(-1, 2)
        gt_delta = tensor_np(data.gt_delta).reshape(-1, 2)
        gt_delta_model = tensor_np(data.gt_delta_model).reshape(-1, 2)

        rough_delta = tensor_np(data.rough_delta).reshape(-1, 2)
        rough_delta_model = tensor_np(data.rough_delta_model).reshape(-1, 2)

        correction = tensor_np(data.correction).reshape(-1, 2)
        correction_model = tensor_np(data.correction_model).reshape(-1, 2)

        rough_radius = tensor_np(data.rough_radius).reshape(-1, 2)
        rough_radius_model = tensor_np(data.rough_radius_model).reshape(-1, 2)

        delta_scale = tensor_np(data.delta_scale).reshape(-1, 2)[0]
        correction_scale = tensor_np(data.correction_scale).reshape(-1, 2)[0]

        all_raw_xy.append(raw_xy)
        all_gt_delta.append(gt_delta)
        all_gt_delta_model.append(gt_delta_model)

        all_rough_delta.append(rough_delta)
        all_rough_delta_model.append(rough_delta_model)

        all_correction.append(correction)
        all_correction_model.append(correction_model)

        all_rough_radius.append(rough_radius)
        all_rough_radius_model.append(rough_radius_model)

        delta_scales.append(delta_scale)
        correction_scales.append(correction_scale)

        node_counts.append(raw_xy.shape[0])
        batch_counts.append(len(np.unique(tensor_np(data.batch_ids))))

        raw_x_ranges.append(raw_xy[:, 0].max() - raw_xy[:, 0].min())
        raw_y_ranges.append(raw_xy[:, 1].max() - raw_xy[:, 1].min())

        gt_x_ranges.append(gt_delta[:, 0].max() - gt_delta[:, 0].min())
        gt_y_ranges.append(gt_delta[:, 1].max() - gt_delta[:, 1].min())

    raw_xy = np.concatenate(all_raw_xy, axis=0)
    gt_delta = np.concatenate(all_gt_delta, axis=0)
    gt_delta_model = np.concatenate(all_gt_delta_model, axis=0)

    rough_delta = np.concatenate(all_rough_delta, axis=0)
    rough_delta_model = np.concatenate(all_rough_delta_model, axis=0)

    correction = np.concatenate(all_correction, axis=0)
    correction_model = np.concatenate(all_correction_model, axis=0)

    rough_radius = np.concatenate(all_rough_radius, axis=0)
    rough_radius_model = np.concatenate(all_rough_radius_model, axis=0)

    delta_scales = np.stack(delta_scales)
    correction_scales = np.stack(correction_scales)

    lines.append("\nSAMPLE STRUCTURE")
    lines.append("-" * 80)
    lines.append(
        f"nodes/sample: min={np.min(node_counts)}, "
        f"mean={np.mean(node_counts):.2f}, "
        f"max={np.max(node_counts)}"
    )
    lines.append(
        f"batches/sample: min={np.min(batch_counts)}, "
        f"mean={np.mean(batch_counts):.2f}, "
        f"max={np.max(batch_counts)}"
    )

    lines.append("\nPER-SAMPLE RAW POSITION SPAN")
    lines.append("-" * 80)
    lines.append(
        f"x span mm: mean={np.mean(raw_x_ranges):.6f}, "
        f"median={np.median(raw_x_ranges):.6f}, "
        f"p95={np.percentile(raw_x_ranges, 95):.6f}, "
        f"max={np.max(raw_x_ranges):.6f}"
    )
    lines.append(
        f"y span mm: mean={np.mean(raw_y_ranges):.6f}, "
        f"median={np.median(raw_y_ranges):.6f}, "
        f"p95={np.percentile(raw_y_ranges, 95):.6f}, "
        f"max={np.max(raw_y_ranges):.6f}"
    )

    lines += axis_stats("RAW XY (mm)", raw_xy)
    lines += axis_stats("GT DELTA (mm, anchor-relative)", gt_delta)
    lines += axis_stats("GT DELTA MODEL (normalized)", gt_delta_model)

    lines += axis_stats("ROUGH DELTA (mm)", rough_delta)
    lines += axis_stats("ROUGH DELTA MODEL (normalized)", rough_delta_model)

    lines += axis_stats("CORRECTION = GT - ROUGH (mm)", correction)
    lines += axis_stats("CORRECTION MODEL = CORRECTION / ROUGH RADIUS", correction_model)

    lines += axis_stats("ROUGH RADIUS (mm)", rough_radius)
    lines += axis_stats("ROUGH RADIUS MODEL", rough_radius_model)

    lines += axis_stats("DELTA SCALE", delta_scales)
    lines += axis_stats("CORRECTION SCALE", correction_scales)

    lines.append("\nROUGH ERROR RELATIVE TO DATA")
    lines.append("-" * 80)

    for d, axis in enumerate(["x", "y"]):
        abs_corr = np.abs(correction[:, d])
        abs_gt = np.abs(gt_delta[:, d])

        lines.append(f"\n{axis.upper()}:")

        lines.append(
            f"  correction absolute error: "
            f"median={np.median(abs_corr):.6f} mm, "
            f"mean={np.mean(abs_corr):.6f} mm, "
            f"p95={np.percentile(abs_corr, 95):.6f} mm, "
            f"max={np.max(abs_corr):.6f} mm"
        )

        lines.append(
            f"  |gt_delta|: "
            f"median={np.median(abs_gt):.6f} mm, "
            f"mean={np.mean(abs_gt):.6f} mm, "
            f"p95={np.percentile(abs_gt, 95):.6f} mm"
        )

        nonzero_radius = rough_radius[:, d] > 0

        if nonzero_radius.any():
            ratio = abs_corr[nonzero_radius] / rough_radius[nonzero_radius, d]

            lines.append(
                f"  |correction| / rough_radius: "
                f"median={np.median(ratio):.4f}, "
                f"mean={np.mean(ratio):.4f}, "
                f"p95={np.percentile(ratio, 95):.4f}, "
                f"max={np.max(ratio):.4f}"
            )

            lines.append(
                f"  fraction > radius: {(ratio > 1.0).mean() * 100:.2f}%"
            )

    lines.append("\nNORMALIZED TARGET CHECKS")
    lines.append("-" * 80)

    for arr, name in [
        (gt_delta_model, "gt_delta_model"),
        (rough_delta_model, "rough_delta_model"),
        (correction_model, "correction_model"),
    ]:
        lines.append(
            f"{name}: "
            f"|x|>1 = {(np.abs(arr[:, 0]) > 1).mean() * 100:.2f}%, "
            f"|y|>1 = {(np.abs(arr[:, 1]) > 1).mean() * 100:.2f}%"
        )

    if hasattr(dataset, "x_spacing_mm") and hasattr(dataset, "rough_radius_x"):
        lines.append("\nRADIUS VS ACQUISITION SPACING")
        lines.append("-" * 80)

        lines.append(
            f"rough_radius_x / x_spacing = "
            f"{dataset.rough_radius_x / dataset.x_spacing_mm:.3f} lateral samples"
        )

        lines.append(
            f"rough_radius_y / y_spacing = "
            f"{dataset.rough_radius_y / dataset.y_spacing_mm:.3f} B-scan intervals"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved coordinate audit to: {output_path}")


def get_dataset(cfg):
    root = Path(cfg["oct_root"])

    bscan_paths = sorted(root.glob("batch_*/*/*_bscans.tif"))
    seg_paths = sorted(root.glob("batch_*/*/*_depth_segmentation.tif"))

    if not bscan_paths:
        raise RuntimeError(f"No OCT B-scan files found under {root}")

    if not seg_paths:
        raise RuntimeError(f"No OCT segmentation files found under {root}")

    # Pair B-scans and segmentations using their full volume path rather than
    # assuming two independently sorted lists are aligned
    seg_lookup = {}

    for seg_path in seg_paths:
        seg_suffix = "_depth_segmentation.tif"

        if not seg_path.name.endswith(seg_suffix):
            continue

        volume_name = seg_path.name.removesuffix(seg_suffix)

        # include batch and participant so identical volume names cannot collide
        key = (seg_path.parent.parent.name, seg_path.parent.name, volume_name)

        seg_lookup[key] = seg_path

    paired_volumes = []

    for bscan_path in bscan_paths:
        volume_name = get_volume_key(bscan_path)

        key = (bscan_path.parent.parent.name, bscan_path.parent.name, volume_name)

        seg_path = seg_lookup.get(key)

        if seg_path is None:
            raise FileNotFoundError(f"No matching segmentation found for:\n{bscan_path}")

        participant_id = get_participant_id(bscan_path)
        paired_volumes.append({"participant_id": participant_id, "bscan_path": bscan_path, "seg_path": seg_path})

    # group all volumes belonging to the same participant
    participant_to_volumes = defaultdict(list)

    for volume in paired_volumes:
        participant_to_volumes[volume["participant_id"]].append(volume)

    participant_ids = sorted(participant_to_volumes)

    if len(participant_ids) < 2:
        raise RuntimeError("A participant-level train/validation split requires at least " f"2 participants, but found {len(participant_ids)}.")

    # shuffle participants deterministically before splitting
    split_seed = cfg.get("split_seed", 42)
    split_rng = random.Random(split_seed)
    split_rng.shuffle(participant_ids)

    train_fraction = cfg.get("train_fraction", 0.8)
    val_fraction = cfg.get("val_fraction", 0.1)
    test_fraction = cfg.get("test_fraction", 0.1)

    train_fraction = cfg.get("train_fraction", 0.8)
    val_fraction = cfg.get("val_fraction", 0.1)
    test_fraction = cfg.get("test_fraction", 0.1)

    fraction_sum = train_fraction + val_fraction + test_fraction

    if not np.isclose(fraction_sum, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0, "f"but got {fraction_sum:.6f}.")

    num_participants = len(participant_ids)

    num_train = int(train_fraction * num_participants)
    num_val = int(val_fraction * num_participants)

    # ensure each split contains at least one participant
    num_train = max(1, num_train)
    num_val = max(1, num_val)

    # leave at least one participant for test
    if num_train + num_val >= num_participants:
        num_train = num_participants - 2
        num_val = 1

    num_test = num_participants - num_train - num_val

    if num_test < 1:
        raise RuntimeError("The requested split does not leave any participants for testing.")

    train_participant_ids = participant_ids[:num_train]
    val_participant_ids = participant_ids[num_train:num_train + num_val]
    test_participant_ids = participant_ids[num_train + num_val:]

    train_participants = set(train_participant_ids)
    val_participants = set(val_participant_ids)
    test_participants = set(test_participant_ids)

    train_volumes = [volume for participant_id in train_participant_ids for volume in participant_to_volumes[participant_id]]
    val_volumes = [volume for participant_id in val_participant_ids for volume in participant_to_volumes[participant_id]]
    test_volumes = [volume for participant_id in test_participant_ids for volume in participant_to_volumes[participant_id]]

    # debugging subset
    n = cfg.get("train_num_volumes")

    if n is not None:
        train_volumes = train_volumes[:n]

    train_bscan_paths = [v["bscan_path"] for v in train_volumes]
    train_seg_paths = [v["seg_path"] for v in train_volumes]

    val_bscan_paths = [v["bscan_path"] for v in val_volumes]
    val_seg_paths = [v["seg_path"] for v in val_volumes]

    test_bscan_paths = [v["bscan_path"] for v in test_volumes]
    test_seg_paths = [v["seg_path"] for v in test_volumes]

    # verify no participant leakage
    assert train_participants.isdisjoint(val_participants)
    assert train_participants.isdisjoint(test_participants)
    assert val_participants.isdisjoint(test_participants)

    print(f"Found {len(paired_volumes)} total volumes under {root}")
    print(f"Found {len(participant_ids)} total participants")

    print(f"Training: {len(train_participants)} participants, "f"{len(train_bscan_paths)} volumes")
    print(f"Validation: {len(val_participants)} participants, "f"{len(val_bscan_paths)} volumes")
    print(f"Testing: {len(test_participants)} participants, "f"{len(test_bscan_paths)} volumes")
    print("Participant overlap: "f"train/val={train_participants & val_participants}, "f"train/test={train_participants & test_participants}, "f"val/test={val_participants & test_participants}")

    common_dataset_args = {
        "dense_size": cfg["dense_size"],
        "crop_l": cfg["crop_l"],
        "min_num_batches": cfg["min_num_batches"],
        "max_num_batches": cfg["max_num_batches"],
        "min_batch_size": cfg["min_batch_size"],
        "max_batch_size": cfg["max_batch_size"],
        "min_unique_scans": cfg["min_unique_scans"],
        "min_total_nodes": cfg["min_total_nodes"],
        "max_total_nodes": cfg["max_total_nodes"],
    }

    train_dt = OCTPuzzleDataset(
        bscan_paths=train_bscan_paths,
        seg_paths=train_seg_paths,
        seed=42,
        randomize_samples=cfg["randomize_samples"],
        samples_per_volume=cfg["samples_per_volume"],
        **common_dataset_args,
    )

    val_dt = OCTPuzzleDataset(
        bscan_paths=val_bscan_paths,
        seg_paths=val_seg_paths,
        seed=100000,
        randomize_samples=False,
        samples_per_volume=cfg.get("val_samples_per_volume", 1),
        **common_dataset_args,
    )

    test_dt = OCTPuzzleDataset(
        bscan_paths=test_bscan_paths,
        seg_paths=test_seg_paths,
        seed=200000,
        randomize_samples=False,
        samples_per_volume=cfg.get("test_samples_per_volume", 1),
        **common_dataset_args,
    )

    return train_dt, val_dt, test_dt