from collections import defaultdict
from pathlib import Path
import random

from .puzzle_dataset import OCTPuzzleDataset


ALLOWED_DT = ["oct"]


def get_participant_id(path: Path) -> str:
    """
    Expected structure:
        root/batch_XXX/PARTICIPANT_ID/volume_bscans.tif
    """
    return path.parent.name


def get_volume_key(path: Path) -> str:
    """Return the filename prefix shared by the B-scan and segmentation files."""
    suffix = "_bscans.tif"

    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected B-scan filename: {path}")

    return path.name.removesuffix(suffix)


def get_dataset(cfg):
    root = Path(cfg["oct_root"])

    bscan_paths = sorted(root.glob("batch_*/*/*_bscans.tif"))
    seg_paths = sorted(root.glob("batch_*/*/*_depth_segmentation.tif"))

    if not bscan_paths:
        raise RuntimeError(f"No OCT B-scan files found under {root}")

    if not seg_paths:
        raise RuntimeError(f"No OCT segmentation files found under {root}")

    # Pair B-scans and segmentations using their full volume path rather than
    # assuming two independently sorted lists are aligned.
    seg_lookup = {}

    for seg_path in seg_paths:
        seg_suffix = "_depth_segmentation.tif"

        if not seg_path.name.endswith(seg_suffix):
            continue

        volume_name = seg_path.name.removesuffix(seg_suffix)

        # Include batch and participant so identical volume names cannot collide.
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

    # Group all volumes belonging to the same participant.
    participant_to_volumes = defaultdict(list)

    for volume in paired_volumes:
        participant_to_volumes[volume["participant_id"]].append(volume)

    participant_ids = sorted(participant_to_volumes)

    if len(participant_ids) < 2:
        raise RuntimeError("A participant-level train/validation split requires at least " f"2 participants, but found {len(participant_ids)}.")

    # Shuffle participants deterministically before splitting.
    split_seed = cfg.get("split_seed", 42)
    split_rng = random.Random(split_seed)
    split_rng.shuffle(participant_ids)

    train_fraction = cfg.get("train_fraction", 0.9)

    num_train_participants = int(train_fraction * len(participant_ids))
    num_train_participants = max(1, min(num_train_participants, len(participant_ids) - 1))

    train_participants = set(participant_ids[:num_train_participants])
    val_participants = set(participant_ids[num_train_participants:])

    train_volumes = [volume for participant_id in participant_ids if participant_id in train_participants for volume in participant_to_volumes[participant_id]]

    val_volumes = [volume for participant_id in participant_ids if participant_id in val_participants for volume in participant_to_volumes[participant_id]]

    # Optional debugging subset.
    # Apply it after the participant split so it cannot affect participant
    # separation.
    n = cfg.get("train_num_volumes")

    if n is not None:
        train_volumes = train_volumes[:n]

    train_bscan_paths = [v["bscan_path"] for v in train_volumes]
    train_seg_paths = [v["seg_path"] for v in train_volumes]

    val_bscan_paths = [v["bscan_path"] for v in val_volumes]
    val_seg_paths = [v["seg_path"] for v in val_volumes]

    participant_overlap = train_participants.intersection(val_participants)
    assert not participant_overlap, (f"Participant leakage detected: {sorted(participant_overlap)}")

    print(f"Found {len(paired_volumes)} total volumes under {root}")
    print(f"Found {len(participant_ids)} total participants")
    print(f"Training: {len(train_participants)} participants, " f"{len(train_bscan_paths)} volumes")
    print(
        f"Validation: {len(val_participants)} participants, "
        f"{len(val_bscan_paths)} volumes")
    print(f"Participant overlap: {participant_overlap}")

    train_dt = OCTPuzzleDataset(
        bscan_paths=train_bscan_paths,
        seg_paths=train_seg_paths,
        dense_size=cfg["dense_size"],
        crop_l=cfg["crop_l"],
        min_num_batches=cfg["min_num_batches"],
        max_num_batches=cfg["max_num_batches"],
        min_batch_size=cfg["min_batch_size"],
        max_batch_size=cfg["max_batch_size"],
        degree=cfg["degree"],
        seed=42,
        randomize_samples=cfg["randomize_samples"],
        samples_per_volume=cfg["samples_per_volume"],
    )

    val_dt = OCTPuzzleDataset(
        bscan_paths=val_bscan_paths,
        seg_paths=val_seg_paths,
        dense_size=cfg["dense_size"],
        crop_l=cfg["crop_l"],
        min_num_batches=cfg["min_num_batches"],
        max_num_batches=cfg["max_num_batches"],
        min_batch_size=cfg["min_batch_size"],
        max_batch_size=cfg["max_batch_size"],
        degree=cfg["degree"],
        seed=100000,
        randomize_samples=False,
        samples_per_volume=1,
    )

    return train_dt, val_dt