import tifffile as tiff
import numpy as np
from pathlib import Path

SEG_PATH = Path(
    "datasets/rl-whole-eye/processed_data/batch_000/"
    "881ca00e6c71/881ca00e6c71_006_depth_segmentation.tif"
)

# Adaptive threshold settings
MIN_ABSOLUTE_PIXELS = 3000
MIN_FRACTION_OF_PEAK = 0.10

# Optional: ignore tiny gaps inside the main region
MAX_GAP_TO_FILL = 1


def contiguous_ranges(indices):
    if len(indices) == 0:
        return []

    ranges = []
    start = prev = int(indices[0])

    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev))
            start = prev = idx

    ranges.append((start, prev))
    return ranges


def fill_small_gaps(valid_mask, max_gap=1):
    valid_mask = valid_mask.copy()
    n = len(valid_mask)

    i = 0
    while i < n:
        if valid_mask[i]:
            i += 1
            continue

        gap_start = i
        while i < n and not valid_mask[i]:
            i += 1
        gap_end = i - 1

        gap_len = gap_end - gap_start + 1

        has_valid_left = gap_start > 0 and valid_mask[gap_start - 1]
        has_valid_right = gap_end < n - 1 and valid_mask[gap_end + 1]

        if has_valid_left and has_valid_right and gap_len <= max_gap:
            valid_mask[gap_start:gap_end + 1] = True

    return valid_mask


def main():
    seg = tiff.imread(SEG_PATH)

    if seg.ndim != 3:
        raise ValueError(f"Expected segmentation stack (N, H, W), got {seg.shape}")

    n_scans = seg.shape[0]

    counts_total = np.zeros(n_scans, dtype=np.int64)
    counts_cs = np.zeros(n_scans, dtype=np.int64)
    counts_iris = np.zeros(n_scans, dtype=np.int64)

    for i in range(n_scans):
        s = seg[i]
        counts_cs[i] = np.sum(s == 1)
        counts_iris[i] = np.sum(s == 2)
        counts_total[i] = np.sum(s > 0)

    peak = int(counts_total.max())
    threshold = max(MIN_ABSOLUTE_PIXELS, int(MIN_FRACTION_OF_PEAK * peak))

    valid_mask = counts_total >= threshold
    valid_mask = fill_small_gaps(valid_mask, max_gap=MAX_GAP_TO_FILL)

    valid = np.where(valid_mask)[0]
    ranges = contiguous_ranges(valid)

    print(f"Loaded: {SEG_PATH}")
    print(f"Shape: {seg.shape}")
    print(f"Peak segmented pixels in one B-scan: {peak}")
    print(f"Adaptive threshold: {threshold}")
    print(f"Valid scans after threshold: {len(valid)} / {n_scans}")

    if len(valid) == 0:
        print("No valid scans found.")
        return

    print("\nContiguous valid ranges:")
    for a, b in ranges:
        print(f"  {a:03d} to {b:03d}  ({b - a + 1} scans)")

    best_start, best_end = max(ranges, key=lambda r: r[1] - r[0] + 1)

    print("\nBest usable scan range:")
    print(f"  inclusive: {best_start:03d} to {best_end:03d}")
    print(f"  python slice: {best_start}:{best_end + 1}")

    print("\nSuggested constants:")
    print(f"VALID_SCAN_START = {best_start}")
    print(f"VALID_SCAN_END = {best_end + 1}  # exclusive")

    print("\nPer-scan counts:")
    for i in range(n_scans):
        if counts_total[i] > 0:
            mark = "BEST" if best_start <= i <= best_end else (
                "VALID" if valid_mask[i] else "low"
            )
            print(
                f"  scan {i:03d}: "
                f"total={counts_total[i]}, "
                f"cornea/sclera={counts_cs[i]}, "
                f"iris={counts_iris[i]} "
                f"[{mark}]"
            )


if __name__ == "__main__":
    main()