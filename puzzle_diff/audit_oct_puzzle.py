import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from dataset import dataset_utils as du


def remove_anchor_edges(edge_index, anchor_idx):
    src, dst = edge_index

    keep = (
        (src != anchor_idx)
        & (dst != anchor_idx)
        & (src != dst)
    )

    return edge_index[:, keep]


def undirected_unique_edges(edge_index):
    edges = set()

    for src, dst in edge_index.t().tolist():
        if src == dst:
            continue

        edge = tuple(sorted((int(src), int(dst))))
        edges.add(edge)

    return sorted(edges)


def connected_components(num_nodes, edges, anchor_idx):
    adjacency = {i: set() for i in range(num_nodes) if i != anchor_idx}

    for i, j in edges:
        if i == anchor_idx or j == anchor_idx:
            continue

        adjacency[i].add(j)
        adjacency[j].add(i)

    visited = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue

        stack = [node]
        visited.add(node)
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def audit_sample(data, sample_number):
    num_nodes = data.x.shape[0]
    anchor_idx = int(data.anchor_idx.view(-1)[0])

    assert 0 <= anchor_idx < num_nodes
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.min().item() >= 0
    assert data.edge_index.max().item() < num_nodes

    assert torch.isfinite(data.x).all()
    assert torch.isfinite(data.gt_delta).all()
    assert torch.isfinite(data.gt_delta_model).all()
    assert torch.isfinite(data.rough_delta).all()
    assert torch.isfinite(data.rough_delta_model).all()
    assert torch.isfinite(data.correction_model).all()

    expected_gt_model = data.gt_delta / data.delta_scale.view(1, 2)

    assert torch.allclose(
        data.gt_delta_model,
        expected_gt_model,
        atol=1e-6,
    )

    expected_correction = (
        data.gt_delta - data.rough_delta
    )

    assert torch.allclose(
        data.correction,
        expected_correction,
        atol=1e-6,
    )

    expected_correction_model = (
        expected_correction
        / data.correction_scale.view(1, 2)
    )

    assert torch.allclose(
        data.correction_model,
        expected_correction_model,
        atol=1e-6,
    )

    assert torch.allclose(
        data.x,
        data.correction_model,
        atol=1e-6,
    )

    zero = torch.zeros(2)

    assert torch.allclose(data.gt_delta[anchor_idx], zero, atol=1e-6)
    assert torch.allclose(data.gt_delta_model[anchor_idx], zero, atol=1e-6)
    assert torch.allclose(data.rough_delta[anchor_idx], zero, atol=1e-6)
    assert torch.allclose(data.rough_delta_model[anchor_idx], zero, atol=1e-6)
    assert torch.allclose(data.correction_model[anchor_idx], zero, atol=1e-6)

    is_anchor = data.is_anchor.bool().view(-1)

    assert is_anchor.sum().item() == 1
    assert is_anchor[anchor_idx].item()

    non_anchor = ~is_anchor

    local_edge_index = remove_anchor_edges(
        data.edge_index,
        anchor_idx,
    )

    local_edges = undirected_unique_edges(local_edge_index)

    degrees = torch.zeros(num_nodes, dtype=torch.long)

    for i, j in local_edges:
        degrees[i] += 1
        degrees[j] += 1

    non_anchor_degrees = degrees[non_anchor]

    components = connected_components(
        num_nodes=num_nodes,
        edges=local_edges,
        anchor_idx=anchor_idx,
    )

    largest_component = max(
        (len(component) for component in components),
        default=0,
    )

    edge_gt_dx = []
    edge_gt_dy = []
    edge_scan_gap = []
    edge_same_batch = []
    edge_same_scan = []

    batch_ids = data.batch_ids.view(-1)
    scan_indices = data.scan_indices.view(-1)

    for i, j in local_edges:
        gt_difference = (
            data.gt_delta[i] - data.gt_delta[j]
        ).abs()

        edge_gt_dx.append(float(gt_difference[0]))
        edge_gt_dy.append(float(gt_difference[1]))
        edge_scan_gap.append(
            abs(int(scan_indices[i]) - int(scan_indices[j]))
        )
        edge_same_batch.append(
            int(batch_ids[i]) == int(batch_ids[j])
        )
        edge_same_scan.append(
            int(scan_indices[i]) == int(scan_indices[j])
        )

    correction = data.correction_model[non_anchor]
    rough_error = (
        data.rough_delta[non_anchor]
        - data.gt_delta[non_anchor]
    )

    result = {
        "sample": sample_number,
        "num_nodes": num_nodes,
        "num_batches": int(torch.unique(batch_ids).numel()),
        "num_unique_scans": int(torch.unique(scan_indices).numel()),

        "num_directed_edges": int(data.edge_index.shape[1]),
        "num_local_edges": len(local_edges),

        "mean_local_degree": float(non_anchor_degrees.float().mean()),
        "median_local_degree": float(non_anchor_degrees.float().median()),
        "min_local_degree": int(non_anchor_degrees.min()),
        "max_local_degree": int(non_anchor_degrees.max()),
        "num_zero_local_degree": int((non_anchor_degrees == 0).sum()),
        "fraction_zero_local_degree": float(
            (non_anchor_degrees == 0).float().mean()
        ),

        "num_components_without_anchor": len(components),
        "largest_component_fraction": (
            largest_component / int(non_anchor.sum())
        ),

        "correction_x_mae_model": float(correction[:, 0].abs().mean()),
        "correction_y_mae_model": float(correction[:, 1].abs().mean()),
        "correction_x_std_model": float(correction[:, 0].std()),
        "correction_y_std_model": float(correction[:, 1].std()),

        "rough_x_mae_physical": float(rough_error[:, 0].abs().mean()),
        "rough_y_mae_physical": float(rough_error[:, 1].abs().mean()),

        "edge_gt_dx_mean": (
            float(np.mean(edge_gt_dx)) if edge_gt_dx else np.nan
        ),
        "edge_gt_dy_mean": (
            float(np.mean(edge_gt_dy)) if edge_gt_dy else np.nan
        ),
        "edge_scan_gap_mean": (
            float(np.mean(edge_scan_gap)) if edge_scan_gap else np.nan
        ),
        "fraction_edges_same_batch": (
            float(np.mean(edge_same_batch)) if edge_same_batch else np.nan
        ),
        "fraction_edges_same_scan": (
            float(np.mean(edge_same_scan)) if edge_same_scan else np.nan
        ),
    }

    return result, non_anchor_degrees.tolist()


def mean_value(results, key):
    values = np.asarray(
        [result[key] for result in results],
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(values.mean())


def print_summary(results, all_degrees):
    total_non_anchor = len(all_degrees)
    zero_degree = sum(degree == 0 for degree in all_degrees)

    print("\n" + "=" * 70)
    print("OCT PUZZLE AUDIT SUMMARY")
    print("=" * 70)

    print(f"Samples audited: {len(results)}")
    print(f"Mean nodes: {mean_value(results, 'num_nodes'):.2f}")
    print(f"Mean batches: {mean_value(results, 'num_batches'):.2f}")
    print(
        f"Mean unique scans: "
        f"{mean_value(results, 'num_unique_scans'):.2f}"
    )

    print("\nGraph connectivity excluding anchor")
    print(
        f"Mean local degree: "
        f"{mean_value(results, 'mean_local_degree'):.3f}"
    )
    print(
        f"Non-anchor nodes with zero local peers: "
        f"{zero_degree}/{total_non_anchor} "
        f"({100 * zero_degree / max(total_non_anchor, 1):.2f}%)"
    )
    print(
        f"Mean connected components: "
        f"{mean_value(results, 'num_components_without_anchor'):.2f}"
    )
    print(
        f"Mean largest-component fraction: "
        f"{mean_value(results, 'largest_component_fraction'):.3f}"
    )

    print("\nResidual target distribution, normalized model coordinates")
    print(
        f"Correction X MAE: "
        f"{mean_value(results, 'correction_x_mae_model'):.6f}"
    )
    print(
        f"Correction Y MAE: "
        f"{mean_value(results, 'correction_y_mae_model'):.6f}"
    )
    print(
        f"Correction X std: "
        f"{mean_value(results, 'correction_x_std_model'):.6f}"
    )
    print(
        f"Correction Y std: "
        f"{mean_value(results, 'correction_y_std_model'):.6f}"
    )

    print("\nNo-correction baseline, physical coordinates")
    print(
        f"Rough X MAE: "
        f"{mean_value(results, 'rough_x_mae_physical'):.6f}"
    )
    print(
        f"Rough Y MAE: "
        f"{mean_value(results, 'rough_y_mae_physical'):.6f}"
    )

    print("\nLocal edge composition")
    print(
        f"Mean true X gap: "
        f"{mean_value(results, 'edge_gt_dx_mean'):.6f}"
    )
    print(
        f"Mean true Y gap: "
        f"{mean_value(results, 'edge_gt_dy_mean'):.6f}"
    )
    print(
        f"Mean scan-index gap: "
        f"{mean_value(results, 'edge_scan_gap_mean'):.3f}"
    )
    print(
        f"Fraction same batch: "
        f"{mean_value(results, 'fraction_edges_same_batch'):.3f}"
    )
    print(
        f"Fraction same scan: "
        f"{mean_value(results, 'fraction_edges_same_scan'):.3f}"
    )

    print("\nDegree distribution")
    print(dict(sorted(Counter(all_degrees).items())))


def main(config_path, num_samples, split):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    train_dataset, val_dataset, test_dataset = du.get_dataset(cfg)

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }

    dataset = datasets[split]

    if len(dataset) == 0:
        raise RuntimeError(f"{split} dataset is empty")

    num_samples = min(num_samples, len(dataset))

    results = []
    all_degrees = []

    print(
        f"\nAuditing {num_samples} samples from "
        f"{split} dataset of length {len(dataset)}"
    )

    for sample_number in range(num_samples):
        try:
            data = dataset[sample_number]

            result, degrees = audit_sample(
                data,
                sample_number,
            )

            results.append(result)
            all_degrees.extend(degrees)

            if sample_number < 5:
                print(
                    f"sample={sample_number} "
                    f"nodes={result['num_nodes']} "
                    f"local_edges={result['num_local_edges']} "
                    f"mean_degree={result['mean_local_degree']:.2f} "
                    f"zero_degree="
                    f"{100 * result['fraction_zero_local_degree']:.1f}% "
                    f"components="
                    f"{result['num_components_without_anchor']}"
                )

        except Exception as error:
            print(
                f"FAILED sample {sample_number}: "
                f"{type(error).__name__}: {error}"
            )
            raise

    print_summary(results, all_degrees)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        num_samples=args.num_samples,
        split=args.split,
    )