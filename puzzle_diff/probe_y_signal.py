import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from dataset import dataset_utils as du
from model.backbones import Eff_GAT


class CoordinateProbe(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Start by predicting zero correction.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features):
        return self.head(features).squeeze(-1)


@torch.no_grad()
def cache_features(
    dataset,
    feature_model,
    device,
    axis,
    split_name,
):
    loader = torch_geometric.loader.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    feature_list = []
    target_list = []

    feature_model.eval()

    for batch_idx, batch in enumerate(loader):
        print(
            f"\rCaching {split_name} "
            f"{batch_idx + 1}/{len(loader)}",
            end="",
            flush=True,
        )

        batch = batch.to(device)

        features = feature_model.visual_features(batch.patches)

        non_anchor = ~batch.is_anchor.bool().view(-1)

        features = features[non_anchor]
        targets = batch.correction_model[non_anchor, axis]

        feature_list.append(features.cpu())
        target_list.append(targets.cpu())

    print()

    if not feature_list:
        raise RuntimeError(
            f"No usable nodes found in the {split_name} dataset."
        )

    features = torch.cat(feature_list, dim=0)
    targets = torch.cat(target_list, dim=0)

    print(
        f"{split_name}: "
        f"{features.shape[0]} nodes, "
        f"{features.shape[1]} features"
    )

    return features, targets


@torch.no_grad()
def evaluate(
    probe,
    loader,
    device,
):
    probe.eval()

    total_abs_error = 0.0
    total_squared_error = 0.0
    total_baseline_abs_error = 0.0
    total_count = 0

    predictions_all = []
    targets_all = []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        predictions = probe(features)

        difference = predictions - targets

        total_abs_error += difference.abs().sum().item()
        total_squared_error += (difference ** 2).sum().item()

        # Zero prediction means keep the rough position unchanged.
        total_baseline_abs_error += targets.abs().sum().item()

        total_count += targets.numel()

        predictions_all.append(predictions.cpu())
        targets_all.append(targets.cpu())

    mae = total_abs_error / max(total_count, 1)
    rmse = (
        total_squared_error / max(total_count, 1)
    ) ** 0.5
    baseline_mae = (
        total_baseline_abs_error / max(total_count, 1)
    )

    predictions_all = torch.cat(predictions_all)
    targets_all = torch.cat(targets_all)

    prediction_std = predictions_all.std().item()
    target_std = targets_all.std().item()

    return {
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "prediction_std": prediction_std,
        "target_std": target_std,
    }


def main(
    config_path,
    epochs,
    axis_name,
    train_samples,
    val_samples,
    probe_batch_size,
    learning_rate,
):
    with open(config_path, "r") as file:
        cfg = yaml.safe_load(file)

    train_dataset, val_dataset, _ = du.get_dataset(cfg)

    train_count = min(train_samples, len(train_dataset))
    val_count = min(val_samples, len(val_dataset))

    train_dataset = Subset(
        train_dataset,
        range(train_count),
    )

    val_dataset = Subset(
        val_dataset,
        range(val_count),
    )

    print(f"Probe train samples: {len(train_dataset)}")
    print(f"Probe validation samples: {len(val_dataset)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    axis = {
        "x": 0,
        "y": 1,
    }[axis_name]

    feature_model = Eff_GAT(
        steps=1,
        model="efficientnet_b0",
        visual_pretrained=True,
        freeze_backbone=True,
        architecture="transformer",
    ).to(device)

    feature_model.eval()

    for parameter in feature_model.parameters():
        parameter.requires_grad = False

    train_features, train_targets = cache_features(
        dataset=train_dataset,
        feature_model=feature_model,
        device=device,
        axis=axis,
        split_name="train",
    )

    val_features, val_targets = cache_features(
        dataset=val_dataset,
        feature_model=feature_model,
        device=device,
        axis=axis,
        split_name="validation",
    )

    # Normalize features using training-set statistics only.
    feature_mean = train_features.mean(
        dim=0,
        keepdim=True,
    )

    feature_std = train_features.std(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-6)

    train_features = (
        train_features - feature_mean
    ) / feature_std

    val_features = (
        val_features - feature_mean
    ) / feature_std

    assert torch.isfinite(train_features).all()
    assert torch.isfinite(val_features).all()
    assert torch.isfinite(train_targets).all()
    assert torch.isfinite(val_targets).all()

    train_tensor_dataset = TensorDataset(
        train_features,
        train_targets,
    )

    val_tensor_dataset = TensorDataset(
        val_features,
        val_targets,
    )

    train_loader = DataLoader(
        train_tensor_dataset,
        batch_size=probe_batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_tensor_dataset,
        batch_size=probe_batch_size,
        shuffle=False,
        num_workers=0,
    )

    feature_dim = train_features.shape[1]

    probe = CoordinateProbe(
        feature_dim=feature_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        probe.parameters(),
        lr=learning_rate,
    )

    initial_train = evaluate(
        probe,
        train_loader,
        device,
    )

    initial_val = evaluate(
        probe,
        val_loader,
        device,
    )

    print("\nInitial zero-output probe")
    print(
        f"train_{axis_name}_mae="
        f"{initial_train['mae']:.4f} "
        f"baseline="
        f"{initial_train['baseline_mae']:.4f}"
    )
    print(
        f"val_{axis_name}_mae="
        f"{initial_val['mae']:.4f} "
        f"baseline="
        f"{initial_val['baseline_mae']:.4f}"
    )
    print()

    for epoch in range(epochs):
        probe.train()

        total_loss = 0.0
        total_abs_error = 0.0
        total_count = 0

        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            predictions = probe(features)

            loss = F.smooth_l1_loss(
                predictions,
                targets,
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                probe.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            total_loss += loss.item() * targets.numel()

            total_abs_error += (
                predictions.detach() - targets
            ).abs().sum().item()

            total_count += targets.numel()

        train_loss = total_loss / max(total_count, 1)
        train_mae = total_abs_error / max(total_count, 1)

        val_metrics = evaluate(
            probe,
            val_loader,
            device,
        )

        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"train_{axis_name}_mae={train_mae:.4f} "
            f"val_{axis_name}_mae="
            f"{val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"zero_baseline="
            f"{val_metrics['baseline_mae']:.4f} "
            f"pred_std="
            f"{val_metrics['prediction_std']:.4f} "
            f"target_std="
            f"{val_metrics['target_std']:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--axis",
        choices=["x", "y"],
        default="y",
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        epochs=args.epochs,
        axis_name=args.axis,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        probe_batch_size=args.probe_batch_size,
        learning_rate=args.learning_rate,
    )