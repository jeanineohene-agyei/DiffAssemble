import argparse
import os
import random
import string
import sys

import pytorch_lightning as pl
import torch
import torch_geometric
import wandb
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

torch.cuda.empty_cache()

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from dataset import dataset_utils as du
from model import spatial_diffusion as sd


# def get_random_string(length):
#     # choose from all lowercase letter
#     letters = string.ascii_lowercase
#     result_str = "".join(random.choice(letters) for i in range(length))
#     return result_str  # print("Random string of length", length, "is:", result_str)


# class Percent(object):
#     def __new__(self, percent_string):
#         if percent_string.endswith("%"):
#             return str(percent_string)
#         else:
#             return int(percent_string)


def main(**cfg):
    train_dt, val_dt, _ = du.get_dataset(cfg)

    dl_train = torch_geometric.loader.DataLoader(
        train_dt,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
        persistent_workers=True,
    )

    dl_val = torch_geometric.loader.DataLoader(
        val_dt,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=False,
    )

    model = sd.GNN_Diffusion(
        steps=cfg["steps"],
        sampling=cfg["sampling"],
        include_dense_vis=cfg.get("include_dense_vis", False),
        dense_vis_epochs=cfg.get("dense_vis_epochs", 0),
        backbone_learning_rate=float(cfg["backbone_learning_rate"]),
        head_learning_rate=float(cfg["head_learning_rate"]),
        backbone_weight_decay=float(cfg.get("backbone_weight_decay", 0.0)),
        head_weight_decay=float(cfg.get("head_weight_decay", 1e-4)),
        inference_ratio=cfg["inference_ratio"],
        classifier_free_w=cfg["classifier_free_w"],
        classifier_free_prob=cfg["classifier_free_prob"],
        noise_weight=cfg["noise_weight"],
        rotation=False,
        model_mean_type=sd.ModelMeanType.START_X
        if cfg["predict_xstart"]
        else sd.ModelMeanType.EPSILON,
        visual_pretrained=cfg["visual_pretrained"],
        freeze_backbone=cfg["freeze_backbone"],
        backbone=cfg["backbone"],
        architecture=cfg["architecture"],
        all_equivariant=False,
    )

    model.initialize_torchmetrics([(1, 1)])

    wandb_logger = WandbLogger(
        project="DiffAssemble-OCT",
        entity="jeaninecoa-ucd",
        offline=cfg["offline"],
        name=cfg["experiment_name"],
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_mean_dist",
        mode="min",
        save_top_k=2,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg["gpus"],
        strategy="ddp" if cfg["gpus"] > 1 else None,
        accumulate_grad_batches=cfg["acc_grad"] if cfg["acc_grad"] > 0 else None,
        check_val_every_n_epoch=2,
        logger=wandb_logger,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
        max_epochs=cfg["max_epochs"],
        max_steps=cfg.get("max_steps", -1),
        # Validate every N training batches.
        val_check_interval=cfg.get("val_check_interval", 1000),
        # limit_val_batches=cfg.get("limit_val_batches", 1.0),
    )

    trainer.fit(model, dl_train, dl_val, ckpt_path=cfg["checkpoint_path"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(**cfg)