import argparse
import os
import sys

import pytorch_lightning as pl
import torch_geometric
import yaml
from pytorch_lightning.loggers import WandbLogger
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from dataset import dataset_utils as du
from model import spatial_diffusion as sd


def main(config_path, checkpoint_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    _, _, test_dt = du.get_dataset(cfg)
    test_loader = torch_geometric.loader.DataLoader(test_dt, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], shuffle=False)

    model = sd.GNN_Diffusion.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.initialize_torchmetrics()

    logger = WandbLogger(
        project="DiffAssemble-OCT",
        entity="jeaninecoa-ucd",
        name=f'{cfg["experiment_name"]}_test',
        offline=cfg.get("offline", True),
    )

    trainer = pl.Trainer(accelerator="gpu", devices=1, logger=logger)
    results = trainer.test(model=model, dataloaders=test_loader)

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    main(args.config, args.checkpoint)