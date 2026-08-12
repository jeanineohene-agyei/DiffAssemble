import enum

import logging

from functools import partial
from pathlib import Path
from typing import Any

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import Tensor
from tqdm import tqdm

import wandb

from .backbones import Eff_GAT


matplotlib.use("agg")


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelScheduler(enum.Enum):
    """
    Which type of output the model predicts.
    """

    LINEAR = enum.auto()  # the model predicts x_{t-1}
    COSINE = enum.auto()  # the model predicts x_0
    COSINE_DISCRETE = enum.auto()  # the model predicts epsilon


def cosine_discrete_beta_schedule(timesteps, s=0.08):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = lambda t: torch.cos(((t / timesteps) + s) / (1 + s) + np.pi / 2)
    betas = 1 - alphas_cumprod(t + 1) / alphas_cumprod(t)
    return torch.clip(betas, 0.0001, 0.9999)


def cosine_beta_schedule(timesteps, s=0.08):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape=None):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out[:, None]  # out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


@torch.jit.script
def greedy_cost_assignment(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    # Compute pairwise distances between positions
    dist = torch.norm(pos1[:, None] - pos2, dim=2)

    # Create a tensor to store the assignments
    assignments = torch.zeros(dist.size(0), 3, dtype=torch.int64)

    # Create a mask to keep track of assigned positions
    mask = torch.ones_like(dist, dtype=torch.bool)

    # Counter for keeping track of the number of assignments
    counter = 0

    # While there are still unassigned positions
    while mask.sum() > 0:
        # Find the minimum distance
        min_val, min_idx = dist[mask].min(dim=0)

        # Get the indices of the two dimensions
        idx = int(min_idx.item())
        ret = mask.nonzero()[idx, :]
        i = ret[0]
        j = ret[1]

        # Add the assignment to the tensor
        assignments[counter, 0] = i
        assignments[counter, 1] = j
        assignments[counter, 2] = min_val

        # Increase the counter
        counter += 1

        # Remove the assigned positions from the distance matrix and the mask
        mask[i, :] = 0
        mask[:, j] = 0

    return assignments[:counter]


class GNN_Diffusion(pl.LightningModule):
    def __init__(
        self,
        steps=600,
        inference_ratio=1,
        sampling="DDPM",
        include_dense_vis=False,
        dense_vis_epochs=0,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-4,
        backbone_weight_decay=0.0,
        head_weight_decay=1e-4,
        save_and_sample_every=1000,
        bb=None,
        classifier_free_prob=0,
        classifier_free_w=0,
        noise_weight=0.0,
        rotation=False,
        model_mean_type: ModelMeanType = ModelMeanType.EPSILON,
        input_channels=2,
        output_channels=2,
        scheduler: ModelScheduler = ModelScheduler.LINEAR,
        visual_pretrained: bool = True,
        freeze_backbone: bool = True,
        backbone: str = "efficientnet_b0",
        n_layers: int = 4,
        architecture: str = "transformer",
        virt_nodes: int = 4,
        all_equivariant=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.visual_pretrained = visual_pretrained
        self.free_backbone = freeze_backbone
        
        self.include_dense_vis = include_dense_vis
        self.dense_vis_epochs = dense_vis_epochs

        self.model_mean_type = model_mean_type
        self.backbone_learning_rate = backbone_learning_rate
        self.head_learning_rate = head_learning_rate
        self.backbone_weight_decay = backbone_weight_decay
        self.head_weight_decay = head_weight_decay
        self.save_and_sample_every = save_and_sample_every
        self.classifier_free_prob = classifier_free_prob
        self.classifier_free_w = classifier_free_w
        self.noise_weight = noise_weight
        self.rotation = rotation

        self.virt_nodes = virt_nodes
        self.all_equivariant = all_equivariant
        self.save_eval_images = False
        ### DIFFUSION STUFF

        if sampling == "DDPM":
            self.inference_ratio = inference_ratio

            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddpm,
            )
            self.eta = 1
        elif sampling == "DDIM":
            self.inference_ratio = inference_ratio
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddim,
            )
            self.eta = 0

        # define beta schedule

        betas = {
            ModelScheduler.LINEAR: linear_beta_schedule,
            ModelScheduler.COSINE: cosine_beta_schedule,
            ModelScheduler.COSINE_DISCRETE: cosine_discrete_beta_schedule,
        }[scheduler](timesteps=steps)

        # self.timesteps = torch.arange(0, 700).flip(0)
        self.register_buffer("betas", betas)
        # self.betas = cosine_beta_schedule(timesteps=steps)
        # define alphas
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", np.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.steps = steps

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.backbone = backbone
        self.n_layers = n_layers
        self.architecture = architecture
        self.init_backbone()

        self.save_hyperparameters()

    def init_backbone(self):
        if self.rotation:
            self.model = Eff_GAT(
                steps=self.steps,
                input_channels=self.input_channels + 2,
                output_channels=self.output_channels + 2,
                all_equivariant=self.all_equivariant,
                model=self.backbone,
                architecture=self.architecture,
                n_layers=self.n_layers,
                virt_nodes=self.virt_nodes,
            )
        else:
            self.model = Eff_GAT(
                steps=self.steps,
                input_channels=self.input_channels,
                output_channels=self.output_channels,
                visual_pretrained=self.visual_pretrained,
                freeze_backbone=self.free_backbone,
                model=self.backbone,
                n_layers=self.n_layers,
                architecture=self.architecture,
                virt_nodes=self.virt_nodes,
            )
    
    def initialize_torchmetrics(self, n_patches=None):
        metric_names = ["mse", "mae", "mean_dist", "x_mae", "y_mae", "x_mse", "y_mse", "rough_mean_dist", "rough_x_mae", "rough_y_mae"]
        self.metrics = nn.ModuleDict({
            f"{split}_{name}": torchmetrics.MeanMetric()
            for split in ["val", "test"]
            for name in metric_names
        })

    def forward(self, xy_pos, time, patch_rgb, edge_index, edge_attr, batch, is_anchor=None, rough_delta=None, rough_radius=None) -> Any:
        return self.model(xy_pos, time, patch_rgb, edge_index, edge_attr, batch, is_anchor=is_anchor, rough_delta=rough_delta, rough_radius=rough_radius)


    def forward_with_feats(
        self,
        xy_pos: Tensor,
        time: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        patch_feats: Tensor,
        batch,
        is_anchor=None,
        rough_delta=None,
        rough_radius=None,
        return_attentions=False,
    ) -> Any:
        out, attentions = self.model.forward_with_feats(
            xy_pos, time, edge_index, edge_attr, patch_feats, batch, is_anchor=is_anchor, rough_delta=rough_delta, rough_radius=rough_radius
        )
        if return_attentions:
            return out, attentions
        return out

    def visual_features(self, patch_rgb):
        return self.model.visual_features(patch_rgb)
    
    # forward diffusion
    def q_sample(self, x_start, t, noise=None, is_anchor=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        if is_anchor is not None:
            anchor_mask = is_anchor.bool().view(-1, 1)
            noise = noise.clone()
            noise[anchor_mask.expand_as(noise)] = 0.0

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l1",
        cond=None,
        edge_index=None,
        edge_attr=None,
        batch=None,
        is_anchor=None,
        rough_delta=None,
        rough_radius=None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise, is_anchor=is_anchor)
        if self.steps == 1:  # Transformer case
            x_noisy = torch.zeros_like(x_noisy)

        patch_feats = self.visual_features(cond)
        
        rough_delta_cond = rough_delta
        rough_radius_cond = rough_radius

        prediction = self.forward_with_feats(
            x_noisy,
            t,
            edge_index,
            edge_attr,
            patch_feats=patch_feats,  # classifier_free_patch_feats,
            batch=batch,
            return_attentions=False,
            is_anchor=is_anchor,
            rough_delta=rough_delta_cond,
            rough_radius=rough_radius_cond,
        )

        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.model_mean_type]

        if is_anchor is not None:
            keep = ~is_anchor.bool().view(-1)
            target = target[keep]
            prediction = prediction[keep]
            
        pred_correction_magnitude = torch.norm(prediction, dim=1).mean()
        target_correction_magnitude = torch.norm(target, dim=1).mean()
            
        diff = prediction - target

        train_stats = {
            "train_x_mae_norm": diff[:, 0].abs().mean(),
            "train_y_mae_norm": diff[:, 1].abs().mean(),
            "train_x_rmse_norm": torch.sqrt((diff[:, 0] ** 2).mean()),
            "train_y_rmse_norm": torch.sqrt((diff[:, 1] ** 2).mean()),
            "train_pred_correction_mag": pred_correction_magnitude,
            "train_target_correction_mag": target_correction_magnitude,
        }

        if loss_type == "l1":
            loss = F.l1_loss(target, prediction)
        elif loss_type == "l2":
            loss = F.mse_loss(target, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(target, prediction)
        else:
            raise NotImplementedError()

        return loss, train_stats

    @torch.no_grad()
    def p_sample_ddpm(self, x, t, t_index, cond, edge_index, edge_attr, patch_feats, batch, is_anchor=None, rough_delta=None, rough_radius=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x
            - betas_t
            * self.forward_with_feats(
                x, t, edge_index, edge_attr, patch_feats=patch_feats, batch=batch, is_anchor=is_anchor, rough_delta=rough_delta, rough_radius=rough_radius
            )
            / sqrt_one_minus_alphas_cumprod_t
        )
        
        if is_anchor is not None:
            anchor_mask = is_anchor.bool().view(-1, 1)
            model_mean = model_mean.clone()
            model_mean[anchor_mask.expand_as(model_mean)] = 0.0

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            out = model_mean + torch.sqrt(posterior_variance_t) * noise

            # also keep anchor fixed after adding DDPM noise
            if is_anchor is not None:
                out = out.clone()
                out[anchor_mask.expand_as(out)] = 0.0

            return out
        
    def _get_variance_old(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = extract(
            self.alphas_cumprod, timestep
        )  # self.alphas_cumprod[timestep]

        alpha_prod_t_prev = (
            extract(self.alphas_cumprod, prev_timestep)
            if (prev_timestep >= 0).all()
            else alpha_prod_t * 0 + 1
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    @torch.no_grad()
    def p_sample_ddim(
        self, x, t, t_index, cond, edge_index, edge_attr, patch_feats, batch, is_anchor=None, rough_delta=None, rough_radius=None
    ):
        if is_anchor is not None:
            anchor_mask = is_anchor.bool().view(-1, 1)
            x = x.clone()
            x[anchor_mask.expand_as(x)] = 0.0
        else:
            anchor_mask = None

        prev_timestep = t - self.inference_ratio

        eta = self.eta
        alpha_prod = extract(self.alphas_cumprod, t, x.shape)

        if (prev_timestep >= 0).all():
            alpha_prod_prev = extract(self.alphas_cumprod, prev_timestep, x.shape)
        else:
            alpha_prod_prev = torch.ones_like(alpha_prod)

        beta = 1 - alpha_prod

        if self.classifier_free_prob > 0.0:
            model_output_cond, attentions = self.forward_with_feats(
                x,
                t,
                edge_index,
                edge_attr,
                patch_feats=patch_feats,
                batch=batch,
                return_attentions=True,
                is_anchor=is_anchor,
                rough_delta=rough_delta,
                rough_radius=rough_radius,
            )

            model_output_uncond = self.forward_with_feats(
                x,
                t,
                edge_index,
                edge_attr,
                patch_feats=torch.zeros_like(patch_feats),
                batch=batch,
                is_anchor=is_anchor,
                rough_delta=rough_delta,
                rough_radius=rough_radius,
            )

            model_output = (
                (1 + self.classifier_free_w) * model_output_cond
                - self.classifier_free_w * model_output_uncond
            )
        else:
            model_output, attentions = self.forward_with_feats(
                x,
                t,
                edge_index,
                edge_attr=edge_attr,
                patch_feats=patch_feats,
                batch=batch,
                return_attentions=True,
                is_anchor=is_anchor,
                rough_delta=rough_delta,
                rough_radius=rough_radius,
            )

        x_0 = {
            ModelMeanType.EPSILON: (
                x - torch.sqrt(beta) * model_output
            ) / torch.sqrt(alpha_prod),
            ModelMeanType.START_X: model_output,
        }[self.model_mean_type]

        if anchor_mask is not None:
            x_0 = x_0.clone()
            x_0[anchor_mask.expand_as(x_0)] = 0.0

        eps = self._predict_eps_from_xstart(x, t, x_0)

        variance = self._get_variance(t, prev_timestep)
        std_eta = eta * torch.sqrt(variance)

        pred_sample_direction = (
            torch.sqrt(1 - alpha_prod_prev - std_eta ** 2) * eps
        )

        prev_sample = (
            torch.sqrt(alpha_prod_prev) * x_0
            + pred_sample_direction
        )

        if eta > 0:
            noise = torch.randn_like(model_output)
            prev_sample = prev_sample + std_eta * noise

        if anchor_mask is not None:
            prev_sample = prev_sample.clone()
            prev_sample[anchor_mask.expand_as(prev_sample)] = 0.0

        return prev_sample, attentions

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, edge_attr, batch, is_anchor=None, rough_delta=None, rough_radius=None):
        # device = next(model.parameters()).device
        device = self.device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        
        img = torch.randn(shape, device=device) * self.noise_weight
        print("initial img std:", img.std().item(), "min:", img.min().item(), "max:", img.max().item())
        # img = einops.rearrange(
        #     img,
        #     "b c (w1 w) (h1 h) -> b (w1 h1) c w h",
        #     h1=self.patches,
        #     w1=self.patches,
        # )

        # imgs = []
        # attentions = []
        atts = None

        patch_feats = self.visual_features(cond)

        # time_t = torch.full((b,), i, device=device, dtype=torch.long)

        # time_t = torch.full((b,), 0, device=device, dtype=torch.long)

        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            img, atts = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                cond=cond,
                edge_index=edge_index,
                edge_attr=edge_attr,
                patch_feats=patch_feats,
                batch=batch,
                is_anchor=is_anchor,
                rough_delta=rough_delta,
                rough_radius=rough_radius,
            )
            
            if is_anchor is not None:
                anchor_mask = is_anchor.bool().view(-1, 1)
                img[anchor_mask.expand_as(img)] = 0.0

            # attentions.append(atts)
            # imgs.append(img)
        return img, atts

    @torch.no_grad()
    def p_sample(
        self, x, t, t_index, cond, edge_index, edge_attr, sampling_func, patch_feats, batch, is_anchor=None, rough_delta=None, rough_radius=None,
    ):
        return sampling_func(x, t, t_index, cond, edge_index, edge_attr, patch_feats, batch, is_anchor=is_anchor, rough_delta=rough_delta, rough_radius=rough_radius)

    @torch.no_grad()
    def sample(
        self,
        image_size,
        batch_size=16,
        channels=3,
        cond=None,
        edge_index=None,
        edge_attr=None,
        batch=None,
    ):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            cond=cond,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            is_anchor=batch.is_anchor,
            rough_delta=batch.rough_delta_model,
            rough_radius=batch.rough_radius_model
        )

    def configure_optimizers(self):
        backbone_params = list(self.model.visual_backbone.parameters())
        backbone_param_ids = {id(p) for p in backbone_params}

        other_params = [
            p for p in self.parameters()
            if id(p) not in backbone_param_ids
        ]

        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.backbone_learning_rate, "weight_decay": self.backbone_weight_decay},
            {"params": other_params, "lr": self.head_learning_rate, "weight_decay": self.head_weight_decay},
        ])

        return optimizer
    
    def on_fit_start(self):
        if self.global_rank == 0:
            final_alpha_bar = self.alphas_cumprod[-1]

            print("\nDiffusion terminal distribution")
            print("final alpha_bar:", final_alpha_bar.item())
            print(
                "final clean coefficient:",
                torch.sqrt(final_alpha_bar).item(),
            )
            print(
                "final noise coefficient:",
                torch.sqrt(1.0 - final_alpha_bar).item(),
            )
            print()

    def training_step(self, batch, batch_idx):
        with torch.enable_grad():
            # Temporary sanity checks
            assert batch.correction_scale.view(-1, 2).shape[1] == 2
            assert torch.isfinite(batch.x).all()
            assert torch.isfinite(batch.rough_delta).all()
            assert torch.isfinite(batch.correction_scale).all()
            
            assert batch.edge_attr.ndim == 2
            assert batch.edge_attr.shape[0] == batch.edge_index.shape[1]
            assert batch.edge_attr.shape[1] == 2
            assert torch.isfinite(batch.edge_attr).all()
        
            batch_size = batch.batch.max().item() + 1
            t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()
            new_t = torch.gather(t, 0, batch.batch)
            
            if batch_idx == 0 and self.local_rank == 0:
                print("correction_model:", batch.x[:5])
                print("gt_delta_model:", batch.gt_delta_model[:5])
                print("rough_delta_model:", batch.rough_delta_model[:5])
                print("rough_radius_model:", batch.rough_radius_model[:5])

                rough_error = batch.rough_delta_model - batch.gt_delta_model
                print("correction x MAE:", batch.x[:, 0].abs().mean().item())
                print("correction y MAE:", batch.x[:, 1].abs().mean().item())
                print("correction target magnitude:", torch.norm(batch.x, dim=1).mean().item())

            loss, train_stats = self.p_losses(
                batch.x,
                new_t,
                loss_type="huber",
                cond=batch.patches,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
                is_anchor=batch.is_anchor,
                rough_delta=batch.rough_delta_model,
                rough_radius=batch.rough_radius_model,
            )

        self.log_dict(train_stats, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, batch_size=batch_size)

        if not self.all_equivariant:
            if batch_idx == 0 and self.local_rank == 0:
                img, _ = self.p_sample_loop(
                    batch.x.shape,
                    batch.patches,
                    batch.edge_index,
                    batch.edge_attr,
                    batch=batch.batch,
                    is_anchor=batch.is_anchor,
                    rough_delta=batch.rough_delta_model,
                    rough_radius=batch.rough_radius_model,
                )

                # img = imgs[-1]
                save_path = Path(f"results/{self.logger.experiment.name}/train")

                for i in range(batch.batch.max().item() + 1):
                    idx = torch.where(batch.batch == i)[0]
                    
                    # scale = batch.delta_scale.view(-1, 2)[i]
                    # pred_correction_model = img[idx, :2]
                    # pred_position_model = (batch.rough_delta_model[idx, :2] + pred_correction_model)
                    # pred_pos = pred_position_model * scale
                    
                    correction_scale = batch.correction_scale.view(-1, 2)[i]
                    pred_correction_model = img[idx, :2]
                    pred_correction = pred_correction_model * correction_scale
                    pred_pos = batch.rough_delta[idx, :2] + pred_correction
                    
                    gt_pos = batch.gt_delta[idx, :2]

                    self.save_oct_image(
                        patches_rgb=batch.patches[idx],
                        enface_rows=batch.enface_rows[idx],
                        dense_enface=batch.dense_enface[i],
                        pos=pred_pos,
                        gt_pos=gt_pos,
                        raw_pos=batch.raw_xy[idx],
                        scan_indices=batch.scan_indices[idx],
                        batch_ids=batch.batch_ids[idx],
                        is_anchor=batch.is_anchor[idx],
                        ind_name=batch.ind_name[i],
                        file_name=save_path,
                        dense_imgs=batch.dense_imgs[i],
                        dense_scan_indices=batch.dense_scan_indices[i],
                        dense_y=batch.dense_y[i],
                        xlim=batch.xlim.view(-1, 2)[i],
                        ylim=batch.ylim.view(-1, 2)[i],
                        full_width=batch.full_width[i],
                        crop_display_width=batch.crop_display_width[i],
                        scan_spacing=batch.scan_spacing[i],
                        title_prefix="train",
                    )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size,)
        return loss

    @torch.no_grad()
    def prediction_step(self, batch, batch_idx):
        indexes = self.p_sample_loop(
            batch.x.shape, batch.patches, batch.edge_index, batch.edge_attr, batch=batch.batch, is_anchor=batch.is_anchor, rough_delta=batch.rough_delta_model, rough_radius=batch.rough_radius_model
        )
        return indexes
    
    @torch.no_grad()
    def shared_eval_step(self, batch, batch_idx, split):
        devices = [self.device.index] if self.device.type == "cuda" else []

        with torch.random.fork_rng(devices=devices):
            seed_offset = 100000 if split == "val" else 200000
            seed = seed_offset + batch_idx

            torch.manual_seed(seed)

            if self.device.type == "cuda":
                torch.cuda.manual_seed(seed)

            img, _ = self.p_sample_loop(
                batch.x.shape,
                batch.patches,
                batch.edge_index,
                batch.edge_attr,
                batch=batch.batch,
                is_anchor=batch.is_anchor,
                rough_delta=batch.rough_delta_model,
                rough_radius=batch.rough_radius_model,
            )

        # img = imgs[-1]

        # for i in range(int(batch.batch.max().item()) + 1):
        for i in range(min(batch.batch.max().item() + 1, 4)):
            idx = torch.where(batch.batch == i)[0]

            scale = batch.delta_scale.view(-1, 2)[i]

            # Convert model-space values back to original coordinate units.
            # pred_correction_model = img[idx, :2]
            # pred_position_model = (batch.rough_delta_model[idx, :2] + pred_correction_model)
            # pred_pos = pred_position_model * scale
            # gt_pos = batch.gt_delta[idx, :2]
            # rough_pos = batch.rough_delta_model[idx, :2] * scale
            
            correction_scale = batch.correction_scale.view(-1, 2)[i]
            pred_correction_model = img[idx, :2]
            pred_correction = pred_correction_model * correction_scale
            pred_pos = batch.rough_delta[idx, :2] + pred_correction
            gt_pos = batch.gt_delta[idx, :2]
            rough_pos = batch.rough_delta[idx, :2]

            non_anchor = ~batch.is_anchor[idx].bool().view(-1)

            pred_eval = pred_pos[non_anchor]
            gt_eval = gt_pos[non_anchor]
            rough_eval = rough_pos[non_anchor]

            if pred_eval.numel() == 0:
                continue

            diff = pred_eval - gt_eval
            rough_diff = rough_eval - gt_eval

            num_nodes = diff.shape[0]
            num_values = diff.numel()

            mse = torch.mean(diff ** 2)
            mae = torch.mean(torch.abs(diff))
            mean_dist = torch.norm(diff, dim=1).mean()

            x_mae = torch.abs(diff[:, 0]).mean()
            y_mae = torch.abs(diff[:, 1]).mean()

            x_mse = torch.mean(diff[:, 0] ** 2)
            y_mse = torch.mean(diff[:, 1] ** 2)

            rough_mean_dist = torch.norm(rough_diff, dim=1).mean()
            rough_x_mae = torch.abs(rough_diff[:, 0]).mean()
            rough_y_mae = torch.abs(rough_diff[:, 1]).mean()

            self.metrics[f"{split}_mse"].update(mse, weight=num_values)
            self.metrics[f"{split}_mae"].update(mae, weight=num_values)
            self.metrics[f"{split}_mean_dist"].update(mean_dist, weight=num_nodes)

            self.metrics[f"{split}_x_mae"].update(x_mae, weight=num_nodes)
            self.metrics[f"{split}_y_mae"].update(y_mae, weight=num_nodes)
            self.metrics[f"{split}_x_mse"].update(x_mse, weight=num_nodes)
            self.metrics[f"{split}_y_mse"].update(y_mse, weight=num_nodes)

            self.metrics[f"{split}_rough_mean_dist"].update(rough_mean_dist, weight=num_nodes)
            self.metrics[f"{split}_rough_x_mae"].update(rough_x_mae, weight=num_nodes)
            self.metrics[f"{split}_rough_y_mae"].update(rough_y_mae, weight=num_nodes)

            # Save only a limited number of fixed examples.
            if self.global_rank == 0 and batch_idx < 4 and i < 2:
                save_path = Path(f"results/{self.logger.experiment.name}/{split}")

                self.save_oct_image(
                    patches_rgb=batch.patches[idx],
                    enface_rows=batch.enface_rows[idx],
                    dense_enface=batch.dense_enface[i],
                    pos=pred_pos,
                    gt_pos=gt_pos,
                    raw_pos=batch.raw_xy[idx, :2],
                    scan_indices=batch.scan_indices[idx],
                    batch_ids=batch.batch_ids[idx],
                    is_anchor=batch.is_anchor[idx],
                    ind_name=batch.ind_name[i],
                    file_name=save_path,
                    dense_imgs=batch.dense_imgs[i],
                    dense_scan_indices=batch.dense_scan_indices[i],
                    dense_y=batch.dense_y[i],
                    xlim=batch.xlim.view(-1, 2)[i],
                    ylim=batch.ylim.view(-1, 2)[i],
                    full_width=batch.full_width[i],
                    crop_display_width=batch.crop_display_width[i],
                    scan_spacing=batch.scan_spacing[i],
                    title_prefix=split,
                )
            
    def validation_step(self, batch, batch_idx):
        self.shared_eval_step(batch, batch_idx, split="val")
        
        
    def shared_eval_epoch_end(self, split):
        mse = self.metrics[f"{split}_mse"].compute()
        mae = self.metrics[f"{split}_mae"].compute()
        mean_dist = self.metrics[f"{split}_mean_dist"].compute()

        x_mae = self.metrics[f"{split}_x_mae"].compute()
        y_mae = self.metrics[f"{split}_y_mae"].compute()

        x_rmse = torch.sqrt(self.metrics[f"{split}_x_mse"].compute())
        y_rmse = torch.sqrt(self.metrics[f"{split}_y_mse"].compute())

        rough_mean_dist = self.metrics[f"{split}_rough_mean_dist"].compute()
        rough_x_mae = self.metrics[f"{split}_rough_x_mae"].compute()
        rough_y_mae = self.metrics[f"{split}_rough_y_mae"].compute()

        rmse = torch.sqrt(mse)
        improvement = rough_mean_dist - mean_dist
        relative_improvement = improvement / rough_mean_dist.clamp_min(1e-8)

        self.log(f"{split}_mse", mse, sync_dist=True)
        self.log(f"{split}_rmse", rmse, sync_dist=True)
        self.log(f"{split}_mae", mae, sync_dist=True)
        self.log(f"{split}_mean_dist", mean_dist, prog_bar=True, sync_dist=True)

        self.log(f"{split}_x_mae", x_mae, sync_dist=True)
        self.log(f"{split}_y_mae", y_mae, sync_dist=True)
        self.log(f"{split}_x_rmse", x_rmse, sync_dist=True)
        self.log(f"{split}_y_rmse", y_rmse, sync_dist=True)

        self.log(f"{split}_rough_mean_dist", rough_mean_dist, sync_dist=True)
        self.log(f"{split}_rough_x_mae", rough_x_mae, sync_dist=True)
        self.log(f"{split}_rough_y_mae", rough_y_mae, sync_dist=True)

        self.log(f"{split}_mean_dist_improvement", improvement, sync_dist=True)
        self.log(f"{split}_relative_improvement", relative_improvement, sync_dist=True)

        for name, metric in self.metrics.items():
            if name.startswith(f"{split}_"):
                metric.reset()
    
    def validation_epoch_end(self, outputs):
        self.shared_eval_epoch_end("val")

    def test_epoch_end(self, outputs) -> None:
        self.shared_eval_epoch_end("test")

    def test_step(self, batch, batch_idx, *args, **kwargs):
        self.shared_eval_step(batch, batch_idx, split="test")

    def on_predict_epoch_start(self):
        logging.info(f"Saving to results/{self.logger.experiment.name}/preds")

    def predict_step(self, batch, batch_idx):
        with torch.no_grad():
            preds = self.p_sample_loop(
                batch.x.shape, batch.patches, batch.edge_index, batch.edge_attr, batch=batch.batch, is_anchor=batch.is_anchor, rough_delta=batch.rough_delta_model, rough_radius=batch.rough_radius_model
            )

            for i in range(batch.batch.max() + 1):
                for k, img in enumerate(preds):
                    idx = torch.where(batch.batch == i)[0]
                    patches_rgb = batch.patches[idx]
                    gt_pos = batch.x[idx, :2]
                    pos = img[idx, :2]
                    n_patches = batch.patches_dim[i].tolist()
                    i_name = batch.ind_name[i]

                    y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                    x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                    real_grid = einops.rearrange(xy, "x y c-> (x y) c")

                    gt_ass = greedy_cost_assignment(gt_pos, real_grid)
                    sort_idx = torch.sort(gt_ass[:, 0])[1]
                    gt_ass = gt_ass[sort_idx]

                    pred_ass = greedy_cost_assignment(pos, real_grid)
                    sort_idx = torch.sort(pred_ass[:, 0])[1]
                    pred_ass = pred_ass[sort_idx]
                    i_name = f"{batch.ind_name[i]:03d}_{k:03d}"
                    save_path = Path(f"results/{self.logger.experiment.name}/val")
                    correct = (gt_ass[:, 1] == pred_ass[:, 1]).all()
                    self.save_image(
                        patches_rgb=patches_rgb,
                        pos=pos,
                        gt_pos=gt_pos,
                        patches_dim=n_patches,
                        ind_name=i_name,
                        file_name=save_path,
                        correct=correct,
                    )
        
    def normalize_enface(self, img):
        img = np.asarray(img, dtype=np.float32)
        valid = img[img > 0]

        if valid.size == 0:
            return img

        lo, hi = np.percentile(valid, [1, 99])
        img = (img - lo) / (hi - lo + 1e-8)

        return np.clip(img, 0.0, 1.0)


    def add_enface_row(self, ax, row, x_center, y_center, crop_width, scan_spacing, zorder=2):
        row = np.asarray(row, dtype=np.float32).reshape(1, -1)

        alpha = (row > 0).astype(np.float32)

        ax.imshow(
            row,
            cmap="gray",
            extent=[
                x_center - crop_width / 2,
                x_center + crop_width / 2,
                y_center - scan_spacing / 2,
                y_center + scan_spacing / 2,
            ],
            origin="upper",
            aspect="auto",
            alpha=alpha,
            vmin=0.0,
            vmax=1.0,
            zorder=zorder,
        )
        # ax.set_aspect("equal", adjustable="box")

    def save_oct_image(
        self,
        patches_rgb,
        enface_rows,
        pos,
        gt_pos,
        raw_pos,
        scan_indices,
        batch_ids,
        is_anchor,
        ind_name,
        file_name: Path,
        dense_imgs,
        dense_enface,
        dense_scan_indices,
        dense_y,
        xlim,
        ylim,
        full_width,
        crop_display_width,
        scan_spacing,
        title_prefix="train",
    ):
        file_name.mkdir(parents=True, exist_ok=True)

        pos = pos.detach().cpu()
        gt_pos = gt_pos.detach().cpu()
        raw_pos = raw_pos.detach().cpu()

        scan_indices = scan_indices.detach().cpu().view(-1)
        batch_ids = batch_ids.detach().cpu().view(-1)
        is_anchor = is_anchor.detach().cpu().bool().view(-1)

        enface_rows = enface_rows.detach().cpu().numpy()

        dense_enface = dense_enface.detach().cpu().numpy()
        dense_scan_indices = dense_scan_indices.detach().cpu().view(-1)
        dense_y = dense_y.detach().cpu().view(-1)

        raw_xlim = xlim.detach().cpu().view(-1).tolist()
        raw_ylim = ylim.detach().cpu().view(-1).tolist()

        raw_xlim = [float(raw_xlim[0]), float(raw_xlim[1])]
        raw_ylim = [float(raw_ylim[0]), float(raw_ylim[1])]

        full_width = float(full_width.detach().cpu().view(-1)[0])
        crop_display_width = float(crop_display_width.detach().cpu().view(-1)[0])
        scan_spacing = float(scan_spacing.detach().cpu().view(-1)[0])

        # Normalize all puzzle en-face rows together so intensity scale
        # remains consistent between nodes.
        enface_rows = self.normalize_enface(enface_rows)

        dense_enface = np.squeeze(dense_enface)
        dense_enface = self.normalize_enface(dense_enface)

        show_dense = self.include_dense_vis and self.current_epoch < self.dense_vis_epochs and dense_enface.size > 0

        if show_dense:
            fig, axes = plt.subplots(1, 4, figsize=(40, 6))
            dense_ax = axes[0]
            original_ax = axes[1]
            gt_ax = axes[2]
            pred_ax = axes[3]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(30, 6))
            original_ax = axes[0]
            gt_ax = axes[1]
            pred_ax = axes[2]

        for ax in axes:
            ax.set_box_aspect(0.35)

        # determine axis limits
        raw_x_span = raw_xlim[1] - raw_xlim[0]
        raw_y_span = raw_ylim[1] - raw_ylim[0]

        needed_x_min = min(float(gt_pos[:, 0].min()) - crop_display_width / 2, float(pos[:, 0].min()) - crop_display_width / 2, 0.0)
        needed_x_max = max(float(gt_pos[:, 0].max()) + crop_display_width / 2, float(pos[:, 0].max()) + crop_display_width / 2, 0.0)

        needed_y_min = min(float(gt_pos[:, 1].min()) - scan_spacing / 2, float(pos[:, 1].min()) - scan_spacing / 2, 0.0)
        needed_y_max = max(float(gt_pos[:, 1].max()) + scan_spacing / 2, float(pos[:, 1].max()) + scan_spacing / 2, 0.0)

        anchored_x_span = max(raw_x_span, needed_x_max - needed_x_min)
        anchored_y_span = max(raw_y_span,  needed_y_max - needed_y_min)

        anchored_x_center = (needed_x_min + needed_x_max) / 2
        anchored_y_center = (needed_y_min + needed_y_max) / 2

        anchored_xlim = [anchored_x_center - anchored_x_span / 2, anchored_x_center + anchored_x_span / 2]
        anchored_ylim = [anchored_y_center - anchored_y_span / 2, anchored_y_center + anchored_y_span / 2]

        # dense reference en-face view
        if show_dense:
            ax = dense_ax

            if len(dense_y) > 0:
                dense_y_max = float(dense_y.max()) + scan_spacing / 2
                dense_y_min = float(dense_y.min()) - scan_spacing / 2
            else:
                dense_y_min, dense_y_max = raw_ylim

            ax.imshow(
                dense_enface,
                cmap="gray",
                extent=[
                    -full_width / 2,
                    full_width / 2,
                    dense_y_min,
                    dense_y_max,
                ],
                origin="upper",
                aspect="auto",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_xlim(raw_xlim)
            ax.set_ylim(raw_ylim)
            # ax.set_aspect("equal", adjustable="box")

            ax.set_xlim(raw_xlim)
            ax.set_ylim(raw_ylim)

            ax.grid(alpha=0.3)
            ax.set_title("Dense en-face reference")
            ax.set_xlabel("Surface x")
            ax.set_ylabel("Surface y")

        # original crop positions
        ax = original_ax

        for p in range(enface_rows.shape[0]):
            self.add_enface_row(
                ax=ax,
                row=enface_rows[p],
                x_center=float(raw_pos[p, 0]),
                y_center=float(raw_pos[p, 1]),
                crop_width=crop_display_width,
                scan_spacing=scan_spacing,
                zorder=2,
            )

            marker = "*" if is_anchor[p] else "x"
            marker_size = 100 if is_anchor[p] else 25

            ax.scatter(
                float(raw_pos[p, 0]),
                float(raw_pos[p, 1]),
                s=marker_size,
                marker=marker,
                zorder=10,
            )

            ax.text(
                float(raw_pos[p, 0]) + 0.005,
                float(raw_pos[p, 1]),
                "s{} b{}".format(
                    int(scan_indices[p]),
                    int(batch_ids[p]),
                ),
                fontsize=6,
                va="center",
                zorder=11,
            )

        ax.set_xlim(raw_xlim)
        ax.set_ylim(raw_ylim)
        
        # ax.set_aspect("equal", adjustable="box")

        ax.grid(alpha=0.3)
        ax.set_title("Original crop positions")
        ax.set_xlabel("Original x")
        ax.set_ylabel("Original y")

        # ground truth after anchor translation
        ax = gt_ax

        for p in range(enface_rows.shape[0]):
            self.add_enface_row(
                ax=ax,
                row=enface_rows[p],
                x_center=float(gt_pos[p, 0]),
                y_center=float(gt_pos[p, 1]),
                crop_width=crop_display_width,
                scan_spacing=scan_spacing,
                zorder=2,
            )

            marker = "*" if is_anchor[p] else "x"
            marker_size = 100 if is_anchor[p] else 25

            ax.scatter(
                float(gt_pos[p, 0]),
                float(gt_pos[p, 1]),
                s=marker_size,
                marker=marker,
                zorder=10,
            )

            ax.text(
                float(gt_pos[p, 0]) + 0.005,
                float(gt_pos[p, 1]),
                "s{} b{}".format(
                    int(scan_indices[p]),
                    int(batch_ids[p]),
                ),
                fontsize=6,
                va="center",
                zorder=11,
            )

        ax.axhline(
            0.0,
            color="black",
            linewidth=1.8,
            alpha=0.9,
            zorder=0,
        )

        ax.axvline(
            0.0,
            color="black",
            linewidth=1.8,
            alpha=0.9,
            zorder=0,
        )

        ax.set_xlim(anchored_xlim)
        ax.set_ylim(anchored_ylim)
        # ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.set_title("Ground truth after anchor translation")
        ax.set_xlabel("Anchor-relative x")
        ax.set_ylabel("Anchor-relative y")

        # predicted / refined positions
        ax = pred_ax

        for p in range(enface_rows.shape[0]):
            self.add_enface_row(
                ax=ax,
                row=enface_rows[p],
                x_center=float(pos[p, 0]),
                y_center=float(pos[p, 1]),
                crop_width=crop_display_width,
                scan_spacing=scan_spacing,
                zorder=2,
            )

            marker = "*" if is_anchor[p] else "x"
            marker_size = 100 if is_anchor[p] else 25

            ax.scatter(
                float(pos[p, 0]),
                float(pos[p, 1]),
                s=marker_size,
                marker=marker,
                zorder=10,
            )

            ax.text(
                float(pos[p, 0]) + 0.005,
                float(pos[p, 1]),
                "s{} b{}".format(
                    int(scan_indices[p]),
                    int(batch_ids[p]),
                ),
                fontsize=6,
                va="center",
                zorder=11,
            )

        ax.axhline(
            0.0,
            color="black",
            linewidth=1.8,
            alpha=0.9,
            zorder=0,
        )

        ax.axvline(
            0.0,
            color="black",
            linewidth=1.8,
            alpha=0.9,
            zorder=0,
        )

        ax.set_xlim(anchored_xlim)
        ax.set_ylim(anchored_ylim)
        # ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.set_title("Refined position")
        ax.set_xlabel("Anchor-relative x")
        ax.set_ylabel("Anchor-relative y")

        # error + save
        non_anchor = ~is_anchor

        if non_anchor.any():
            mean_dist = torch.norm(
                pos[non_anchor, :2] - gt_pos[non_anchor, :2],
                dim=1,
            ).mean().item()
        else:
            mean_dist = 0.0

        sample_id = int(
            ind_name.detach().cpu().view(-1)[0]
        )

        fig.suptitle(
            "{} epoch {} sample {} mean_dist {:.4f}".format(
                title_prefix,
                self.current_epoch,
                sample_id,
                mean_dist,
            )
        )

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        out_path = file_name / "oct_epoch{}_sample{}.png".format(
            self.current_epoch,
            sample_id,
        )

        fig.canvas.draw()

        im = PIL.Image.frombytes(
            "RGB",
            fig.canvas.get_width_height(),
            fig.canvas.tostring_rgb(),
        )

        self.logger.experiment.log({
            f"{file_name.stem}_oct": wandb.Image(im),
            "global_step": self.global_step,
        })

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)