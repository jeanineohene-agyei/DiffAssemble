import timm
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .exophormer_gnn import Exophormer_GNN
from .Transformer_GNN import Transformer_GNN


class Eff_GAT(nn.Module):
    """
    This model has 45M parameters


    Args:
        nn (_type_): _description_
    """
    def __init__(
        self,
        steps,
        input_channels=2,
        output_channels=2,
        n_layers=4,
        visual_pretrained=True,
        freeze_backbone=False,
        model="efficientnet_b0",
        architecture="transformer",
        virt_nodes=4,
        all_equivariant=False
    ) -> None:
        super().__init__()
        self.visual_backbone = timm.create_model(
                model, pretrained=visual_pretrained, features_only=True
            )
        self.all_equivariant=all_equivariant
        self.model = model
        self.combined_features_dim = {
            "resnet18": 3136,
            "resnet50": 12352,
            # visual + noisy xy + time + anchor + rough location
            "efficientnet_b0": 6144 + 32 + 32 + 16 + 128,
            #97792 + 32 + 32 resnet50
        }[model]

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.freeze_backbone = freeze_backbone

        if architecture == "transformer":
            self.gnn_backbone = Transformer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
                edge_dim=2,
            )
        elif architecture == "exophormer":
            self.gnn_backbone = Exophormer_GNN(
                self.combined_features_dim,
                n_layers=n_layers,
                hidden_dim=32 * 8,
                heads=8,
                output_size=self.combined_features_dim,
                virt_nodes=virt_nodes
            )
        
        self.anchor_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
            nn.Linear(8, 16),
        )
        
        self.rough_location_mlp = nn.Sequential(
            # rough_delta_x, rough_delta_y, radius_x, radius_y
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 32),
            nn.GELU(),
            nn.Linear(32, output_channels),
            nn.Tanh(),
        )
        nn.init.zeros_(self.final_mlp[-2].weight)
        nn.init.zeros_(self.final_mlp[-2].bias)
        
        self.time_emb = nn.Embedding(steps, 32)
        self.pos_mlp = nn.Sequential(
            nn.Linear(input_channels, 16), nn.GELU(), nn.Linear(16, 32)
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )


        self.linear1 = nn.Linear(8192, 544) #  # dimension for resnet18

        self.linear2 = nn.Linear(4096, 544)  # dimension for resnet18

        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, xy_pos, time, patch_rgb, edge_index, edge_attr, batch, is_anchor, rough_delta, rough_radius):
        patch_feats = self.visual_features(patch_rgb)
        final_feats = self.forward_with_feats(
            xy_pos, time, edge_index, edge_attr, patch_feats=patch_feats, batch=batch, is_anchor=is_anchor, rough_delta=rough_delta, rough_radius=rough_radius
        )
        return final_feats

    def forward_with_feats(
        self: nn.Module,
        xy_pos: Tensor,
        time: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        patch_feats: Tensor,
        batch,
        is_anchor,
        rough_delta: Tensor,
        rough_radius: Tensor,
    ):

        time_feats = self.time_emb(time)  # embedding, int -> 32
        pos_feats = self.pos_mlp(xy_pos)  # MLP, (x, y) -> 32
        rough_input = torch.cat([rough_delta.float(), rough_radius.float()], dim=-1)
        rough_feats = self.rough_location_mlp(rough_input)
        anchor_feats = self.anchor_mlp(is_anchor.float())

        combined_feats = torch.cat([patch_feats, pos_feats, time_feats, anchor_feats, rough_feats], -1)
        combined_feats = self.mlp(combined_feats)

        # GNN
        feats, attentions = self.gnn_backbone(
            x=combined_feats, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )


        # Residual + final transform
        final_feats = self.final_mlp(
            feats + combined_feats)
        return final_feats, attentions


    def visual_features(self, patch_rgb):
        patch_rgb = (patch_rgb - self.mean) / self.std

        if self.freeze_backbone:
            with torch.no_grad():
                feats = self.visual_backbone.forward(patch_rgb)
        else:
            if self.all_equivariant:
                feats = [self.visual_backbone.forward(patch_rgb[:, i, :, :, :]) for i in range(4)]
                feats = [(feats[1][i] + feats[2][i] + feats[3][i] + feats[0][i])/4 for i in range(len(feats[1]))]

            else:
                feats = self.visual_backbone.forward(patch_rgb)
        feats = {
            # "efficientnet_b0": [
            #     feats[2].reshape(patch_rgb.shape[0], -1),
            #     feats[3].reshape(patch_rgb.shape[0], -1),
            # ],
            "efficientnet_b0": [
                F.adaptive_avg_pool2d(feats[2], (16, 4)).flatten(1),
                F.adaptive_avg_pool2d(feats[3], (8, 4)).flatten(1)
            ],
            "resnet50": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
            "resnet18": [
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
            ],
                
            "resnet18equiv":[
                feats[2].reshape(patch_rgb.shape[0], -1),
                feats[3].reshape(patch_rgb.shape[0], -1),
              ]
        }[self.model]
        #if self.all_equivariant:
        #    feats[0] = self.linear1(feats[0].view(feats[0].size(0),-1)) # concatenation
        #    feats[1] = self.linear2(feats[1].view(feats[1].size(0),-1)) # concatenation features
            #feats[0] = self.linear1(feats[0])
            #feats[1] = self.linear2(feats[1])
        #    return torch.cat(feats, -1)
        #else:
        patch_feats = torch.cat(feats, -1)
        return patch_feats
