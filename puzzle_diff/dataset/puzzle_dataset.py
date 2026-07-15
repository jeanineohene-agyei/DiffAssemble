import math
from typing import List

import einops
import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torchvision.transforms as transforms
from scipy.sparse.linalg import eigsh
from torch import Tensor
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from pathlib import Path
import tifffile as tiff

# import albumentations
# import cv2


def generate_random_expander(num_nodes, degree, rng=None, max_num_iters=5, exp_index=0):
    """Generates a random d-regular expander graph with n nodes.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).
    Args:
      num_nodes: Number of nodes in the desired graph.
      degree: Desired degree.
      rng: random number generator
      max_num_iters: maximum number of iterations
    Returns:
      senders: tail of each edge.
      receivers: head of each edge.
    """
    if isinstance(degree, str):
        degree = round((int(degree[:-1]) * (num_nodes - 1)) / 100)
    num_nodes = num_nodes

    if rng is None:
        rng = np.random.default_rng()
    eig_val = -1
    eig_val_lower_bound = (
        max(0, degree - 2 * math.sqrt(degree - 1) - 0.1) if degree > 0 else 0
    )  # allow the use of zero degree

    max_eig_val_so_far = -1
    max_senders = []
    max_receivers = []
    cur_iter = 1

    # (bave): This is a hack.  This should hopefully fix the bug
    if num_nodes <= degree:
        degree = num_nodes - 1

    # (ali): if there are too few nodes, random graph generation will fail. in this case, we will
    # add the whole graph.
    if num_nodes <= 10:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    max_senders.append(i)
                    max_receivers.append(j)
    else:
        while eig_val < eig_val_lower_bound and cur_iter <= max_num_iters:
            senders, receivers = generate_random_regular_graph(num_nodes, degree, rng)

            eig_val = get_eigenvalue(senders, receivers, num_nodes=num_nodes)
            if len(eig_val) == 0:
                print(
                    "num_nodes = %d, degree = %d, cur_iter = %d, mmax_iters = %d, senders = %d, receivers = %d"
                    % (
                        num_nodes,
                        degree,
                        cur_iter,
                        max_num_iters,
                        len(senders),
                        len(receivers),
                    )
                )
                eig_val = 0
            else:
                eig_val = eig_val[0]
            if eig_val > max_eig_val_so_far:
                max_eig_val_so_far = eig_val
                max_senders = senders
                max_receivers = receivers

            cur_iter += 1
    max_senders = torch.tensor(max_senders, dtype=torch.long).view(-1, 1)
    max_receivers = torch.tensor(max_receivers, dtype=torch.long).view(-1, 1)
    expander_edges = torch.cat([max_senders, max_receivers], dim=1)
    return expander_edges


def get_eigenvalue(senders, receivers, num_nodes):
    edge_index = torch.tensor(np.stack([senders, receivers]))
    edge_index, edge_weight = get_laplacian(
        edge_index, None, normalization=None, num_nodes=num_nodes
    )
    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    return eigsh(L, k=2, which="SM", return_eigenvectors=False)


def generate_random_regular_graph(num_nodes, degree, rng=None):
    """Generates a random d-regular connected graph with n nodes.
    Returns the list of edges. This list is symmetric; i.e., if
    (x, y) is an edge so is (y,x).
    Args:
      num_nodes: Number of nodes in the desired graph.
      degree: Desired degree.
      rng: random number generator
    Returns:
      senders: tail of each edge.
      receivers: head of each edge.
    """
    if (num_nodes * degree) % 2 != 0:
        raise TypeError("nodes * degree must be even")
    if rng is None:
        rng = np.random.default_rng()
    if degree == 0:
        return np.array([]), np.array([])
    nodes = rng.permutation(np.arange(num_nodes))
    num_reps = degree // 2
    num_nodes = len(nodes)

    ns = np.hstack([np.roll(nodes, i + 1) for i in range(num_reps)])
    edge_index = np.vstack((np.tile(nodes, num_reps), ns))

    if degree % 2 == 0:
        senders, receivers = np.concatenate(
            [edge_index[0], edge_index[1]]
        ), np.concatenate([edge_index[1], edge_index[0]])
        return senders, receivers
    else:
        edge_index = np.hstack(
            (edge_index, np.vstack((nodes[: num_nodes // 2], nodes[num_nodes // 2 :])))
        )
        senders, receivers = np.concatenate(
            [edge_index[0], edge_index[1]]
        ), np.concatenate([edge_index[1], edge_index[0]])
        return senders, receivers


class RandomCropAndResizedToOriginal(transforms.RandomResizedCrop):
    def forward(self, img):
        size = img.size
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, size, self.interpolation)


def _get_augmentation(augmentation_type: str = "none"):
    switch = {
        "weak": [transforms.RandomHorizontalFlip(p=0.5)],
        "hard": [
            transforms.RandomHorizontalFlip(p=0.5),
            RandomCropAndResizedToOriginal(
                size=(1, 1), scale=(0.8, 1), interpolation=InterpolationMode.BICUBIC
            ),
        ],
    }
    return switch.get(augmentation_type, [])


@torch.jit.script
def divide_images_into_patches(
    img, patch_per_dim: List[int], patch_size: int
) -> List[Tensor]:
    # img2 = einops.rearrange(img, "c h w -> h w c")

    # divide images in non-overlapping patches based on patch size
    # output dim -> a
    img2 = img.permute(1, 2, 0)
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    y = torch.linspace(-1, 1, patch_per_dim[0])
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    # print(patch_per_dim)

    return xy, patches


# generation of a unique graph for each number of nodes
def create_graph(patch_per_dim, degree, unique_graph):
    # Create an empty dictionary
    patch_edge_index_dict = {}
    for patch_dim in patch_per_dim:
        if degree == -1:
            num_patches = patch_dim[0] * patch_dim[1]
            adj_mat = torch.ones(num_patches, num_patches)
            edge_index, _ = adj_mat.nonzero().t().contiguous()
        else:
            num_patches = patch_dim[0] * patch_dim[1]
            edge_index = (
                generate_random_expander(
                    num_nodes=num_patches, degree=degree, rng=unique_graph
                )
                .t()
                .contiguous()
            )
        patch_edge_index_dict[patch_dim] = edge_index
    return patch_edge_index_dict


class Puzzle_Dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment="",
        degree=-1,
        unique_graph=None,
        random=False,
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn
        self.patch_per_dim = patch_per_dim
        self.unique_graph = unique_graph
        self.augment = augment
        self.random = random

        self.transforms = transforms.Compose(
            [
                *_get_augmentation(augment),
                transforms.ToTensor(),
            ]
        )
        self.patch_size = patch_size
        self.degree = degree

        if self.unique_graph is not None:
            self.edge_index = create_graph(
                self.patch_per_dim, self.degree, self.unique_graph
            )

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size
        img = img.resize((width, height))#, resample=Resampling.BICUBIC)
        img = self.transforms(img)

        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")

        indexes = torch.arange(patch_per_dim[0] * patch_per_dim[1]).reshape(
            xy.shape[:-1]
        )
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")
        if self.random:
            patches = patches[torch.randperm(len(patches))]
        if self.degree == -1:
            adj_mat = torch.ones(
                patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
            )

            edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
        else:
            if not self.unique_graph:
                edge_index = generate_random_expander(
                    patch_per_dim[0] * patch_per_dim[1], self.degree
                ).T
        data = pyg_data.Data(
            x=xy,
            indexes=indexes,
            patches=patches,
            edge_index=self.edge_index[patch_per_dim]
            if self.unique_graph
            else edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class OCTPuzzleDataset(pyg_data.Dataset):
    def __init__(
        self,
        bscan_paths,
        seg_paths,
        dense_size=30,
        min_num_batches=2,
        max_num_batches=8,
        min_batch_size=3,
        max_batch_size=3,
        crop_l=220,
        valid_scan_start=50,
        valid_scan_end=200,
        margin_y=40,
        margin_x_left=260,
        margin_x_right=260,
        min_mask_pixels=500,
        display_height=0.55,
        scan_spacing=0.10,
        width_stretch=0.8,
        degree=-1,
        seed=42,
    ):
        super().__init__()
        self.bscan_paths = list(bscan_paths)
        self.seg_paths = list(seg_paths)
        assert len(self.bscan_paths) == len(self.seg_paths)

        self.dense_size = dense_size
        self.min_num_batches = min_num_batches
        self.max_num_batches = max_num_batches
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.crop_l = crop_l

        self.valid_scan_start = valid_scan_start
        self.valid_scan_end = valid_scan_end
        self.margin_y = margin_y
        self.margin_x_left = margin_x_left
        self.margin_x_right = margin_x_right
        self.min_mask_pixels = min_mask_pixels
        
        self.rotation_overrides = {
            "a947ccaffbe1": 1,
        }

        self.default_rotation = 3

        self.display_height = display_height
        self.scan_spacing = scan_spacing
        self.width_stretch = width_stretch
        self.degree = degree
        if seed is None:
            self.seed = np.random.SeedSequence().generate_state(1)[0].item()
        else:
            self.seed = seed

    def len(self):
        return len(self.bscan_paths)

    def norm_img(self, img):
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, [1, 99])
        return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)

    # def rotate_volume(self, vol):
    #     return np.rot90(vol, k=3, axes=(1, 2)).copy()
    
    def rotate_volume(self, vol, bscan_path):
        participant = Path(bscan_path).parent.name

        k = self.rotation_overrides.get(
            participant,
            self.default_rotation,
        )

        return np.rot90(vol, k=k, axes=(1, 2)).copy()

    def global_anatomy_crop(self, stack, seg):
        mask = seg > 0
        rows = np.where(mask.any(axis=(0, 2)))[0]
        cols = np.where(mask.any(axis=(0, 1)))[0]

        r0 = max(0, rows[0] - self.margin_y)
        r1 = min(stack.shape[1], rows[-1] + self.margin_y)
        c0 = max(0, cols[0] - self.margin_x_left)
        c1 = min(stack.shape[2], cols[-1] + self.margin_x_right)

        return stack[:, r0:r1, c0:c1], seg[:, r0:r1, c0:c1]

    def full_display_width(self, full_shape):
        H, W = full_shape
        return self.display_height * (W / H) * self.width_stretch

    def col_to_full_x(self, col, full_shape):
        H, W = full_shape
        width = self.full_display_width(full_shape)
        return ((col / (W - 1)) - 0.5) * width

    def pixel_to_display_xy(self, row, col, y_scan, full_shape):
        H, W = full_shape
        x = self.col_to_full_x(col, full_shape)
        y = y_scan + (0.5 - (row / (H - 1))) * self.display_height
        return x, y

    def anatomy_center_pixels(self, img, seg):
        m = seg > 0

        if m.sum() < self.min_mask_pixels:
            threshold = np.percentile(img.astype(np.float32), 90)
            m = img > threshold

        rr, cc = np.where(m)
        if len(rr) == 0:
            return None

        return rr.mean(), cc.mean()

    def contiguous_ranges(self, indices):
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

    def fill_small_gaps(self, valid_mask, max_gap=1):
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
            has_left = gap_start > 0 and valid_mask[gap_start - 1]
            has_right = gap_end < n - 1 and valid_mask[gap_end + 1]

            if has_left and has_right and gap_len <= max_gap:
                valid_mask[gap_start:gap_end + 1] = True

        return valid_mask

    def best_valid_scan_range(self, seg):
        counts = np.sum(seg > 0, axis=(1, 2))
        peak = int(counts.max())

        threshold = max(
            self.min_mask_pixels,
            int(0.10 * peak),
        )

        valid_mask = counts >= threshold
        valid_mask = self.fill_small_gaps(valid_mask, max_gap=1)

        valid = np.where(valid_mask)[0]
        ranges = self.contiguous_ranges(valid)

        if len(ranges) == 0:
            return 0, seg.shape[0]

        best_start, best_end = max(ranges, key=lambda r: r[1] - r[0] + 1)
        return best_start, best_end + 1

    def sample_dense_indices(self, rng, n_scans, valid_start, valid_end):
        valid_start = max(0, int(valid_start))
        valid_end = min(n_scans, int(valid_end))

        if valid_end - valid_start < self.dense_size:
            valid_start = 0
            valid_end = n_scans

        start_min = valid_start
        start_max = valid_end - self.dense_size

        if start_max < start_min:
            start_min = 0
            start_max = n_scans - self.dense_size

        start = rng.integers(start_min, start_max + 1)
        return list(range(start, start + self.dense_size))

    def sample_one_batch(self, rng, dense_indices):
        max_size = min(self.max_batch_size, len(dense_indices))
        min_size = min(self.min_batch_size, max_size)

        batch_size = rng.integers(min_size, max_size + 1)
        start = rng.integers(0, len(dense_indices) - batch_size + 1)

        return dense_indices[start:start + batch_size]

    def sample_batches(self, rng, dense_indices):
        num_batches = rng.integers(self.min_num_batches, self.max_num_batches + 1)
        return [self.sample_one_batch(rng, dense_indices) for _ in range(num_batches)]

    def sample_crop_cols(self, rng, W):
        center = rng.integers(self.crop_l // 2, W - self.crop_l // 2 + 1)
        return center - self.crop_l // 2, center + self.crop_l // 2

    def make_complete_graph(self, num_nodes):
        adj_mat = torch.ones(num_nodes, num_nodes)
        edge_index, _ = pyg.utils.dense_to_sparse(adj_mat)
        return edge_index

    def crop_to_patch_tensor(self, img_crop):
        img_crop = self.norm_img(img_crop)
        tensor = torch.from_numpy(img_crop).float()[None, :, :]
        tensor = tensor.repeat(3, 1, 1)
        return tensor

    def get(self, idx):
        rng = np.random.default_rng(self.seed + idx)

        stack = tiff.imread(self.bscan_paths[idx])
        seg = tiff.imread(self.seg_paths[idx])

        stack = self.rotate_volume(stack, self.bscan_paths[idx])
        seg = self.rotate_volume(seg, self.bscan_paths[idx])

        # stack, seg = self.global_anatomy_crop(stack, seg)

        valid_start, valid_end = self.best_valid_scan_range(seg)

        dense_indices = self.sample_dense_indices(rng, stack.shape[0], valid_start, valid_end)

        full_shape = stack[0].shape
        full_width = self.full_display_width(full_shape)

        center = (len(dense_indices) - 1) / 2
        ys = (center - np.arange(len(dense_indices))) * self.scan_spacing
        idx_to_y = dict(zip(dense_indices, ys))

        x_pad = 0.05 * full_width
        xlim = [-full_width / 2 - x_pad, full_width / 2 + x_pad]

        y_pad = self.scan_spacing * 2
        ylim = [float(ys.min() - y_pad), float(ys.max() + y_pad)]

        crop_display_width = (
            self.col_to_full_x(self.crop_l, full_shape)
            - self.col_to_full_x(0, full_shape)
        )

        dense_imgs = []
        dense_scan_indices = []

        for scan_idx in dense_indices:
            dense_imgs.append(self.crop_to_patch_tensor(stack[scan_idx]))
            dense_scan_indices.append(scan_idx)

        dense_imgs = torch.stack(dense_imgs)
        dense_scan_indices = torch.tensor(dense_scan_indices).long()
        dense_y = torch.tensor(ys, dtype=torch.float32)

        node_patches = []
        node_raw_xy = []
        node_scan_indices = []
        node_batch_ids = []
        node_crop_windows = []

        batches = self.sample_batches(rng, dense_indices)

        valid_batch_id = 0

        for batch_indices in batches:
            c0, c1 = self.sample_crop_cols(rng, stack.shape[2])

            current_patches = []
            current_xy = []
            current_scan_indices = []

            for scan_idx in batch_indices:
                y_scan = idx_to_y[scan_idx]

                img_crop = stack[scan_idx, :, c0:c1]
                seg_crop = seg[scan_idx, :, c0:c1]

                center_px = self.anatomy_center_pixels(img_crop, seg_crop)

                if center_px is None:
                    continue

                row, local_col = center_px
                full_col = c0 + local_col

                gx, gy = self.pixel_to_display_xy(row, full_col, y_scan, full_shape)

                current_patches.append(self.crop_to_patch_tensor(img_crop))
                current_xy.append([gx, gy])
                current_scan_indices.append(scan_idx)

            if len(current_patches) != len(batch_indices):
                continue

            for patch, xy, scan_idx in zip(current_patches, current_xy, current_scan_indices):
                node_patches.append(patch)
                node_raw_xy.append(xy)
                node_scan_indices.append(scan_idx)
                node_batch_ids.append(valid_batch_id)
                node_crop_windows.append([c0, c1])

            valid_batch_id += 1

        if valid_batch_id < self.min_num_batches:
            raise RuntimeError(
                "OCT sample produced fewer than {} valid batches at index {}".format(
                    self.min_num_batches, idx
                )
            )

        if len(node_patches) < 2:
            raise RuntimeError(
                "OCT sample produced fewer than 2 valid scan nodes at index {}".format(idx)
            )

        patches = torch.stack(node_patches)
        raw_xy = torch.tensor(node_raw_xy, dtype=torch.float32)

        num_nodes = raw_xy.shape[0]
        edge_index = self.make_complete_graph(num_nodes)

        anchor_idx = int(rng.integers(0, num_nodes))
        anchor_xy = raw_xy[anchor_idx].clone()

        gt_delta = raw_xy - anchor_xy
        gt_delta[anchor_idx] = 0.0

        is_anchor = torch.zeros(num_nodes, 1, dtype=torch.float32)
        is_anchor[anchor_idx, 0] = 1.0

        data = pyg_data.Data(
            x=gt_delta,
            patches=patches,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([[num_nodes, 1]]),

            scan_indices=torch.tensor(node_scan_indices).long(),
            batch_ids=torch.tensor(node_batch_ids).long(),
            crop_windows=torch.tensor(node_crop_windows).long(),

            raw_xy=raw_xy,

            gt_delta=gt_delta,
            anchor_idx=torch.tensor([anchor_idx]).long(),
            anchor_xy=anchor_xy[None, :],
            is_anchor=is_anchor,

            dense_imgs=dense_imgs.unsqueeze(0),
            dense_scan_indices=dense_scan_indices.unsqueeze(0),
            dense_y=dense_y.unsqueeze(0),

            xlim=torch.tensor(xlim, dtype=torch.float32),
            ylim=torch.tensor(ylim, dtype=torch.float32),
            full_width=torch.tensor([full_width], dtype=torch.float32),
            crop_display_width=torch.tensor([crop_display_width], dtype=torch.float32),
            display_height=torch.tensor([self.display_height], dtype=torch.float32),
        )

        return data

