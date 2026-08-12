import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
from pathlib import Path
import tifffile as tiff


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
        min_total_nodes=20,
        max_total_nodes=20,
        min_unique_scans=10,
        crop_l=220,
        valid_scan_start=50,
        valid_scan_end=200,
        margin_y=40,
        margin_x_left=260,
        margin_x_right=260,
        min_mask_pixels=500,
        seed=42,
        randomize_samples=False,
        samples_per_volume=1,
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
        self.min_total_nodes = min_total_nodes
        self.max_total_nodes = max_total_nodes
        self.min_unique_scans = min_unique_scans
        self.valid_scan_start = valid_scan_start
        self.valid_scan_end = valid_scan_end
        self.margin_y = margin_y
        self.margin_x_left = margin_x_left
        self.margin_x_right = margin_x_right
        self.min_mask_pixels = min_mask_pixels
        self.randomize_samples = randomize_samples
        self.samples_per_volume = int(samples_per_volume)
        
        self.rotation_overrides = {"a947ccaffbe1": 1}

        self.default_rotation = 3
        
        # physical anterior OCT acquisition geometry (from the paper where the data comes from)
        self.full_fov_x_mm = 35.0
        self.full_fov_y_mm = 17.5
        self.full_num_ascans = 500
        self.full_num_bscans = 250

        self.x_spacing_mm = self.full_fov_x_mm / self.full_num_ascans
        self.y_spacing_mm = self.full_fov_y_mm / self.full_num_bscans

        self.rough_radius_x = 10 * self.x_spacing_mm
        self.rough_radius_y = 10 * self.y_spacing_mm

        self.enface_depth_pixels = 100
        
        if seed is None:
            self.seed = np.random.SeedSequence().generate_state(1)[0].item()
        else:
            self.seed = seed

    def len(self):
        return len(self.bscan_paths) * self.samples_per_volume

    def norm_img(self, img):
        img = img.astype(np.float32)
        lo, hi = np.percentile(img, [1, 99])
        return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    
    def rotate_volume(self, vol, bscan_path):
        participant = Path(bscan_path).parent.name
        k = self.rotation_overrides.get(participant, self.default_rotation)
        return np.rot90(vol, k=k, axes=(1, 2)).copy()
    

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

        threshold = max(self.min_mask_pixels, int(0.10 * peak))

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
        max_attempts = 100
        for _ in range(max_attempts):
            num_batches = rng.integers(self.min_num_batches, self.max_num_batches + 1)
            batches = [self.sample_one_batch(rng, dense_indices) for _ in range(num_batches)]
            
            total_nodes = sum(len(batch) for batch in batches)
            unique_scans = len(set(scan_idx for batch in batches for scan_idx in batch))

            if total_nodes >= self.min_total_nodes and total_nodes <= self.max_total_nodes and unique_scans >= self.min_unique_scans:
                return batches
            
        raise RuntimeError(f"Could not sample a puzzle with at least {self.min_total_nodes} nodes, at most {self.max_total_nodes} nodes, and {self.min_unique_scans} unique scans.")

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
    
    def create_rough_relative_locations(self, gt_delta, anchor_idx, rng):
        """
        Simulate rough GPS-like relative-location metadata.

        For each scan:
            rough_delta = gt_delta + random error

        The error is uniformly sampled inside a circle with radius
        self.rough_location_radius, so the true location is guaranteed to
        lie inside the stated uncertainty circle around the rough point.

        Args:
            gt_delta: Tensor [num_nodes, 2]
                Ground-truth positions relative to the anchor.
            anchor_idx: int
                Anchor node index.
            rng: np.random.Generator

        Returns:
            rough_delta: Tensor [num_nodes, 2]
                Noisy anchor-relative location estimate.
            rough_radius: Tensor [num_nodes, 1]
                Physical error radius associated with each estimate.
        """
        num_nodes = gt_delta.shape[0]

        theta = rng.uniform(0.0, 2.0 * np.pi, size=num_nodes)

        # sqrt gives uniform sampling over the area of the circle
        r = np.sqrt(rng.uniform(0.0, 1.0, size=num_nodes))
        dx = r * self.rough_radius_x * np.cos(theta)
        dy = r * self.rough_radius_y * np.sin(theta)

        noise = torch.from_numpy(np.stack([dx, dy], axis=1).astype(np.float32))
        rough_delta = gt_delta + noise

        # anchor defines the local coordinate system
        rough_delta[anchor_idx] = 0.0

        rough_radius = torch.zeros((num_nodes, 2), dtype=torch.float32)
        rough_radius[:, 0] = float(self.rough_radius_x)
        rough_radius[:, 1] = float(self.rough_radius_y)

        # anchor is known to be the origin
        rough_radius[anchor_idx] = 0.0

        return rough_delta, rough_radius
    
    def make_rough_neighborhood_graph(self, rough_delta, rough_radius, anchor_idx, min_neighbors=3):
        num_nodes = rough_delta.shape[0]
        edges = set()

        for i in range(num_nodes):
            if i == anchor_idx:
                continue

            dx = torch.abs(rough_delta[:, 0] - rough_delta[i, 0])
            dy = torch.abs(rough_delta[:, 1] - rough_delta[i, 1])

            x_limit = rough_radius[:, 0] + rough_radius[i, 0]
            y_limit = rough_radius[:, 1] + rough_radius[i, 1]

            valid = (dx <= x_limit) & (dy <= y_limit)
            valid[i] = False
            valid[anchor_idx] = False

            neighbors = torch.where(valid)[0]

            for j in neighbors.tolist():
                edges.add((i, j))
                edges.add((j, i))

        # source anchor sends to every prediction node
        for j in range(num_nodes):
            if j != anchor_idx:
                edges.add((anchor_idx, j))

        return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    
    def col_to_surface_x(self, col, width):
        center_col = (width - 1) / 2.0
        return (float(col) - center_col) * self.x_spacing_mm


    def make_enface_row(self, img_crop, seg_crop, depth_pixels=100):
        H, W = img_crop.shape
        enface_row = np.zeros(W, dtype=np.float32)

        for x in range(W):
            rows = np.where(seg_crop[:, x] > 0)[0]

            if len(rows) == 0:
                continue

            surface_row = rows[0]
            depth_end = min(surface_row + depth_pixels, H)
            values = img_crop[surface_row:depth_end, x]

            if len(values) > 0:
                enface_row[x] = values.mean()

        return torch.from_numpy(enface_row).float()
    
    
    def make_enface_projection(self, stack, seg, scan_indices, depth_pixels=100):
        rows = []
        for scan_idx in scan_indices:
            rows.append(
                self.make_enface_row(stack[scan_idx], seg[scan_idx], depth_pixels=depth_pixels))
        return torch.stack(rows)


    def get(self, idx):
        volume_idx = idx // self.samples_per_volume
        sample_idx = idx % self.samples_per_volume
        
        if self.randomize_samples:
            worker_seed = torch.initial_seed()
            sample_seed = (worker_seed + volume_idx * 1_000_003 + sample_idx * 10_007) % (2**32)
        else:
            # validation remains fixed and repeatable
            sample_seed = (self.seed + volume_idx * 1_000_003 + sample_idx * 10_007) % (2**32)

        rng = np.random.default_rng(sample_seed)

        bscan_path, seg_path = self.bscan_paths[volume_idx], self.seg_paths[volume_idx]
        stack, seg = tiff.imread(bscan_path), tiff.imread(seg_path)
        stack, seg = self.rotate_volume(stack, bscan_path), self.rotate_volume(seg, bscan_path)

        valid_start, valid_end = self.best_valid_scan_range(seg)

        dense_indices = self.sample_dense_indices(rng, stack.shape[0], valid_start, valid_end)
        dense_enface = self.make_enface_projection(stack, seg, dense_indices, depth_pixels=100)

        full_shape = stack[0].shape
        _, full_width_pixels = full_shape

        full_width_mm = full_width_pixels * self.x_spacing_mm

        center_row = (len(dense_indices) - 1) / 2.0
        ys = (center_row - np.arange(len(dense_indices))) * self.y_spacing_mm
        idx_to_y = dict(zip(dense_indices, ys))

        dense_height_mm = len(dense_indices) * self.y_spacing_mm

        xlim = [-full_width_mm / 2.0, full_width_mm / 2.0]
        ylim = [-dense_height_mm / 2.0, dense_height_mm / 2.0]

        crop_display_width = self.crop_l * self.x_spacing_mm
        
        dense_imgs, dense_scan_indices = [], []
        for scan_idx in dense_indices:
            dense_imgs.append(self.crop_to_patch_tensor(stack[scan_idx]))
            dense_scan_indices.append(scan_idx)

        dense_imgs = torch.stack(dense_imgs)
        dense_scan_indices = torch.tensor(dense_scan_indices).long()
        dense_y = torch.tensor(ys, dtype=torch.float32)

        node_patches = []
        node_enface = []
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
            current_enface = []

            for scan_idx in batch_indices:
                y_scan = idx_to_y[scan_idx]

                img_crop = stack[scan_idx, :, c0:c1]
                seg_crop = seg[scan_idx, :, c0:c1]
                
                enface_row = self.make_enface_row(img_crop, seg_crop, depth_pixels=100)
                center_px = self.anatomy_center_pixels(img_crop, seg_crop)

                if center_px is None:
                    continue

                gx = self.col_to_surface_x((c0 + c1 - 1) / 2.0, full_width_pixels)
                gy = float(y_scan)
               
                current_patches.append(self.crop_to_patch_tensor(img_crop))
                current_enface.append(enface_row)
                current_xy.append([gx, gy])
                current_scan_indices.append(scan_idx)

            if len(current_patches) != len(batch_indices):
                continue

            for patch, enface_row, xy, scan_idx in zip(current_patches, current_enface, current_xy, current_scan_indices):
                node_patches.append(patch)
                node_enface.append(enface_row)
                node_raw_xy.append(xy)
                node_scan_indices.append(scan_idx)
                node_batch_ids.append(valid_batch_id)
                node_crop_windows.append([c0, c1])

            valid_batch_id += 1

        if valid_batch_id < self.min_num_batches:
            raise RuntimeError("OCT sample produced fewer than {} valid batches at index {}".format(self.min_num_batches, idx))

        if len(node_patches) < 2:
            raise RuntimeError("OCT sample produced fewer than 2 valid scan nodes at index {}".format(idx))

        patches = torch.stack(node_patches)
        enface_rows = torch.stack(node_enface)
        raw_xy = torch.tensor(node_raw_xy, dtype=torch.float32)

        num_nodes = raw_xy.shape[0]

        anchor_idx = int(rng.integers(0, num_nodes))
        # anchor_idx = num_nodes // 2
        anchor_xy = raw_xy[anchor_idx].clone()

        gt_delta = raw_xy - anchor_xy
        gt_delta[anchor_idx] = 0.0
        
        x_scale = full_width_mm
        y_scale = dense_height_mm
        delta_scale = torch.tensor([x_scale, y_scale], dtype=torch.float32)
        
        # coordinates used by the model/diffusion process
        gt_delta_model = gt_delta / delta_scale
        gt_delta_model[anchor_idx] = 0.0
        
        # rough GPS-like conditioning metadata
        rough_delta, rough_radius = self.create_rough_relative_locations(gt_delta=gt_delta, anchor_idx=anchor_idx, rng=rng)
        
        edge_index = self.make_rough_neighborhood_graph(rough_delta, rough_radius, anchor_idx)
        # edge_index = self.make_complete_graph(num_nodes)

        # normalize rough coordinates using the same scale as the target
        rough_delta_model = rough_delta / delta_scale
        rough_delta_model[anchor_idx] = 0.0
        
        src = edge_index[0]
        dst = edge_index[1]
        
        # directed displacement from source node to destination node
        edge_attr = rough_delta_model[dst] - rough_delta_model[src]
        
        # residual correction in physical coordinates
        correction = gt_delta - rough_delta
        correction[anchor_idx] = 0.0

        # normalize the residual by the uncertainty radius on each axis
        # This makes X and Y targets comparable:
        #   correction_x / 0.04
        #   correction_y / 0.20
        correction_scale = torch.tensor([self.rough_radius_x, self.rough_radius_y], dtype=torch.float32)
        
        correction_model = correction / correction_scale
        correction_model[anchor_idx] = 0.0

        # rough_radius has shape [num_nodes, 2]:
        #   [:, 0] = raw X uncertainty radius
        #   [:, 1] = raw Y uncertainty radius
        #
        # Convert each raw radius into model coordinates using the matching
        # axis-specific normalization scale
        rough_radius_model = rough_radius / delta_scale.unsqueeze(0)
        rough_radius_model[anchor_idx] = 0.0
        
        is_anchor = torch.zeros(num_nodes, 1, dtype=torch.float32)
        is_anchor[anchor_idx, 0] = 1.0
        
        data = pyg_data.Data(
            # diffusion target
            x=correction_model,

            # model visual input
            patches=patches,

            # visualization-only en-face information
            enface_rows=enface_rows,
            dense_enface=dense_enface.unsqueeze(0),

            # graph, where edges exist and what information is passed along them
            edge_index=edge_index,
            edge_attr=edge_attr,

            # sample metadata
            ind_name=torch.tensor([volume_idx]).long(),
            sample_idx=torch.tensor([sample_idx]).long(),
            sample_seed=torch.tensor([sample_seed]).long(),
            patches_dim=torch.tensor([[num_nodes, 1]]),

            # node metadata
            scan_indices=torch.tensor(node_scan_indices).long(),
            batch_ids=torch.tensor(node_batch_ids).long(),
            crop_windows=torch.tensor(node_crop_windows).long(),

            # original physical coordinates in mm
            raw_xy=raw_xy,
            gt_delta=gt_delta,

            # model-coordinate target and normalization
            gt_delta_model=gt_delta_model,
            delta_scale=delta_scale.unsqueeze(0),

            # rough position conditioning in physical coordinates (raw mm)
            rough_delta=rough_delta,
            rough_radius=rough_radius,

            # rough position conditioning in model coordinates (normalized)
            rough_delta_model=rough_delta_model,
            rough_radius_model=rough_radius_model,

            # Residual correction target: raw, normalized and norm scale
            correction=correction,
            correction_model=correction_model,
            correction_scale=correction_scale.unsqueeze(0),

            # anchor information
            anchor_idx=torch.tensor([anchor_idx]).long(),
            anchor_xy=anchor_xy.unsqueeze(0),
            is_anchor=is_anchor,

            # dense reference information
            dense_imgs=dense_imgs.unsqueeze(0),
            dense_scan_indices=dense_scan_indices.unsqueeze(0),
            dense_y=dense_y.unsqueeze(0),

            # plot / physical FOV information
            xlim=torch.tensor(xlim, dtype=torch.float32),
            ylim=torch.tensor(ylim, dtype=torch.float32),

            full_width=torch.tensor([full_width_mm], dtype=torch.float32),
            crop_display_width=torch.tensor([crop_display_width], dtype=torch.float32),

            # physical spacing between adjacent B-scans
            scan_spacing=torch.tensor([self.y_spacing_mm], dtype=torch.float32),
        )

        return data

