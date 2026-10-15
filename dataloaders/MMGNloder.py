# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset


class ZscoreStandardizer(object):
    """
    Normalization transformation
    if reduce_dim = [0]: The mean is computed over different time volumes.
    if reduce_dim = []:  The mean is computed over all data points.
    """
    def __init__(self, x, reduce_dim=[0]):
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze()
        self.std  = torch.std(x, reduce_dim, keepdim=True).squeeze()
        self.epsilon = 1e-10
        assert self.mean.shape == self.std.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.epsilon)

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.epsilon) + self.mean


class MinMaxStandardizer(object):
    """
    Min-Max transformation
    if reduce_dim = [0]: The min/max is computed over different time volumes.
    if reduce_dim = []:  The min/max is computed over all data points.
    """
    def __init__(self, x, reduce_dim=[0]):
        if reduce_dim:
            self.minVal = torch.min(x, reduce_dim[0], keepdim=True)[0].squeeze()
            self.maxVal = torch.max(x, reduce_dim[0], keepdim=True)[0].squeeze()
        else:
            self.minVal = torch.min(x)
            self.maxVal = torch.max(x)
        self.epsilon = 1e-10
        assert self.minVal.shape == self.maxVal.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        # 避免除零
        return (x - self.minVal) / (self.maxVal - self.minVal + self.epsilon)

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.maxVal - self.minVal + self.epsilon) + self.minVal


def _load_mask(dat, T, H, W):
    """
    读取并返回 3D 掩膜 (T,H,W)：海=1，陆=0。
    若 npz 无 mask，则返回全 1。
    允许 2D mask(H,W) 自动扩到 T。
    """
    if "mask" in dat.files:
        m = dat["mask"]
        if m.ndim == 2:
            m = np.broadcast_to(m[None, :, :], (T, H, W))
        elif m.ndim == 3:
            pass
        else:
            raise ValueError("mask must be 2D or 3D.")
        mask = torch.from_numpy(m).bool()
    else:
        mask = torch.ones((T, H, W), dtype=torch.bool)
    return mask


def get_MMGN(data_dir, n_data, preprocessing, reduce_dim, task, sampling_rate):
    # load data
    dat = np.load(data_dir, allow_pickle=True)
    lat = torch.from_numpy(dat["lats"])
    lon = torch.from_numpy(dat["lons"])
    gst = torch.from_numpy(dat["temperature"][:n_data, :, :])  # (T,H,W)
    T, H, W = gst.shape

    # 掩膜：海=1，陆=0（若无则全 1）
    mask_3d = _load_mask(dat, T, H, W)  # (T,H,W) bool
    del dat

    # pre-process data
    lat_lon_pre, gst_pre = preprocessing
    lat_lon_reduce_dim, gst_reduce_dim = reduce_dim
    normalizer = {}

    if lat_lon_pre == "zscore":
        normalizer["lat"] = ZscoreStandardizer(lat, lat_lon_reduce_dim)
        normalizer["lon"] = ZscoreStandardizer(lon, lat_lon_reduce_dim)
    elif lat_lon_pre == "minmax":
        normalizer["lat"] = MinMaxStandardizer(lat, lat_lon_reduce_dim)
        normalizer["lon"] = MinMaxStandardizer(lon, lat_lon_reduce_dim)
    else:
        raise ValueError(f"Unknown preprocessing for lat/lon: {lat_lon_pre}")

    if gst_pre == "zscore":
        normalizer["temperature"] = ZscoreStandardizer(gst, gst_reduce_dim)
    elif gst_pre == "minmax":
        normalizer["temperature"] = MinMaxStandardizer(gst, gst_reduce_dim)
    else:
        raise ValueError(f"Unknown preprocessing for gst: {gst_pre}")

    lat = normalizer["lat"].do(lat)
    lon = normalizer["lon"].do(lon)
    gst = normalizer["temperature"].do(gst)

    grid_lat, grid_lon = torch.meshgrid(lat, lon, indexing="ij")
    normalizer["shape_hw"] = (int(lat.numel()), int(lon.numel()))

    # ===== 只在海域采样 =====
    # 将时间维“或”在一起得到 (H,W) 有效点全集（某时刻为海就算有效）
    valid2d = mask_3d.any(dim=0)  # (H,W) bool
    valid_flat = valid2d.reshape(-1).nonzero(as_tuple=False).squeeze(1)  # [N_valid]
    HW = H * W

    def sample_from_valid(n_points: int):
        if valid_flat.numel() == 0:
            raise ValueError("No valid ocean pixels in mask.")
        n_points = max(1, min(n_points, valid_flat.numel()))
        perm = torch.randperm(valid_flat.numel())
        return valid_flat[perm[:n_points]]

    if task == "task1":
        # 保留原作者用 sampling_rate[1] 的习惯
        n_points = int(np.round(sampling_rate[1] * HW))
        base = sample_from_valid(n_points)     # 固定位置
        indices = base.repeat(T, 1)            # (T, P)

    elif task == "task2":
        rates = np.random.uniform(sampling_rate[0], sampling_rate[1], T)
        n_list = [max(1, int(np.round(r * HW))) for r in rates]
        max_n_points = min(max(n_list), valid_flat.numel())
        base = sample_from_valid(max_n_points)  # 固定位置全集
        ind_lst = []
        for npt in n_list:
            npt = max(1, min(npt, max_n_points))
            rep = 5 if sampling_rate[0] == 0.01 else 2
            repeated = base[:npt].repeat(rep)
            ind_lst.append(repeated[:max_n_points][None, :])
        indices = torch.cat(ind_lst, dim=0)     # (T, max_n_points)

    elif task == "task3":
        n_points = max(1, int(np.round(sampling_rate[1] * HW)))
        ind_lst = []
        for _ in range(T):
            one = sample_from_valid(n_points)
            ind_lst.append(one[None, :])
        indices = torch.cat(ind_lst, dim=0)     # (T, P)

    elif task == "task4":
        rates = np.random.uniform(sampling_rate[0], sampling_rate[1], T)
        n_list = [max(1, int(np.round(r * HW))) for r in rates]
        max_n_points = min(max(n_list), valid_flat.numel())
        ind_lst = []
        for npt in n_list:
            one = sample_from_valid(npt)
            rep = 5 if sampling_rate[0] == 0.01 else 2
            repeated = one.repeat(rep)
            ind_lst.append(repeated[:max_n_points][None, :])
        indices = torch.cat(ind_lst, dim=0)     # (T, max_n_points)

    else:
        raise ValueError(f"Unknown task: {task}")

    # 返回顺序基本保持原样，多返回一个 mask_3d 以便上游 masked loss/metric
    return grid_lat, grid_lon, gst, indices, mask_3d, normalizer


class MMGN(Dataset):
    def __init__(self,
        temperature: torch.Tensor,   # (T,H,W)
        grid_lat: torch.Tensor,      # (H,W)
        grid_lon: torch.Tensor,      # (H,W)
        indices: torch.Tensor,       # (T,P)
        mask_3d: torch.Tensor,       # (T,H,W) bool
    ):
        self.temperature = temperature
        self.grid_lat = grid_lat.reshape(-1)  # (H*W,)
        self.grid_lon = grid_lon.reshape(-1)
        self.indices = indices                # (T,P)
        self.mask_3d = mask_3d.bool()

    def __len__(self):
        return len(self.temperature)          # = T

    def __getitem__(self, idx):
        # idx 表示时间步
        ind = self.indices[idx]               # (P,)
        lat = self.grid_lat[ind]              # (P,)
        lon = self.grid_lon[ind]              # (P,)

        temp = self.temperature[idx].reshape(-1)   # (H*W,)
        temp = temp[ind]                           # (P,)

        # 对应时间步的掩膜
        m = self.mask_3d[idx].reshape(-1)[ind].to(torch.float32)  # (P,)

        # input lat, lon | output temperature | idx for latent variables | mask
        input_x = torch.cat((lat.unsqueeze(1), lon.unsqueeze(1)), dim=1)  # (P,2)
        output_y = temp.unsqueeze(1)                                      # (P,1)
        return input_x, output_y, idx, m.unsqueeze(1)                     # ★ 新增 mask


class vizMMGN(Dataset):
    def __init__(self,
        temperature: torch.Tensor,   # (T,H,W)
        grid_lat: torch.Tensor,      # (H,W)
        grid_lon: torch.Tensor,      # (H,W)
        mask_3d: torch.Tensor,       # (T,H,W) bool
    ):
        self.temperature = temperature
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.mask_3d  = mask_3d.bool()

    def __len__(self):
        return len(self.temperature)

    def __getitem__(self, idx):
        lat  = self.grid_lat.reshape(-1, 1)
        lon  = self.grid_lon.reshape(-1, 1)
        temp = self.temperature[idx].reshape(-1, 1)
        mask = self.mask_3d[idx].reshape(-1, 1).to(torch.float32)
        input_x = torch.cat((lat, lon), dim=1)  # (H*W,2)
        output_y = temp                         # (H*W,1)
        return input_x, output_y, idx, mask     # ★ 新增 mask
