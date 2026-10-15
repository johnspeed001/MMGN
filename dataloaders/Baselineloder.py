# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from einops import rearrange


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
    允许 2D mask(H,W)，自动在时间维复制。
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


def get_Baseline(data_dir, n_data, preprocessing, reduce_dim, task, sampling_rate):
    """
    与原函数签名保持一致；新增返回 mask（3D）。
    采样只在海域有效点上进行，避免把陆地区域（缺测/NaN→0）采进训练造成 NaN 或梯度异常。
    """
    dat = np.load(data_dir, allow_pickle=True)
    lat = torch.from_numpy(dat["lats"])
    lon = torch.from_numpy(dat["lons"])
    gst = torch.from_numpy(dat["temperature"][:n_data, :, :])  # (T,H,W)
    T, H, W = gst.shape

    # 新增：mask（若文件无 mask 字段就全 1）
    mask_3d = _load_mask(dat, T, H, W)  # (T,H,W) bool
    del dat

    t = torch.linspace(1e-10, 1.0, n_data)

    # ==================== 预处理 ====================
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

    # 新增：明确告诉下游图像尺寸
    normalizer["shape_hw"] = (int(lat.numel()), int(lon.numel()))  # e.g. (192, 288)

    # ==================== 只在海域采样 ====================
    # 将时间上的有效性合并为 (H,W) 一张有效性图（某个时间为海即视为有效）
    valid2d = mask_3d.any(dim=0)                 # (H,W) bool
    valid_flat = valid2d.reshape(-1).nonzero(as_tuple=False).squeeze(1)  # [N_valid]

    def sample_from_valid(n_points: int):
        n_points = min(n_points, valid_flat.numel())
        # 如果 n_points==0（极端小采样率或全陆地），回退至少采 1 个（避免下游整形报错）
        n_points = max(n_points, 1)
        perm = torch.randperm(valid_flat.numel())
        return valid_flat[perm[:n_points]]

    HW = H * W

    # 保持原作者四种 task 的接口与“长度对齐”风格（task2 / task4）
    if task == "task1":
        n_points = int(np.round(sampling_rate[0] * HW))
        base = sample_from_valid(n_points)             # 固定位置
        indices = base.repeat(T, 1)

    elif task == "task2":
        rates = np.random.uniform(sampling_rate[0], sampling_rate[1], T)
        n_list = [int(np.round(r * HW)) for r in rates]
        max_n_points = max(1, min(max(n_list), valid_flat.numel()))
        base = sample_from_valid(max_n_points)         # 固定位置的全集
        ind_lst = []
        for npt in n_list:
            npt = max(1, min(npt, max_n_points))
            # 复用原项目“重复拼接后截断”的思路，保证每个时间步长度对齐
            if sampling_rate[0] == 0.01:
                rep = 5
            else:
                rep = 2
            repeated = base[:npt].repeat(rep)
            ind_lst.append(repeated[:max_n_points][None, :])
        indices = torch.cat(ind_lst, dim=0)            # (T, max_n_points)

    elif task == "task3":
        n_points = int(np.round(sampling_rate[0] * HW))
        n_points = max(1, n_points)
        ind_lst = []
        for _ in range(T):
            one = sample_from_valid(n_points)
            ind_lst.append(one[None, :])
        indices = torch.cat(ind_lst, dim=0)            # 每个时间步随机位置，但等长

    elif task == "task4":
        rates = np.random.uniform(sampling_rate[0], sampling_rate[1], T)
        n_list = [max(1, int(np.round(r * HW))) for r in rates]
        max_n_points = min(max(n_list), valid_flat.numel())
        ind_lst = []
        for npt in n_list:
            one = sample_from_valid(npt)
            # 用重复拼接把长度补到 max_n_points，保证后续 reshape 一致
            if sampling_rate[0] == 0.01:
                rep = 5
            else:
                rep = 2
            repeated = one.repeat(rep)
            ind_lst.append(repeated[:max_n_points][None, :])
        indices = torch.cat(ind_lst, dim=0)            # (T, max_n_points)

    else:
        raise ValueError(f"Unknown task: {task}")

    return grid_lat, grid_lon, t, gst, indices, mask_3d, normalizer


def assemble_xy(grid_lat, grid_lon, t, gst, indices, mask_3d):
    """
    将 (T,H,W) 展平到采样点，并为每个点随机抽取一个时间步，
    同时返回对应 mask，方便上游做 masked-loss/metric。
    """
    n_data = gst.shape[0]

    grid_lat = grid_lat.reshape(-1)    # (H*W,)
    grid_lon = grid_lon.reshape(-1)
    indices  = indices.reshape(-1)     # (T * P,)

    # shuffle 全部索引
    idx = torch.randperm(indices.numel())
    indices = indices[idx]

    grid_lat = grid_lat[indices]
    grid_lon = grid_lon[indices]

    # 每个点随机抽一个时间步
    idx_time = torch.randint(0, t.numel(), (grid_lat.numel(),))
    t_vec = t[idx_time]

    gst_flat  = rearrange(gst,     'n h w -> n (h w)')
    mask_flat = rearrange(mask_3d, 'n h w -> n (h w)').bool()

    gst_vec  = gst_flat[idx_time, indices]                      # (Npoints,)
    mask_vec = mask_flat[idx_time, indices].to(torch.float32)   # (Npoints,)

    # 回到 (n_data, P)
    P = indices.numel() // n_data
    grid_lat = grid_lat.reshape(n_data, P)
    grid_lon = grid_lon.reshape(n_data, P)
    t_vec    = t_vec.reshape(n_data, P)
    gst_vec  = gst_vec.reshape(n_data, P)
    mask_vec = mask_vec.reshape(n_data, P)

    return grid_lat, grid_lon, t_vec, gst_vec, mask_vec


class Baseline(Dataset):
    def __init__(self, temperature, grid_lat, grid_lon, time_t, indices, mask_3d):
        self.grid_lat, self.grid_lon, self.time_t, self.temperature, self.mask = \
            assemble_xy(grid_lat, grid_lon, time_t, temperature, indices, mask_3d)
        # 释放外部变量引用
        del grid_lat, grid_lon, time_t, temperature, indices, mask_3d

    def __len__(self):
        return self.temperature.shape[0]

    def __getitem__(self, idx):
        lat  = self.grid_lat[idx]
        lon  = self.grid_lon[idx]
        t    = self.time_t[idx]
        temp = self.temperature[idx]
        m    = self.mask[idx]
        # input: (P,3)  output: (P,1)  mask: (P,1)
        input_x = torch.cat((lat.unsqueeze(1), lon.unsqueeze(1), t.unsqueeze(1)), dim=1)
        output_y = temp.unsqueeze(1)
        return input_x, output_y, m.unsqueeze(1)


class vizBaseline(Dataset):
    """
    可视化/验证集：整张网格（不随机采点），同样带 mask。
    """
    def __init__(self, temperature, grid_lat, grid_lon, time_t, mask_3d):
        self.temperature = temperature
        self.grid_lat = grid_lat
        self.grid_lon = grid_lon
        self.time_t   = time_t
        self.mask_3d  = mask_3d.bool()

    def __len__(self):
        return len(self.temperature)

    def __getitem__(self, idx):
        lat = self.grid_lat
        lon = self.grid_lon
        t   = torch.ones_like(lat) * self.time_t[idx]
        temp = self.temperature[idx]
        mask = self.mask_3d[idx].to(torch.float32)
        input_x = torch.cat((lat.reshape(-1, 1), lon.reshape(-1, 1), t.reshape(-1, 1)), dim=1)
        output_y = temp.reshape(-1, 1)
        return input_x, output_y, mask.reshape(-1, 1)


def _flat_to_hw(idx_flat, W):
    return torch.stack([idx_flat // W, idx_flat % W], dim=-1).long()

class LNOTrain(Dataset):
    def __init__(self, temperature, indices, mask_3d, H: int, W: int):
        self.temperature = temperature      # (T,H,W)
        self.indices     = indices          # (T,P)
        self.mask_3d     = mask_3d          # (T,H,W) bool
        self.H, self.W   = int(H), int(W)

    def __len__(self): return self.temperature.shape[0]

    def __getitem__(self, t_idx):
        temp = self.temperature[t_idx].reshape(-1)
        mask = self.mask_3d[t_idx].reshape(-1).float()
        idx_flat = self.indices[t_idx]                   # (P,)
        obs_val  = temp[idx_flat].unsqueeze(1)           # (P,1)
        mask_val = mask[idx_flat].unsqueeze(1)           # (P,1)
        coords_idx = _flat_to_hw(idx_flat, self.W)       # (P,2)
        obs_idx    = coords_idx
        return coords_idx, obs_idx, obs_val, self.H, self.W, obs_val, mask_val

class LNOViz(Dataset):
    def __init__(self, temperature, indices, mask_3d, H: int, W: int, use_ratio: float = 1.0):
        self.temperature = temperature
        self.indices     = indices
        self.mask_3d     = mask_3d
        self.H, self.W   = int(H), int(W)
        self.use_ratio   = float(use_ratio)
        HW = H * W
        self.all_coords = torch.arange(HW)

    def __len__(self): return self.temperature.shape[0]

    def __getitem__(self, t_idx):
        temp = self.temperature[t_idx].reshape(-1)
        mask = self.mask_3d[t_idx].reshape(-1).float()
        coords_flat = self.all_coords
        coords_idx  = _flat_to_hw(coords_flat, self.W)   # (H*W,2)
        y_coords    = temp.unsqueeze(1)                  # (H*W,1)
        m_coords    = mask.unsqueeze(1)                  # (H*W,1)

        idx_flat = self.indices[t_idx]
        if self.use_ratio < 1.0:
            k = max(1, int(len(idx_flat) * self.use_ratio))
            idx_flat = idx_flat[:k]
        obs_val = temp[idx_flat].unsqueeze(1)            # (P,1)
        obs_idx = _flat_to_hw(idx_flat, self.W)          # (P,2)
        return coords_idx, obs_idx, obs_val, self.H, self.W, y_coords, m_coords
