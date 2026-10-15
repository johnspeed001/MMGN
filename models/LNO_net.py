# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from models.lno_core.lno_2d_core import LNO2DCore

def _scatter_points_to_grid(H, W, idx_hw, values_bN1):
    B, N, _ = idx_hw.shape
    grid = torch.zeros(B, 1, H, W, device=values_bN1.device, dtype=values_bN1.dtype)
    b = torch.arange(B, device=values_bN1.device)[:, None].expand(B, N)
    h = idx_hw[..., 0].long().clamp_(0, H-1)
    w = idx_hw[..., 1].long().clamp_(0, W-1)
    # 如有重复观测点，做累加平均更稳：
    grid.index_put_((b, torch.zeros_like(b), h, w), values_bN1[..., 0], accumulate=True)
    return grid

def _gather_points_from_grid(grid_b1HW, idx_hw):
    B, _, H, W = grid_b1HW.shape
    b = torch.arange(B, device=grid_b1HW.device)[:, None].expand_as(idx_hw[...,0])
    h = idx_hw[...,0].long().clamp_(0, H-1)
    w = idx_hw[...,1].long().clamp_(0, W-1)
    return grid_b1HW[b, 0, h, w][..., None]  # (B,N,1)

class LNO2D(nn.Module):
    """
    forward(coords_idx, obs_idx, obs_val, H, W, mask_idx=None)
      -> pred_at_coords (B,N,1), pred_grid (B,1,H,W)
    """
    def __init__(self, in_ch=2, out_ch=1, width=64, modes1=4, modes2=4):
        super().__init__()
        self.core = LNO2DCore(in_ch=in_ch, out_ch=out_ch, width=width, modes1=modes1, modes2=modes2)

    @torch.no_grad()
    def _build_mask_grid(self, H, W, mask_idx):
        B, K, _ = mask_idx.shape
        mask = torch.zeros(B, 1, H, W, device=mask_idx.device)
        b = torch.arange(B, device=mask_idx.device)[:, None].expand(B, K)
        h = mask_idx[..., 0].long().clamp_(0, H-1)
        w = mask_idx[..., 1].long().clamp_(0, W-1)
        mask[b, 0, h, w] = 1.0
        return mask

    def forward(self, coords_idx, obs_idx, obs_val, H: int, W: int, mask_idx=None):
        obs_grid  = _scatter_points_to_grid(H, W, obs_idx, obs_val)                  # (B,1,H,W)
        mask_grid = self._build_mask_grid(H, W, obs_idx if mask_idx is None else mask_idx)
        x_in = torch.cat([obs_grid, mask_grid], dim=1)                                # (B,2,H,W)
        pred_grid = self.core(x_in)                                                   # (B,1,H,W)
        pred_at_coords = _gather_points_from_grid(pred_grid, coords_idx)              # (B,N,1)
        return pred_at_coords, pred_grid
