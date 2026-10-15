# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
import numpy as np
import math

from typing import Any
from omegaconf import DictConfig
from einops import rearrange
from models.ResMLP_net import ResMLP
from models.SIREN_net import SirenNet
from models.FFN_net import FFNet
from models.MMGNet_net import MMGNet
from models.MMGN_NEW import MMGN_NEW
from models.LNO_net import LNO2D
from models.MARE_net import MARE
from utils import (get_optimizer, get_scheduler, get_loss, toNumpy)

from torchmetrics.regression import MeanSquaredError
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# build model
# ---------------------------------------------------------
def get_model(cfg):
    """
    Set model.
    Args:
        cfg: Model configuration.
    Returns:
        Model will be use for modeling.
    """
    if cfg.name == "MMGN":
        model = MMGNet(
            cfg.in_size,
            cfg.hidden_size,
            cfg.n_train,
            cfg.latent_size,
            cfg.latent_init,
            cfg.out_size,
            cfg.n_layers,
            cfg.input_scale,
            cfg.alpha,
            cfg.filter
        )
    elif cfg.name == "ResMLP":
        model = ResMLP(
            res_in_dim=cfg.res_in_dim,
            res_out_dim=cfg.res_out_dim,
            res_width=cfg.res_width,
            res_depth=cfg.res_depth,
            net_act=cfg.net_act
        )
    elif cfg.name == "SIREN":
        model = SirenNet(
            dim_in=cfg.dim_in,
            dim_hidden=cfg.dim_hidden,
            dim_out=cfg.dim_out,
            num_layers=cfg.num_layers,
            w0=cfg.w0,
            w0_initial=cfg.w0_initial,
            use_bias=cfg.use_bias,
            final_activation=cfg.final_activation  # 字符串，如 'swish'
        )
    elif cfg.name == "FFN":
        model = FFNet(
            cfg.encode_method,
            cfg.gauss_sigma,
            cfg.gauss_input_size,
            cfg.gauss_encoded_size,
            cfg.pos_freq_const,
            cfg.pos_freq_num,
            cfg.net_in,
            cfg.net_hidden,
            cfg.net_out,
            cfg.net_layers,
            cfg.net_act
        )
    elif cfg.name == "MMGN_NEW":
        model = MMGN_NEW(
            in_size=cfg.in_size,
            hidden_size=cfg.hidden_size,
            n_data=cfg.n_train,  # 和训练样本数一致
            latent_size=cfg.latent_size,
            latent_init=cfg.latent_init,
            out_size=cfg.out_size,
            n_layers=cfg.n_layers,
            input_scale=cfg.input_scale,
            alpha=cfg.alpha,
        filter       = getattr(cfg, "filter", "Mixed"),
        # v2 里新增的可选项
        use_layernorm    = getattr(cfg, "use_layernorm", True),
        layer_scale_init = getattr(cfg, "layer_scale_init", 1e-2),
        droppath         = getattr(cfg, "droppath", 0.0),
        use_film         = getattr(cfg, "use_film", True),
        latent_dropout   = getattr(cfg, "latent_dropout", 0.0),
        latent_proj      = getattr(cfg, "latent_proj", 0),
        mix_use_gabor    = getattr(cfg, "mix_use_gabor", True),
        mix_use_fourier  = getattr(cfg, "mix_use_fourier", True),
        mix_use_linear   = getattr(cfg, "mix_use_linear", True),
        )
    elif cfg.name == "LNO":
        model = LNO2D(
            in_ch=cfg.in_ch,
            out_ch=cfg.dim_out,
            width=cfg.width,
            modes1=getattr(cfg, "modes1", 4),
            modes2=getattr(cfg, "modes2", 4),
        )
    elif cfg.name == "MARE":
        model = MARE(
            dim_in=cfg.dim_in,  # 2 或 3（含 t）
            dim_hidden=cfg.dim_hidden,  # 例如 256 或 320
            dim_out=cfg.dim_out,  # 1（标量）或 2（u,v）
            num_layers_local=cfg.num_layers_local,
            num_layers_global=cfg.num_layers_global,
            num_bands=cfg.num_bands,  # 例如 8
            global_low_bands=cfg.global_low_bands,  # 例如 3
            use_siren=cfg.use_siren  # True/False
        )



    else:
        raise ValueError(f"Unknown model name: {cfg.name}")
    return model


def plotSample(yhat, yref, mask, dir_save, sample_name):
    """yhat/yref/mask: (H, W) numpy arrays; mask=1有效、0无效"""
    import numpy as _np
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgray")  # 无效区灰色
    plt.close("all")
    # 屏蔽无效像素
    if mask is not None:
        m = mask.astype(bool)
        yref_plot = _np.where(m, yref, _np.nan)
        yhat_plot = _np.where(m, yhat, _np.nan)
        vmin = _np.nanmin(yref_plot)
        vmax = _np.nanmax(yref_plot)
    else:
        yref_plot, yhat_plot = yref, yhat
        vmin = vmax = None
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax0.set_title("Reference")
    im0 = ax0.imshow(yref_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    ax0.set_xticks([]);
    ax0.set_yticks([])
    fig.colorbar(im0, ax=ax0)
    ax1.set_title("Prediction")
    im1 = ax1.imshow(yhat_plot, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_xticks([]);
    ax1.set_yticks([])
    fig.colorbar(im1, ax=ax1)
    plt.savefig(dir_save + "/" + sample_name + ".png", bbox_inches="tight")
    plt.close(fig)


# ----------------------- common utils -----------------------
_EPS = 1e-12


def _masked_l1(yhat, yref, mask):
    """yhat/yref: (B, N, 1) or (B, 1, H, W); mask 同形状（最后维度为1）"""
    if mask is None:
        return (yhat - yref).abs().mean()
    num = ((yhat - yref).abs() * mask).sum()
    den = mask.sum().clamp_min(_EPS)
    return num / den


def _masked_mse(yhat, yref, mask):
    if mask is None:
        return ((yhat - yref) ** 2).mean()
    num = (((yhat - yref) ** 2) * mask).sum()
    den = mask.sum().clamp_min(_EPS)
    return num / den


def _masked_images_for_ssim(yhat_img, yref_img, mask_img):
    """
    让无效像素在 pred=ref，从而对 SSIM “等价忽略”。
    输入形状: (B, 1, H, W)
    """
    if mask_img is None:
        return yhat_img, yref_img
    return yref_img * (1.0 - mask_img) + yhat_img * mask_img, yref_img


def _auto_hw_from_L(L, normalizer):
    # 0) 显式 shape 优先
    try:
        if isinstance(normalizer, dict) and ("shape_hw" in normalizer):
            h, w = normalizer["shape_hw"]
            if int(h) * int(w) == int(L):
                return int(h), int(w)
    except Exception:
        pass

    # 1) 其次尝试由 lat/lon 推断（兼容 zscore/minmax）
    try:
        lat_std = normalizer.get("lat", None)
        lon_std = normalizer.get("lon", None)
        lat_len = getattr(getattr(lat_std, "mean", None), "numel", lambda: None)() if lat_std is not None else None
        lon_len = getattr(getattr(lon_std, "mean", None), "numel", lambda: None)() if lon_std is not None else None
        if lat_len and lon_len and (lat_len * lon_len == L):
            return int(lat_len), int(lon_len)
    except Exception:
        pass

    # 2) 兜底：因子分解
    candidates = [(i, L // i) for i in range(1, int(L ** 0.5) + 1) if L % i == 0]
    h, w = max(candidates, key=lambda x: x[0])
    return h, w


# --------------------------------------
# Lightning Module for "Baseline" (ResMLP/SIREN/FFN)
# --------------------------------------
class BaselineModule(pl.LightningModule):
    def __init__(self, normalizer, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.cfg_model = params_model
        self.cfg_optim = params_optim
        self.cfg_scheduler = params_scheduler

        self.model = get_model(self.cfg_model)
        self.optimizer = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion = get_loss(self.cfg_optim.loss)  # 仅在无 mask 时使用

        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1

        self.validation_step_yhat = []
        self.validation_step_yref = []
        self.validation_step_mask = []

        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def step(self, batch: Any):
        """
        batch: (x, y[, mask])
          x    (B, N, 2/3)
          y    (B, N, 1)
          mask (B, N, 1), 1=有效
        """
        x, y = batch[:2]
        if len(batch) == 3:
            mask = batch[2]
        else:
            mask = None

        yhat = self.model(x)  # (B, N, 1)
        if mask is None:
            loss = self.criterion(yhat, y)
            mse = ((yhat - y) ** 2).mean()
        else:
            loss = _masked_l1(yhat, y, mask)
            mse = _masked_mse(yhat, y, mask)
        return loss, mse, yhat, y, mask

    def training_step(self, batch: Any, batch_idx: int):
        loss, mse, yhat, yref, mask = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse", mse, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, mse, yhat, yref, mask = self.step(batch)
        # 缓存到 epoch_end
        self.validation_step_yhat.append(yhat.detach())
        self.validation_step_yref.append(yref.detach())
        if mask is not None:
            self.validation_step_mask.append(mask.detach())
        return {"yref": yref, "yhat": yhat}

    def on_validation_epoch_end(self):
        yhats = torch.stack(self.validation_step_yhat)  # (nB, B, N, 1)
        yrefs = torch.stack(self.validation_step_yref)  # (nB, B, N, 1)
        masks = torch.stack(self.validation_step_mask) if len(self.validation_step_mask) > 0 else None

        L = yhats.shape[2]
        h, w = _auto_hw_from_L(L, self.normalizer)
        # reshape to (T, 1, H, W)
        yhats_img = rearrange(yhats, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        yrefs_img = rearrange(yrefs, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        if masks is not None:
            masks_img = rearrange(masks, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        else:
            masks_img = None

        # masked MSE
        val_mse = _masked_mse(yhats_img, yrefs_img, masks_img)
        self.log("validation/mse", val_mse, sync_dist=self.sync_dist)

        # PSNR（基于 masked MSE + 掩码内动态范围）
        if masks_img is not None:
            valid_vals = yrefs_img[masks_img.bool()]
        else:
            valid_vals = yrefs_img[torch.isfinite(yrefs_img)]

        q1 = torch.quantile(valid_vals, 0.01)
        q99 = torch.quantile(valid_vals, 0.99)
        data_range = (q99 - q1).clamp_min(_EPS)

        psnr = 10.0 * torch.log10((data_range ** 2) / val_mse.clamp_min(_EPS))
        self.log("validation/psnr", psnr, sync_dist=self.sync_dist)

        # SSIM：把无效像素置为“pred=ref”后再算
        yhat4ssim, yref4ssim = _masked_images_for_ssim(yhats_img, yrefs_img, masks_img)
        ssim = self.m_SSIM(yhat4ssim, yref4ssim)
        self.log("validation/ssim", ssim, sync_dist=self.sync_dist)

        # 可视化：最多三张，避免越界
        b_vis = min(3, yhats_img.shape[0])
        for idx in range(b_vis):
            plotSample(
                toNumpy(torch.squeeze(yhats_img[idx, :, :, :])),
                toNumpy(torch.squeeze(yrefs_img[idx, :, :, :])),
                toNumpy(torch.squeeze(masks_img[idx, :, :, :])) if masks_img is not None else None,
                self.cfg_model.save_dir,
                f"val_epoch_{self.current_epoch}_idx_{idx}"
            )

        # 清空缓存
        self.validation_step_yhat.clear()
        self.validation_step_yref.clear()
        self.validation_step_mask.clear()

    def test_step(self, batch: Any, batch_idx: int):
        loss, mse, yhat, yref, mask = self.step(batch)
        self.log("test/mse", mse)
        # 可选：保存原始点格式的预测/真值
        np.save(self.cfg_model.save_dir + "/predictions.npy", toNumpy(yhat))
        np.save(self.cfg_model.save_dir + "/targets.npy", toNumpy(yref))
        try:
            if mask is not None:
                np.save(self.cfg_model.save_dir + "/predictions_masked.npy", toNumpy(yhat * mask))
        except Exception:
            pass

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


# --------------------------------------
# Lightning Module for "MMGNet"
# --------------------------------------
class MMGNetModule(pl.LightningModule):
    def __init__(self, normalizer, params_data: DictConfig, params_model: DictConfig, params_optim: DictConfig,
                 params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.cfg_data = params_data
        self.cfg_model = params_model
        self.cfg_optim = params_optim
        self.cfg_scheduler = params_scheduler

        self.cfg_model.n_train = self.cfg_data.n_train_val[0]
        self.model = get_model(self.cfg_model)
        self.optimizer = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion = get_loss(self.cfg_optim.loss)

        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1

        self.validation_step_yhat = []
        self.validation_step_yref = []
        self.validation_step_mask = []

        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def step(self, batch: Any):
        """
        batch: (x, y, idx[, mask])
          x    (B, N, 2)
          y    (B, N, 1)
          idx  (B,)
          mask (B, N, 1), 1=有效
        """
        x, y, idx = batch[:3]
        if len(batch) == 4:
            mask = batch[3]
        else:
            mask = None

        yhat = self.model(x, idx)
        if mask is None:
            loss = self.criterion(yhat, y)
            mse = ((yhat - y) ** 2).mean()
        else:
            loss = _masked_l1(yhat, y, mask)
            mse = _masked_mse(yhat, y, mask)

        return loss, mse, yhat, y, mask

    def training_step(self, batch: Any, batch_idx: int):
        loss, mse, yhat, yref, mask = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse", mse, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, mse, yhat, yref, mask = self.step(batch)
        self.validation_step_yhat.append(yhat.detach())
        self.validation_step_yref.append(yref.detach())
        if mask is not None:
            self.validation_step_mask.append(mask.detach())
        return {"yref": yref, "yhat": yhat}

    def on_validation_epoch_end(self):
        yhats = torch.stack(self.validation_step_yhat)  # (nB, B, N, 1)
        yrefs = torch.stack(self.validation_step_yref)  # (nB, B, N, 1)
        masks = torch.stack(self.validation_step_mask) if len(self.validation_step_mask) > 0 else None

        L = yhats.shape[2]
        h, w = _auto_hw_from_L(L, self.normalizer)
        yhats_img = rearrange(yhats, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        yrefs_img = rearrange(yrefs, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        if masks is not None:
            masks_img = rearrange(masks, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        else:
            masks_img = None

        # masked MSE
        val_mse = _masked_mse(yhats_img, yrefs_img, masks_img)
        self.log("validation/mse", val_mse, sync_dist=self.sync_dist)

        # PSNR（基于 masked MSE + 掩码内动态范围）
        if masks_img is not None:
            valid = masks_img.bool()
            minv = yrefs_img[valid].min()
            maxv = yrefs_img[valid].max()
        else:
            minv = yrefs_img.min()
            maxv = yrefs_img.max()
        data_range = (maxv - minv).detach().clamp_min(_EPS)
        psnr = 10.0 * torch.log10((data_range ** 2) / val_mse.clamp_min(_EPS))
        self.log("validation/psnr", psnr, sync_dist=self.sync_dist)

        # SSIM
        yhat4ssim, yref4ssim = _masked_images_for_ssim(yhats_img, yrefs_img, masks_img)
        ssim = self.m_SSIM(yhat4ssim, yref4ssim)
        self.log("validation/ssim", ssim, sync_dist=self.sync_dist)

        # 可视化
        b_vis = min(3, yhats_img.shape[0])
        for idx in range(b_vis):
            plotSample(
                toNumpy(torch.squeeze(yhats_img[idx, :, :, :])),
                toNumpy(torch.squeeze(yrefs_img[idx, :, :, :])),
                toNumpy(torch.squeeze(masks_img[idx, :, :, :])) if masks_img is not None else None,
                self.cfg_model.save_dir,
                f"val_epoch_{self.current_epoch}_idx_{idx}"
            )

        self.validation_step_yhat.clear()
        self.validation_step_yref.clear()
        self.validation_step_mask.clear()

    def test_step(self, batch: Any, batch_idx: int):
        loss, mse, yhat, yref, mask = self.step(batch)
        self.log("test/mse", mse)
        np.save(self.cfg_model.save_dir + "/predictions.npy", toNumpy(yhat))
        np.save(self.cfg_model.save_dir + "/targets.npy", toNumpy(yref))
        try:
            if mask is not None:
                np.save(self.cfg_model.save_dir + "/predictions_masked.npy", toNumpy(yhat * mask))
        except Exception:
            pass

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

def _as_pyint(v):
    # 安全把任意类型的 H/W 转为 python int
    # 支持 int、0D/1D torch.Tensor、numpy 数字等
    try:
        import torch
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, torch.Tensor):
            if v.numel() == 1:
                return int(v.item())
            else:
                return int(v.view(-1)[0].item())
        # 兜底
        return int(v)
    except Exception:
        # 万一是奇怪类型，直接拿第一个元素
        return int(v[0])

class LNOModule(pl.LightningModule):
    def __init__(self, normalizer, params_model: DictConfig, params_optim: DictConfig, params_scheduler: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_model = params_model
        self.cfg_optim = params_optim
        self.cfg_scheduler = params_scheduler
        self.model     = get_model(self.cfg_model)
        self.optimizer = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion = get_loss(self.cfg_optim.loss)
        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1
        self.validation_step_yhat, self.validation_step_yref, self.validation_step_mask = [], [], []
        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def training_step(self, batch, batch_idx):
        coords_idx, obs_idx, obs_val, H, W, y_coords, mask_coords = batch
        for t in [coords_idx, obs_idx, obs_val, y_coords, mask_coords]:
            if t.ndim == 2: t.unsqueeze_(0)
        H_int, W_int = _as_pyint(H), _as_pyint(W)
        yhat_at_coords, _ = self.model(coords_idx, obs_idx, obs_val, H_int, W_int)
        loss = _masked_l1(yhat_at_coords, y_coords, mask_coords)
        mse  = _masked_mse(yhat_at_coords, y_coords, mask_coords)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse",  mse,  on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        coords_idx, obs_idx, obs_val, H, W, y_coords, mask_coords = batch
        for t in [coords_idx, obs_idx, obs_val, y_coords, mask_coords]:
            if t.ndim == 2: t.unsqueeze_(0)
        H_int, W_int = _as_pyint(H), _as_pyint(W)
        yhat_at_coords, _ = self.model(coords_idx, obs_idx, obs_val, H_int, W_int)
        self.validation_step_yhat.append(yhat_at_coords.detach())
        self.validation_step_yref.append(y_coords.detach())
        self.validation_step_mask.append(mask_coords.detach())
        return {"yhat": yhat_at_coords, "yref": y_coords}

    def on_validation_epoch_end(self):
        yhats = torch.stack(self.validation_step_yhat)
        yrefs = torch.stack(self.validation_step_yref)
        masks = torch.stack(self.validation_step_mask)
        val_mse = _masked_mse(yhats, yrefs, masks)
        self.log("validation/mse", val_mse, sync_dist=self.sync_dist)
        valid_vals = yrefs[masks.bool()]
        q1, q99 = torch.quantile(valid_vals, 0.01), torch.quantile(valid_vals, 0.99)
        data_range = (q99 - q1).clamp_min(_EPS)
        psnr = 10.0 * torch.log10((data_range ** 2) / val_mse.clamp_min(_EPS))
        self.log("validation/psnr", psnr, sync_dist=self.sync_dist)
        # SSIM 需还原到图片
        L = yhats.shape[2]; h,w = _auto_hw_from_L(L, self.normalizer)
        yh_img = rearrange(yhats, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        yr_img = rearrange(yrefs, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        mk_img = rearrange(masks, f'n1 n2 (h w) c -> (n1 n2) c h w', h=h, w=w)
        yhat4ssim, yref4ssim = _masked_images_for_ssim(yh_img, yr_img, mk_img)
        ssim = self.m_SSIM(yhat4ssim, yref4ssim); self.log("validation/ssim", ssim, sync_dist=self.sync_dist)
        # 可视化
        b_vis = min(3, yh_img.shape[0])
        for i in range(b_vis):
            plotSample(
                toNumpy(torch.squeeze(yh_img[i])),
                toNumpy(torch.squeeze(yr_img[i])),
                toNumpy(torch.squeeze(mk_img[i])),
                self.cfg_model.save_dir,
                f"LNO_val_epoch_{self.current_epoch}_idx_{i}"
            )
        self.validation_step_yhat.clear(); self.validation_step_yref.clear(); self.validation_step_mask.clear()

    def test_step(self, batch, batch_idx):
        coords_idx, obs_idx, obs_val, H, W, y_coords, mask_coords = batch
        for t in [coords_idx, obs_idx, obs_val, y_coords, mask_coords]:
            if t.ndim == 2: t.unsqueeze_(0)
        H_int, W_int = _as_pyint(H), _as_pyint(W)
        yhat_at_coords, _ = self.model(coords_idx, obs_idx, obs_val, H_int, W_int)
        mse = _masked_mse(yhat_at_coords, y_coords, mask_coords)
        self.log("test/mse", mse)
        np.save(self.cfg_model.save_dir + "/predictions.npy", toNumpy(yhat_at_coords))
        np.save(self.cfg_model.save_dir + "/targets.npy",     toNumpy(y_coords))
        return {"mse": mse}

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]