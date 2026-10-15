# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 工具
# ---------------------------
def exists(x): return x is not None
def default(x, d): return x if exists(x) else d

# ---------------------------
# SIREN 组件（本地高频分支）
# ---------------------------
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class SirenLayer(nn.Module):
    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False):
        super().__init__()
        self.is_first = is_first
        self.in_features = dim_in
        self.lin = nn.Linear(dim_in, dim_out)
        self.act = Sine(w0)
        self.init(c=c, w0=w0)

    def init(self, c, w0):
        dim = self.in_features
        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        nn.init.uniform_(self.lin.weight, -w_std, w_std)
        nn.init.uniform_(self.lin.bias,   -w_std, w_std)

    def forward(self, x):
        return self.act(self.lin(x))

class SirenMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden=256, dim_out=1, num_layers=4, w0=30.0, w0_first=30.0):
        super().__init__()
        layers = [SirenLayer(dim_in, dim_hidden, w0=w0_first, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(dim_hidden, dim_hidden, w0=w0))
        layers.append(nn.Linear(dim_hidden, dim_out))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------------------
# 多尺度 Fourier + 方向旋转编码（DirPE）
#   - 给定 (x,y[,t])，先把 (x,y) 旋转一个角度 θ(x,y)（由小 MLP 估计）后再做 Fourier 特征
#   - 同时做“层级门控”以模拟密度自适应（SD-Hash 的轻量替代）
# ---------------------------
class DirFourierEncoding(nn.Module):
    def __init__(self, dim_in=3, num_bands=8, base_freq=1.0, learn_theta=True, gate=True):
        """
        dim_in: 2 或 3（x,y[,t]）
        num_bands: 频带数（sin/cos 成对输出）
        """
        super().__init__()
        assert dim_in in (2, 3)
        self.has_t = (dim_in == 3)
        self.num_bands = num_bands
        self.base = base_freq
        # 方向角 θ(x,y)
        self.learn_theta = learn_theta
        if learn_theta:
            self.theta_mlp = nn.Sequential(
                nn.Linear(2, 32), nn.ReLU(True),
                nn.Linear(32, 32), nn.ReLU(True),
                nn.Linear(32, 1)
            )
        # 频带门控
        self.gate = gate
        if gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(dim_in, 64), nn.ReLU(True),
                nn.Linear(64, num_bands), nn.Sigmoid()
            )

    def forward(self, x):  # x: (B,N,2/3), 归一化到 [-1,1] 最佳
        xy = x[..., :2]                                  # (B,N,2)
        if self.learn_theta:
            theta = self.theta_mlp(xy)                   # (B,N,1)
            c, s = torch.cos(theta), torch.sin(theta)
            # 旋转 (x,y)
            xr =  c * xy[..., :1] + s * xy[..., 1:]
            yr = -s * xy[..., :1] + c * xy[..., 1:]
            xy_r = torch.cat([xr, yr], dim=-1)
        else:
            xy_r = xy

        # 频带
        device = x.device
        k = torch.arange(self.num_bands, device=device).float()
        freq = (2.0 ** k) * self.base                     # (K,)

        # sin/cos 特征
        # (B,N,2) * (K,) -> (B,N,2,K)
        angles = xy_r.unsqueeze(-1) * freq               # (B,N,2,K)
        sincos = torch.cat([torch.sin(math.pi * angles),
                             torch.cos(math.pi * angles)], dim=2) # (B,N,4,K)

        feats_xy = sincos.flatten(2,3)                   # (B,N,4K)

        if self.has_t:
            t = x[..., 2:3]
            angles_t = t * freq.view(1,1,-1)            # (B,N,K)
            feats_t = torch.cat([torch.sin(math.pi * angles_t),
                                 torch.cos(math.pi * angles_t)], dim=-1)  # (B,N,2K)
            feats = torch.cat([feats_xy, feats_t], dim=-1)                 # (B,N,6K)
        else:
            feats = feats_xy                             # (B,N,4K)

        # 频带门控：按点自适应选择频带强度
        if self.gate:
            g = self.gate_mlp(x)                         # (B,N,K) in [0,1]
            # 把 g 复制到 sin/cos 与 (x,y)/(t) 两套通道
            if self.has_t:
                g_full = torch.cat([g.repeat(1,1,4), g.repeat(1,1,2)], dim=-1)  # (B,N,6K)
            else:
                g_full = g.repeat(1,1,4)                                         # (B,N,4K)
            feats = feats * g_full

        return feats  # 编码后的特征

    @property
    def out_dim(self):
        return self.num_bands * (6 if hasattr(self, 'has_t') and self.has_t else 4)

# ---------------------------
# 轻量“时间状态”→ FiLM 调制（Tiny-SSM 代理）
#   - 从 t（或全局平均 t）得到 γ, β，调制后续 MLP
# ---------------------------
class TimeFiLM(nn.Module):
    def __init__(self, use_time=True, hidden=128):
        super().__init__()
        self.use_time = use_time
        if use_time:
            self.mlp = nn.Sequential(
                nn.Linear(1, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 2)  # gamma, beta
            )
        else:
            self.mlp = None

    def forward(self, x):   # x: (B,N,3/2)
        if (not self.use_time) or (x.size(-1) < 3):
            gamma = beta = None
        else:
            t = x[..., 2:3]
            t_bar = t.mean(dim=1, keepdim=True)  # (B,1,1)
            gb = self.mlp(t_bar.squeeze(1))     # (B,2)
            gamma, beta = gb[:, :1], gb[:, 1:]
            gamma = gamma.unsqueeze(-1)         # (B,1,1)
            beta  = beta.unsqueeze(-1)          # (B,1,1)
        return gamma, beta

def film(x, gamma, beta):
    if gamma is None or beta is None:
        return x
    return x * (1 + gamma) + beta

# ---------------------------
# 全局（低频）+ 本地（高频）双分支 & 门控
# ---------------------------
class MARE(nn.Module):
    """
    输入: x (B,N,2/3), idx 可忽略（保接口）
    输出: y (B,N,dim_out)
    """
    def __init__(self,
                 dim_in=3,
                 dim_hidden=256,
                 dim_out=1,
                 num_layers_local=4,
                 num_layers_global=2,
                 num_bands=8,
                 global_low_bands=3,   # 全局通道只用前几个低频
                 use_siren=True):
        super().__init__()
        self.enc = DirFourierEncoding(dim_in=dim_in, num_bands=num_bands, base_freq=1.0,
                                      learn_theta=True, gate=True)
        enc_dim = self.enc.out_dim

        # 局部分支（高频）：SIREN 或 GELU-MLP
        if use_siren:
            self.local = SirenMLP(enc_dim, dim_hidden, dim_out, num_layers=num_layers_local,
                                  w0=30.0, w0_first=30.0)
        else:
            self.local = nn.Sequential(
                nn.Linear(enc_dim, dim_hidden), nn.GELU(),
                *sum([[nn.Linear(dim_hidden, dim_hidden), nn.GELU()] for _ in range(num_layers_local-2)], []),
                nn.Linear(dim_hidden, dim_out)
            )

        # 全局分支（低频）：只取前 global_low_bands 个频带做一个轻量 MLP
        g_in = (6 if dim_in==3 else 4) * global_low_bands
        self.global_proj = nn.Linear(enc_dim, g_in)  # 选低频用线性投影近似“抽取”
        g_layers = [nn.Linear(g_in, dim_hidden), nn.ReLU(True)]
        for _ in range(num_layers_global-2):
            g_layers += [nn.Linear(dim_hidden, dim_hidden), nn.ReLU(True)]
        g_layers += [nn.Linear(dim_hidden, dim_out)]
        self.global_net = nn.Sequential(*g_layers)

        # 门控
        self.gate = nn.Sequential(
            nn.Linear(enc_dim, dim_hidden//2), nn.GELU(),
            nn.Linear(dim_hidden//2, 1), nn.Sigmoid()
        )

        # 时间调制（FiLM）
        self.time_mod = TimeFiLM(use_time=(dim_in==3), hidden=128)

    def forward(self, x, idx=None):  # x: (B,N,2/3)
        feats = self.enc(x)                           # (B,N,F)
        gamma, beta = self.time_mod(x)                # (B,1,1) or None

        # 分支
        y_local  = self.local(film(feats, gamma, beta))     # (B,N,1)
        y_global = self.global_net(self.global_proj(feats))  # (B,N,1)

        g = self.gate(feats)                          # (B,N,1) in [0,1]
        y = g * y_local + (1.0 - g) * y_global
        return y
