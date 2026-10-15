import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from functools import partial
from torch.nn import init
from torch.nn.parameter import Parameter

# ---------------------------------------------------------
# Swish function
# ---------------------------------------------------------
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

# ---------------------------------------------------------
# non-linear activation set
# ---------------------------------------------------------
nl_s = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'elu': partial(nn.ELU)}

# ---------------------------------------------------------
# initialization methods set
# ---------------------------------------------------------
init_s = {"uniform": partial(torch.nn.init.uniform_),
          "normal": partial(torch.nn.init.normal_),
          "ones": partial(torch.nn.init.ones_),
          "zeros": partial(torch.nn.init.zeros_),
          "xavier_u": partial(torch.nn.init.xavier_uniform_),
          "xavier_n": partial(torch.nn.init.xavier_normal_),
          "kaiming_u": partial(torch.nn.init.kaiming_uniform_),
          "kaiming_n": partial(torch.nn.init.kaiming_normal_),
          "trunc_n": partial(torch.nn.init.trunc_normal_),
          "orthogonal": partial(torch.nn.init.orthogonal_)}

# ---------------------------------------------------------
# linear transformation fusing coordinates and latent information
# (与原版一致; 注意：深层处把 "coord" 换成上层特征)
# ---------------------------------------------------------
class FuseLinear(nn.Module):
    def __init__(self, d_coord: int, d_latent: int, d_out: int) -> None:
        super(FuseLinear, self).__init__()
        self.b    = d_coord
        self.A    = Parameter(torch.empty(d_out, d_coord))
        self.B    = Parameter(torch.empty(d_out, d_latent))
        self.bias = Parameter(torch.empty(d_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.b)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, coord: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        # coord: (B, P, C) or (B, P, H)
        # latent: (B, L)
        # out: (B, P, H_out)
        linear_coord  = torch.einsum('bpi,oi->bpo', coord, self.A)
        linear_latent = torch.einsum('bj,oj->bo', latent, self.B).unsqueeze(1)
        return linear_coord + linear_latent + self.bias

# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x: (B, P, C)
        return self.linear(x)

# ---------------------------------------------------------
# Fourier layer
# ---------------------------------------------------------
class FourierLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale):
        """
        注意：out_features 是“cos/sin 前的一半”，最终输出通道数 = 2 * out_features
        """
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # x: (B, P, C), out: (B, P, 2*out_features)
        proj = F.linear(x, self.weight * self.weight_scale)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

# ---------------------------------------------------------
# Gabor layer
# ---------------------------------------------------------
class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,)))
        # 按论文做频率/带宽缩放
        with torch.no_grad():
            self.linear.weight.mul_(weight_scale * torch.sqrt(self.gamma[:, None]))
            self.linear.bias.uniform_(-np.pi, np.pi)

    def forward(self, x):
        # x: (B, P, C) -> (B, P, H)
        D = (x ** 2).sum(-1)[..., None] + (self.mu ** 2).sum(-1)[None, :] - 2 * x @ self.mu.T
        return torch.sin(self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])

# ---------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep)
        return x * mask / keep

# ---------------------------------------------------------
# MixedFilter：每层对 (Gabor/Fourier/Linear) 做潜码门控的加权混合
# ---------------------------------------------------------
class MixedFilter(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_scale: float,
        alpha: float,
        use_gabor: bool = True,
        use_fourier: bool = True,
        use_linear: bool = True,
        latent_size: int = 0,
        gate_hidden: int = 64
    ):
        super().__init__()
        self.branches = nn.ModuleDict()
        self.branch_keys = []

        if use_gabor:
            self.branches["gabor"] = GaborLayer(in_features, out_features, weight_scale, alpha/(1.0), beta=1.0)
            self.branch_keys.append("gabor")

        if use_fourier:
            # Fourier 输出通道是 2*k，这里让 k = out_features // 2 以对齐 H
            self.branches["fourier"] = FourierLayer(in_features, out_features // 2, weight_scale)
            self.branch_keys.append("fourier")

        if use_linear:
            self.branches["linear"] = LinearLayer(in_features, out_features)
            self.branch_keys.append("linear")

        self.n_branch = len(self.branch_keys)
        assert self.n_branch > 0, "MixedFilter must enable at least one branch."

        # 门控：由潜码 -> 分支权重，softmax 保证和为 1
        if latent_size > 0 and self.n_branch > 1:
            self.gate = nn.Sequential(
                nn.Linear(latent_size, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.n_branch)
            )
        else:
            self.gate = None

    def forward(self, x, latent=None):
        # x: (B, P, C) -> 每个分支输出 (B, P, H)
        outs = []
        for k in self.branch_keys:
            outs.append(self.branches[k](x))
        if self.n_branch == 1:
            return outs[0]

        if (latent is None) or (self.gate is None):
            # 默认平均
            out = torch.stack(outs, dim=0).mean(0)
        else:
            # 权重: (B, n_branch) -> (B, 1, n_branch) 以广播到 P
            w = F.softmax(self.gate(latent), dim=-1).unsqueeze(1)
            # 堆叠: (n_branch, B, P, H) -> (B, P, H)
            stack = torch.stack(outs, dim=-1)         # (B, P, H, n_branch)
            out = (stack * w.unsqueeze(2)).sum(-1)    # (B, P, H)
        return out

# ---------------------------------------------------------
# FiLM 调制模块: from latent -> (gamma, beta)
# ---------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, latent_size, hidden_size, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hidden_size)
        )

    def forward(self, latent):
        # latent: (B, L) -> gamma, beta: (B, H)
        gamma, beta = self.net(latent).chunk(2, dim=-1)
        return gamma, beta

# ---------------------------------------------------------
# Multiplicative and Modulated Gabor Network (v2)
#  - filter="Mixed" 启用混合滤波 + 潜码门控
#  - 其它 filter 值与旧版保持完全兼容
# ---------------------------------------------------------
class MMGN_NEW(nn.Module):
    def __init__(
            self,
            in_size: int = 2,
            hidden_size: int = 300,
            n_data: int = 2,
            latent_size: int = 370,
            latent_init: str = "zeros",
            out_size: int = 1,
            n_layers: int = 7,
            input_scale: float = 370.0,
            alpha: float = 1.0,
            filter: str = "Gabor",          # "Gabor" | "Fourier" | "Linear" | "Mixed"
            # === v2 新增，可选项（均有默认，保持兼容）===
            use_layernorm: bool = True,
            layer_scale_init: float = 1e-2,
            droppath: float = 0.0,
            use_film: bool = True,
            latent_dropout: float = 0.0,
            latent_proj: int = 0,           # >0 时把潜码降维到该大小后再用
            mix_use_gabor: bool = True,
            mix_use_fourier: bool = True,
            mix_use_linear: bool = True
        ):
        super(MMGN_NEW, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.n_layers = n_layers
        self.filter_type = filter
        self.use_layernorm = use_layernorm
        self.use_film = use_film

        # 潜码表
        self.latents = nn.Parameter(torch.FloatTensor(n_data, latent_size))
        self.latents = init_s[latent_init](self.latents)
        self.latent_dp = nn.Dropout(latent_dropout) if latent_dropout > 0 else nn.Identity()
        if latent_proj and latent_proj > 0:
            self.latent_proj = nn.Linear(latent_size, latent_proj)
            latent_size_eff = latent_proj
        else:
            self.latent_proj = nn.Identity()
            latent_size_eff = latent_size

        # 逐层的双线性融合（第一层“特征”输入用 0 张量对齐原实现）
        self.bilinear = nn.ModuleList(
            [FuseLinear(in_size, latent_size_eff, hidden_size)] +
            [FuseLinear(hidden_size, latent_size_eff, hidden_size) for _ in range(int(n_layers))]
        )

        # 过滤器/编码器
        if filter == "Gabor":
            self.filters = nn.ModuleList(
                [GaborLayer(in_size, hidden_size, input_scale/np.sqrt(n_layers+1), alpha/(n_layers+1)) for _ in range(n_layers + 1)]
            )
            self.mixed = False
        elif filter == "Fourier":
            self.filters = nn.ModuleList(
                [FourierLayer(in_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1)) for _ in range(n_layers + 1)]
            )
            self.mixed = False
        elif filter == "Linear":
            self.filters = nn.ModuleList(
                [LinearLayer(in_size, hidden_size) for _ in range(n_layers + 1)]
            )
            self.mixed = False
        elif filter == "Mixed":
            # 每层一个 MixedFilter，由潜码门控三种分支的加权
            self.filters = nn.ModuleList(
                [MixedFilter(in_size, hidden_size,
                             weight_scale=input_scale/np.sqrt(n_layers+1),
                             alpha=alpha/(n_layers+1),
                             use_gabor=mix_use_gabor,
                             use_fourier=mix_use_fourier,
                             use_linear=mix_use_linear,
                             latent_size=latent_size_eff)
                 for _ in range(n_layers + 1)]
            )
            self.mixed = True
        else:
            raise ValueError(f"Unknown filter type: {filter}")

        # FiLM（可选）
        self.films = nn.ModuleList([FiLM(latent_size_eff, hidden_size) for _ in range(n_layers + 1)]) if use_film else None

        # 残差正则：LayerNorm + LayerScale + DropPath
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers + 1)]) if use_layernorm else None
        self.ls_params = nn.ParameterList([nn.Parameter(layer_scale_init * torch.ones(hidden_size)) for _ in range(n_layers + 1)])
        self.drop_path = DropPath(droppath) if droppath > 0 else nn.Identity()

        # 输出层
        self.output_layer = nn.Linear(hidden_size, out_size)

    def _get_latent(self, idx):
        # idx: list/int/1D tensor -> (B, L_eff)
        if not torch.is_tensor(idx):
            idx = torch.tensor(idx, dtype=torch.long, device=self.latents.device)
        latent = self.latents[idx, :]                    # (B, L)
        latent = self.latent_dp(latent)                  # dropout
        latent = self.latent_proj(latent)                # 可能降维
        return latent

    def _apply_film(self, h, latent, layer_id):
        if not self.use_film:
            return h
        gamma, beta = self.films[layer_id](latent)       # (B, H), (B, H)
        return h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, x, idx):
        """
        x:   (B, P, C)  coords
        idx: (B,) 索引到 latent 表
        """
        B, P, _ = x.shape
        latent = self._get_latent(idx)                   # (B, L_eff)

        # ---- 第 0 层（与原实现对齐：filters[0](x) * bilinear[0](0, latent)）----
        h0_coord = torch.zeros(B, P, self.in_size, device=x.device, dtype=x.dtype)
        s0 = self.bilinear[0](h0_coord, latent)          # (B, P, H)
        f0 = self.filters[0](x, latent) if self.mixed else self.filters[0](x)   # (B, P, H)
        u0 = F.silu(s0) * f0                             # multiplicative core
        u0 = self._apply_film(u0, latent, 0)             # 可选 FiLM
        h = u0                                           # 第一层不做残差相加，直赋值更稳

        if self.use_layernorm:
            h = self.norms[0](h)

        # ---- 深层循环 ----
        for i in range(1, len(self.filters)):
            s = self.bilinear[i](h, latent)              # (B, P, H)  使用上层特征替代“coord”
            f = self.filters[i](x, latent) if self.mixed else self.filters[i](x)
            u = F.silu(s) * f                            # multiplicative
            u = self._apply_film(u, latent, i)

            # 残差 + 层缩放 + droppath
            h = h + self.drop_path(u * self.ls_params[i])

            if self.use_layernorm:
                h = self.norms[i](h)

        return self.output_layer(h)


# --------------------------- quick self-test ---------------------------
if __name__ == "__main__":
    model = MMGN_NEW(
        in_size = 2,
        hidden_size = 300,
        n_data = 5,
        latent_size = 370,
        latent_init = "zeros",
        out_size = 1,
        n_layers = 7,
        input_scale = 370,
        alpha = 1.0,
        filter = "Mixed",          # 试试混合滤波；改回 "Linear/Gabor/Fourier" 即退化到旧版
        use_layernorm = True,
        layer_scale_init = 1e-2,
        droppath = 0.1,
        use_film = True,
        latent_dropout = 0.1,
        latent_proj = 256,         # 可选：把 L=370 压到 256 再用
        mix_use_gabor = True,
        mix_use_fourier = True,
        mix_use_linear = True
    ).cuda() if torch.cuda.is_available() else MMGN_NEW()

    B, P = 5, 700
    input_coord = torch.rand((B, P, 2))
    input_idx = torch.arange(B)
    with torch.no_grad():
        output_y = model(input_coord, input_idx)
        print("out:", output_y.shape)   # (B, P, 1)
