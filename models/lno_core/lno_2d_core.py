# models/lno_core/lno_2d_core.py
import os
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast

class PR2d(nn.Module):
    """
    Pole-Residue 2D layer, 参数以“实部/虚部”两组 float32 保存，前向再拼成 complex。
    这样反向梯度都是 float，避免 AMP 对 ComplexFloat 的 unscale 报错。
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, in_channels * out_channels)

        # ---- 用 float32 存参数（实/虚分离）----
        def p(shape):
            return nn.Parameter(scale * torch.randn(*shape, dtype=torch.float32))
        self.pole1_re   = p((in_channels, out_channels, self.modes1))
        self.pole1_im   = p((in_channels, out_channels, self.modes1))
        self.pole2_re   = p((in_channels, out_channels, self.modes2))
        self.pole2_im   = p((in_channels, out_channels, self.modes2))
        self.res_re     = p((in_channels, out_channels, self.modes1, self.modes2))
        self.res_im     = p((in_channels, out_channels, self.modes1, self.modes2))

    @staticmethod
    def _fftfreq_as_omega(n: int, d: float, device):
        f = torch.fft.fftfreq(n, d=d, device=device, dtype=torch.float32)
        return (2.0 * np.pi * 1j * f).to(torch.complex64)

    @staticmethod
    def _fft2_safe(x: torch.Tensor):
        # 可用环境变量强制 CPU FFT：LNO_FORCE_CPU_FFT=1
        force_cpu = os.getenv("LNO_FORCE_CPU_FFT", "0") == "1"
        x32 = x.float().contiguous()
        if force_cpu:
            alpha_cpu = torch.fft.fft2(x32.cpu(), dim=(-2, -1))
            return alpha_cpu.to(x32.device)
        try:
            with autocast(enabled=False):
                return torch.fft.fft2(x32, dim=(-2, -1))
        except RuntimeError as e:
            if "cuFFT error" in str(e):
                alpha_cpu = torch.fft.fft2(x32.cpu(), dim=(-2, -1))
                return alpha_cpu.to(x32.device)
            raise

    @staticmethod
    def _ifft2_safe(x: torch.Tensor, s_hw):
        force_cpu = os.getenv("LNO_FORCE_CPU_FFT", "0") == "1"
        if force_cpu:
            y_cpu = torch.fft.ifft2(x.cpu(), s=s_hw).real
            return y_cpu.to(x.device)
        try:
            with autocast(enabled=False):
                return torch.fft.ifft2(x, s=s_hw).real
        except RuntimeError as e:
            if "cuFFT error" in str(e):
                y_cpu = torch.fft.ifft2(x.cpu(), s=s_hw).real
                return y_cpu.to(x.device)
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, H, W) real32
        return: (B, Cout, H, W) real32
        """
        B, Cin, H, W = x.shape
        device = x.device

        # 组装复数权重（前向期临时 complex64，参数本体仍是 float32）
        w_pole1 = torch.complex(self.pole1_re, self.pole1_im).to(torch.complex64)  # (Cin,Cout,m1)
        w_pole2 = torch.complex(self.pole2_re, self.pole2_im).to(torch.complex64)  # (Cin,Cout,m2)
        w_res   = torch.complex(self.res_re,  self.res_im ).to(torch.complex64)    # (Cin,Cout,m1,m2)

        # 频域变量
        alpha = self._fft2_safe(x)                               # (B,Cin,H,W) complex64
        lam1  = self._fftfreq_as_omega(H, 1.0 / H, device)       # (H,)
        lam2  = self._fftfreq_as_omega(W, 1.0 / W, device)       # (W,)

        # 稳态项
        A1 = 1.0 / (lam1.view(1, 1, 1, H) - w_pole1.unsqueeze(-1))     # (Cin,Cout,m1,H)
        A2 = 1.0 / (lam2.view(1, 1, 1, W) - w_pole2.unsqueeze(-1))     # (Cin,Cout,m2,W)
        T  = torch.einsum("comn,conw->comw", w_res, A2)                # (Cin,Cout,m1,W)
        G  = torch.einsum("comh,comw->cohw", A1, T)                    # (Cin,Cout,H,W)
        out_res1 = torch.einsum("bchw,cohw->bohw", alpha, G)           # (B,Cout,H,W)
        x1 = self._ifft2_safe(out_res1, s_hw=(H, W))                   # (B,Cout,H,W) real

        # 瞬态项
        S1 = torch.einsum("bchw,comh->bcowm", alpha, A1)               # (B,Cin,Cout,W,m1)
        M  = torch.einsum("conw,bcowm->bconm", A2, S1).movedim(-1, -2) # (B,Cin,Cout,m1,m2)
        out_res2 = torch.einsum("comn,bconm->bomn", w_res, M)          # (B,Cout,m1,m2)

        yy = torch.linspace(0, 1, H, device=device, dtype=torch.float32)
        xx = torch.linspace(0, 1, W, device=device, dtype=torch.float32)
        E1 = torch.exp(w_pole1.unsqueeze(-1) * yy.view(1,1,1,H))       # (Cin,Cout,m1,H)
        E2 = torch.exp(w_pole2.unsqueeze(-1) * xx.view(1,1,1,W))       # (Cin,Cout,m2,W)
        A  = E1.permute(1, 2, 3, 0)                                    # (Cout,m1,H,Cin)
        B2 = E2.permute(1, 2, 3, 0)                                    # (Cout,m2,W,Cin)
        M2 = torch.einsum("omhc,onwc->ohwmn", A, B2)                   # (Cout,H,W,m1,m2)
        x2 = torch.einsum("bomn,ohwmn->bohw", out_res2, M2).real
        x2 = x2 / (H * W)

        return (x1 + x2).float()


# -----------------------------
#  LNO block（工程化）
# -----------------------------
class LNO2DCore(nn.Module):
    """
    输入:  (B, in_ch, H, W)  —— 例如 [obs, mask]
    输出:  (B, out_ch, H, W)
    """
    def __init__(self, in_ch=2, out_ch=1, width=64, modes1=4, modes2=4):
        super().__init__()
        self.width = int(width)
        self.fc0 = nn.Linear(in_ch + 2, self.width)  # +2: 拼 (x,y)
        self.pr = PR2d(self.width, self.width, modes1=modes1, modes2=modes2)
        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1, bias=True)
        self.norm = nn.InstanceNorm2d(self.width)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_ch)

    @staticmethod
    def _make_xy_grid(B, H, W, device):
        gx = torch.linspace(0.0, 1.0, H, device=device, dtype=torch.float32).view(1, H, 1, 1).repeat(B, 1, W, 1)
        gy = torch.linspace(0.0, 1.0, W, device=device, dtype=torch.float32).view(1, 1, W, 1).repeat(B, H, 1, 1)
        return torch.cat([gx, gy], dim=-1)  # (B,H,W,2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        xy = self._make_xy_grid(B, H, W, x.device)         # (B,H,W,2)
        x_feat = torch.permute(x, (0, 2, 3, 1))            # (B,H,W,C)
        x_feat = self.fc0(torch.cat([x_feat, xy], dim=-1)) # (B,H,W,width)
        x_feat = torch.permute(x_feat, (0, 3, 1, 2))       # (B,width,H,W)

        x_feat_n = self.norm(x_feat).contiguous()
        y1 = self.norm(self.pr(x_feat_n))
        y2 = self.w0(x_feat)
        y  = y1 + y2                                       # (B,width,H,W)

        y = torch.permute(y, (0, 2, 3, 1))                 # (B,H,W,width)
        y = torch.sin(self.fc1(y))
        y = self.fc2(y)                                    # (B,H,W,out_ch)
        return torch.permute(y, (0, 3, 1, 2))              # (B,out_ch,H,W)
