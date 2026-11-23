import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ U-Net parts
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.tproj = nn.Linear(tdim, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.tproj(F.silu(temb))[:, :, None, None]     # inject the timestep
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)   # (B, HW, HW)
        out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """28x28 -> 14x14 -> 7x7 (with self-attention) -> 14x14 -> 28x28, skip connections."""

    def __init__(self, in_ch=1, base=64, mults=(1, 2, 4), tdim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(base), nn.Linear(base, tdim),
                                      nn.SiLU(), nn.Linear(tdim, tdim))
        self.conv_in = nn.Conv2d(in_ch, base, 3, padding=1)
        chs = [base * m for m in mults]

        self.downs = nn.ModuleList()
        prev = base
        for i, c in enumerate(chs):
            down = nn.Conv2d(c, c, 3, 2, 1) if i < len(chs) - 1 else nn.Identity()
            self.downs.append(nn.ModuleList([ResBlock(prev, c, tdim), ResBlock(c, c, tdim), down]))
            prev = c

        self.mid1 = ResBlock(prev, prev, tdim)
        self.mid_attn = SelfAttention(prev)
        self.mid2 = ResBlock(prev, prev, tdim)

        self.ups = nn.ModuleList()
        for i, c in reversed(list(enumerate(chs))):
            up = (nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1))
                  if i > 0 else nn.Identity())
            self.ups.append(nn.ModuleList([ResBlock(prev + c, c, tdim), ResBlock(c, c, tdim), up]))
            prev = c

        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, in_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = self.conv_in(x)
        skips = []
        for r1, r2, down in self.downs:
            h = r2(r1(h, temb), temb)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for r1, r2, up in self.ups:
            h = torch.cat([h, skips.pop()], dim=1)
            h = r2(r1(h, temb), temb)
            h = up(h)
        return self.out(h)


# ------------------------------------------------------------------ diffusion
class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_ab = self.alpha_bars.sqrt()
        self.sqrt_1m_ab = (1.0 - self.alpha_bars).sqrt()

    def q_sample(self, x0, t, noise):
        """Jump straight to x_t from x_0 (closed form of the forward process)."""
        return self.sqrt_ab[t][:, None, None, None] * x0 + self.sqrt_1m_ab[t][:, None, None, None] * noise

    def loss(self, model, x0, generator=None):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device, generator=generator)
        noise = torch.randn(x0.shape, device=x0.device, generator=generator)
        return F.mse_loss(model(self.q_sample(x0, t, noise), t), noise)

    @torch.no_grad()
    def sample(self, model, n, shape=(1, 28, 28), device="cuda", snapshot_ts=()):
        """Reverse process x_T -> x_0. Returns final samples and {t: x_t} snapshots."""
        model.eval()
        x = torch.randn(n, *shape, device=device)
        snaps = {self.T: x.clone()}
        for t in reversed(range(self.T)):
            tt = torch.full((n,), t, device=device, dtype=torch.long)
            eps = model(x, tt)
            a, ab, b = self.alphas[t], self.alpha_bars[t], self.betas[t]
            mean = (x - b / (1 - ab).sqrt() * eps) / a.sqrt()
            x = mean + b.sqrt() * torch.randn_like(x) if t > 0 else mean
            if t in snapshot_ts:
                snaps[t] = x.clone()
        return x.clamp(-1, 1), snaps
