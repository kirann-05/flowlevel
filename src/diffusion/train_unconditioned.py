# -*- coding: utf-8 -*-
"""
train_unconditioned.py - Unconditional DDPM for ablation study.

Same flat UNet as train_diffusion.py but with FiLM conditioning
completely removed. Trained without skill embeddings.

This gives us the ablation baseline for Table 4:
  "Conditioned diffusion vs. unconditioned diffusion"

Run: python flowlevel/src/diffusion/train_unconditioned.py
Output: models/diffusion_unconditioned.pt
"""

import os, sys, math
PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DATA_DIR   = os.path.join(PROJECT_DIR, "data")
MODEL_DIR  = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

N_TILES    = 8
T_STEPS    = 1000
EPOCHS     = 150
BATCH_SIZE = 64
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

levels     = np.load(os.path.join(DATA_DIR, "levels.npy"))
H, W = levels.shape[1], levels.shape[2]
print(f"Levels {levels.shape} | Device {DEVICE}")
print("Training UNCONDITIONED model (no FiLM, no skill embeddings)")

# One-hot encode
loh = np.eye(N_TILES, dtype=np.float32)[levels].transpose(0, 3, 1, 2) * 2 - 1
X   = torch.tensor(loh)
loader = DataLoader(TensorDataset(X), batch_size=BATCH_SIZE, shuffle=True)

betas     = torch.linspace(1e-4, 0.02, T_STEPS, device=DEVICE)
alphas    = 1.0 - betas
alpha_hat = torch.cumprod(alphas, 0)

def q_sample(x0, t, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    a = alpha_hat[t].view(-1, 1, 1, 1)
    return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise, noise


# ── Unconditional Block (GroupNorm + Conv only, no FiLM) ─────────────────────
class UncondBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.c1   = nn.Conv2d(ic, oc, 3, padding=1)
        self.c2   = nn.Conv2d(oc, oc, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, oc), oc)
    def forward(self, x):
        return F.gelu(self.norm(self.c2(F.gelu(self.c1(x)))))


# ── Unconditional UNet ────────────────────────────────────────────────────────
class UNetUnconditioned(nn.Module):
    """
    Same flat architecture as the conditioned model.
    Timestep embedding projected into a bias added at bottleneck only.
    No skill embedding, no FiLM anywhere.
    """
    def __init__(self, in_ch=N_TILES):
        super().__init__()
        # Timestep MLP — projects to a single bias added at bottleneck
        self.t_mlp = nn.Sequential(
            nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 64)
        )
        self.t_proj = nn.Linear(64, 64)  # project to bottleneck channels

        self.e1  = UncondBlock(in_ch, 32)
        self.e2  = UncondBlock(32,    64)
        self.bt  = UncondBlock(64,    64)
        self.d2  = UncondBlock(128,   32)   # bt(64) + e2(64) = 128
        self.d1  = UncondBlock(64,    32)   # d2(32) + e1(32) = 64
        self.out = nn.Conv2d(32, in_ch, 1)

    def sin_t(self, t):
        h = 16
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h)
        a = t[:, None].float() * f[None]
        return torch.cat([torch.cos(a), torch.sin(a)], -1)   # (B, 32)

    def forward(self, x, t):
        te = self.t_proj(self.t_mlp(self.sin_t(t)))  # (B, 64)

        e1 = self.e1(x)                                   # (B, 32, 5, 5)
        e2 = self.e2(e1)                                  # (B, 64, 5, 5)

        # Inject timestep into bottleneck as spatial bias
        b_in = e2 + te.view(-1, 64, 1, 1)
        b    = self.bt(b_in)                              # (B, 64, 5, 5)

        d2 = self.d2(torch.cat([b,  e2], 1))             # (B, 32, 5, 5)
        d1 = self.d1(torch.cat([d2, e1], 1))             # (B, 32, 5, 5)
        return self.out(d1)                               # (B,  8, 5, 5)


# ── Training ──────────────────────────────────────────────────────────────────
model = UNetUnconditioned().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training {EPOCHS} epochs...")

for epoch in tqdm(range(EPOCHS), desc="Unconditioned Diffusion"):
    total = 0.0
    for (x0,) in loader:
        x0 = x0.to(DEVICE)
        t  = torch.randint(0, T_STEPS, (x0.shape[0],), device=DEVICE)
        xt, noise = q_sample(x0, t)
        loss = F.mse_loss(model(xt, t), noise)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    if (epoch + 1) % 30 == 0:
        tqdm.write(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.5f}")

save_path = os.path.join(MODEL_DIR, "diffusion_unconditioned.pt")
torch.save({
    "model_state": model.state_dict(),
    "config": {"n_tiles": N_TILES, "t_steps": T_STEPS, "H": H, "W": W,
                "conditioned": False},
    "alpha_hat": alpha_hat.cpu(),
    "betas":     betas.cpu(),
}, save_path)
print(f"Unconditioned diffusion model saved: {save_path}")
