# -*- coding: utf-8 -*-
"""
train_diffusion.py - Skill-conditioned DDPM on 5x5 Sokoban levels.
Shallow UNet, no pooling below 5x5, explicit channel sizes.
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

DATA_DIR   = os.path.join(PROJECT_DIR, "data_10x10")
MODEL_DIR  = os.path.join(PROJECT_DIR, "models_10x10")
os.makedirs(MODEL_DIR, exist_ok=True)

N_TILES    = 8
EMBED_DIM  = 16
T_STEPS    = 1000
EPOCHS     = 200        # increased for 10x10
BATCH_SIZE = 64
LR         = 2e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

levels     = np.load(os.path.join(DATA_DIR, "levels.npy"))
embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
H, W = levels.shape[1], levels.shape[2]
print(f"Levels {levels.shape} | Embeddings {embeddings.shape} | Device {DEVICE}")

loh = np.eye(N_TILES, dtype=np.float32)[levels].transpose(0,3,1,2) * 2 - 1
X   = torch.tensor(loh)
C   = torch.tensor(embeddings, dtype=torch.float32)
loader = DataLoader(TensorDataset(X, C), batch_size=BATCH_SIZE, shuffle=True)

betas     = torch.linspace(1e-4, 0.02, T_STEPS, device=DEVICE)
alphas    = 1.0 - betas
alpha_hat = torch.cumprod(alphas, 0)

def q_sample(x0, t, noise=None):
    if noise is None: noise = torch.randn_like(x0)
    a = alpha_hat[t].view(-1,1,1,1)
    return torch.sqrt(a)*x0 + torch.sqrt(1-a)*noise, noise


class FiLM(nn.Module):
    def __init__(self, cd, ch):
        super().__init__()
        self.fc = nn.Linear(cd, ch*2)
    def forward(self, x, c):
        s, b = self.fc(c).chunk(2, -1)
        return x*(1+s.view(-1,x.shape[1],1,1)) + b.view(-1,x.shape[1],1,1)


class Block(nn.Module):
    def __init__(self, ic, oc, cd):
        super().__init__()
        self.c1   = nn.Conv2d(ic, oc, 3, padding=1)
        self.c2   = nn.Conv2d(oc, oc, 3, padding=1)
        self.film = FiLM(cd, oc)
        self.norm = nn.GroupNorm(8, oc)
    def forward(self, x, c):
        return F.gelu(self.norm(self.film(self.c2(F.gelu(self.c1(x))), c)))


class UNet(nn.Module):
    """
    Standard UNet with 2 pooling levels for 10x10 input.
    10x10 -> pool -> 5x5 -> pool -> 2x2 -> bottleneck -> upsample -> upsample
    """
    def __init__(self, in_ch=N_TILES, cd=EMBED_DIM+32):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(32,64), nn.GELU(), nn.Linear(64,32))
        base = 64
        self.e1 = Block(in_ch,  base,   cd)   # 10x10
        self.e2 = Block(base,   base*2, cd)   # 5x5 after pool
        self.bt = Block(base*2, base*2, cd)   # 2x2 after pool
        self.d2 = Block(base*4, base,   cd)   # 5x5 after upsample + concat
        self.d1 = Block(base*2, base,   cd)   # 10x10 after upsample + concat
        self.out = nn.Conv2d(base, in_ch, 1)
        self.pool = nn.MaxPool2d(2)
        self.up2  = nn.Upsample(size=(5,5), mode='nearest')
        self.up1  = nn.Upsample(size=(10,10), mode='nearest')

    def sin_t(self, t):
        h = 16
        f = torch.exp(-math.log(10000)*torch.arange(h, device=t.device)/h)
        a = t[:,None].float()*f[None]
        return torch.cat([torch.cos(a), torch.sin(a)], -1)

    def forward(self, x, t, sk):
        c  = torch.cat([sk, self.t_mlp(self.sin_t(t))], -1)
        e1 = self.e1(x, c)                                   # (B, 64,  10, 10)
        e2 = self.e2(self.pool(e1), c)                       # (B, 128,  5,  5)
        b  = self.bt(self.pool(e2), c)                       # (B, 128,  2,  2)
        d2 = self.d2(torch.cat([self.up2(b), e2], 1), c)    # (B, 64,   5,  5)
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1), c)   # (B, 64,  10, 10)
        return self.out(d1)                                   # (B, 8,   10, 10)


model = UNet().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training {EPOCHS} epochs on {DEVICE}...")

for epoch in tqdm(range(EPOCHS), desc="Diffusion"):
    total = 0.0
    for x0, cond in loader:
        x0, cond = x0.to(DEVICE), cond.to(DEVICE)
        t = torch.randint(0, T_STEPS, (x0.shape[0],), device=DEVICE)
        xt, noise = q_sample(x0, t)
        loss = F.mse_loss(model(xt, t, cond), noise)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    if (epoch+1) % 30 == 0:
        tqdm.write(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.5f}")

save_path = os.path.join(MODEL_DIR, "diffusion_model.pt")
torch.save({
    "model_state": model.state_dict(),
    "config": {"n_tiles": N_TILES, "embed_dim": EMBED_DIM,
                "t_steps": T_STEPS, "H": H, "W": W},
    "alpha_hat": alpha_hat.cpu(),
    "betas":     betas.cpu(),
}, save_path)
print(f"Diffusion model saved: {save_path}")
