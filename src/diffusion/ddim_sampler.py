# -*- coding: utf-8 -*-
"""
ddim_sampler.py - DDIM sampling from existing FlowLevel diffusion model.

DDIM (Song et al. 2021) uses a deterministic non-Markovian sampler
that converges in far fewer steps than DDPM. This often fixes the
all-zero output problem that occurs when DDPM iterates 1000 steps
on a small dataset (1,213 samples).

Strategy:
  - Load existing diffusion_model.pt
  - Build DDIM sub-sequence of T_DDIM steps from T=1000 beta schedule
  - Run reverse sampling for N_SAMPLES levels at each skill target
  - Validate output (argmax → tile array → player/box count)
  - Save successful samples

Run: python flowlevel/src/diffusion/ddim_sampler.py
Output:
  - data/ddim_samples.npy         — tile arrays for valid samples
  - data/ddim_sample_report.txt   — validity stats per skill level
"""

import os, sys, math
PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR  = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_TILES    = 8
EMBED_DIM  = 16
T_DDIM     = 200        # number of DDIM denoising steps (vs 1000 for DDPM)
ETA        = 0.0        # DDIM eta=0 → fully deterministic sampling
N_SAMPLES  = 20         # samples per skill level
SKILL_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

PLAYER_TILES = {2, 6}
CRATE_TILES  = {3, 5}
TARGET_TILES = {4, 5, 6}


# ── Rebuild UNet architecture (matches train_diffusion.py) ───────────────────
class FiLM(nn.Module):
    def __init__(self, cd, ch):
        super().__init__()
        self.fc = nn.Linear(cd, ch * 2)
    def forward(self, x, c):
        s, b = self.fc(c).chunk(2, -1)
        return x * (1 + s.view(-1, x.shape[1], 1, 1)) + b.view(-1, x.shape[1], 1, 1)

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
    def __init__(self, in_ch=N_TILES, cd=EMBED_DIM + 32):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        self.e1  = Block(in_ch, 32, cd)
        self.e2  = Block(32,    64, cd)
        self.bt  = Block(64,    64, cd)
        self.d2  = Block(128,   32, cd)
        self.d1  = Block(64,    32, cd)
        self.out = nn.Conv2d(32, in_ch, 1)

    def sin_t(self, t):
        h = 16
        f = torch.exp(-math.log(10000) * torch.arange(h, device=t.device) / h)
        a = t[:, None].float() * f[None]
        return torch.cat([torch.cos(a), torch.sin(a)], -1)

    def forward(self, x, t, sk):
        c  = torch.cat([sk, self.t_mlp(self.sin_t(t))], -1)
        e1 = self.e1(x,  c)
        e2 = self.e2(e1, c)
        b  = self.bt(e2, c)
        d2 = self.d2(torch.cat([b,  e2], 1), c)
        d1 = self.d1(torch.cat([d2, e1], 1), c)
        return self.out(d1)


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading diffusion model...")
ckpt       = torch.load(os.path.join(MODEL_DIR, "diffusion_model.pt"),
                        map_location=DEVICE)
model      = UNet().to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

alpha_hat_full = ckpt["alpha_hat"].to(DEVICE)   # (T=1000,)
betas_full     = ckpt["betas"].to(DEVICE)

# Load embeddings for skill conditioning
embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))   # (N, 16)
metrics_np = np.load(os.path.join(DATA_DIR, "metrics.npy"))      # (N, 4)
sort_idx   = np.argsort(metrics_np[:, 1])
embeddings = embeddings[sort_idx]
N_LEVELS   = len(embeddings)

print(f"Model loaded. Device: {DEVICE}. Running DDIM T={T_DDIM} steps.")


# ── Build DDIM sub-sequence ───────────────────────────────────────────────────
T_FULL = 1000
step_size = T_FULL // T_DDIM
# Select T_DDIM evenly spaced timesteps from [0, T_FULL)
ddim_ts = list(reversed(range(0, T_FULL, step_size)))[:T_DDIM]
# alpha_hat values at those timesteps
ah = alpha_hat_full[ddim_ts]              # (T_DDIM,)


@torch.no_grad()
def ddim_sample(skill_embed: torch.Tensor, H: int = 5, W: int = 5) -> torch.Tensor:
    """
    DDIM deterministic reverse diffusion for one sample.
    skill_embed: (1, 16)
    Returns: (N_TILES, H, W) float tensor
    """
    x = torch.randn(1, N_TILES, H, W, device=DEVICE)

    for i, t_idx in enumerate(ddim_ts):
        t_tensor = torch.full((1,), t_idx, device=DEVICE, dtype=torch.long)
        predicted_noise = model(x, t_tensor, skill_embed)

        ah_t    = ah[i]
        ah_prev = ah[i + 1] if i + 1 < len(ddim_ts) else torch.tensor(1.0, device=DEVICE)

        # DDIM update rule (Song et al. 2021, Eq. 12)
        x0_pred = (x - torch.sqrt(1 - ah_t) * predicted_noise) / torch.sqrt(ah_t)
        x0_pred = x0_pred.clamp(-1, 1)

        # Direction pointing to xt
        dir_xt = torch.sqrt(1 - ah_prev) * predicted_noise

        x = torch.sqrt(ah_prev) * x0_pred + dir_xt
        # ETA=0 means no random noise added (fully deterministic)

    return x.squeeze(0)   # (N_TILES, H, W)


def tensor_to_level(x: torch.Tensor) -> np.ndarray:
    """Convert (N_TILES, H, W) float → (H, W) integer tile array."""
    return x.argmax(dim=0).cpu().numpy().astype(np.int32)


def is_valid(lvl: np.ndarray) -> bool:
    flat = lvl.flatten()
    n_player  = sum(1 for t in flat if t in PLAYER_TILES)
    n_crates  = sum(1 for t in flat if t in CRATE_TILES)
    n_targets = sum(1 for t in flat if t in TARGET_TILES)
    return n_player == 1 and n_crates >= 1 and n_targets >= 1 and n_crates == n_targets


def skill_to_embed(skill: float) -> torch.Tensor:
    """Pick a representative embedding from the difficulty-sorted array."""
    skill = max(0.0, min(1.0, skill))
    center = int(skill * (N_LEVELS - 1))
    window = max(10, N_LEVELS // 10)
    lo, hi = max(0, center - window // 2), min(N_LEVELS - 1, center + window // 2)
    idx = np.random.randint(lo, hi + 1)
    emb = torch.tensor(embeddings[idx], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return emb


# ── Run sampling ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DDIM Sampling Results")
print("=" * 55)

all_valid_samples = []
report_lines      = []

for skill in SKILL_LEVELS:
    valid_count = 0
    samples     = []

    for _ in range(N_SAMPLES):
        emb    = skill_to_embed(skill)
        output = ddim_sample(emb)
        level  = tensor_to_level(output)
        samples.append(level)
        if is_valid(level):
            valid_count += 1
            all_valid_samples.append(level)

    pct  = valid_count / N_SAMPLES * 100
    line = f"  Skill {skill:.1f}  |  {valid_count}/{N_SAMPLES} valid  ({pct:.0f}%)"
    print(line)
    report_lines.append(line)

    # Show a sample tile distribution for inspection
    if samples:
        sample = samples[0]
        unique, counts = np.unique(sample, return_counts=True)
        tile_dist = dict(zip(unique.tolist(), counts.tolist()))
        print(f"           Sample[0] tile dist: {tile_dist}")

total_valid = len(all_valid_samples)
summary     = f"\nTotal valid: {total_valid}/{len(SKILL_LEVELS) * N_SAMPLES} samples"
print(summary)
report_lines.append(summary)


# ── Save ──────────────────────────────────────────────────────────────────────
if all_valid_samples:
    out_arr = np.array(all_valid_samples, dtype=np.int32)
    np.save(os.path.join(DATA_DIR, "ddim_samples.npy"), out_arr)
    print(f"Saved: data/ddim_samples.npy  shape={out_arr.shape}")
else:
    print("No valid samples generated. DDPM all-zero problem persists.")
    print("Next step: collect 50k rollouts to scale dataset to ~6k levels.")

with open(os.path.join(DATA_DIR, "ddim_sample_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"DDIM Sampling Report (T_DDIM={T_DDIM}, ETA={ETA})\n")
    f.write(f"Skill levels: {SKILL_LEVELS}\n")
    f.write(f"Samples per skill: {N_SAMPLES}\n\n")
    f.write("\n".join(report_lines))

print(f"Report saved: data/ddim_sample_report.txt")
print("\nDDIM sampling complete.")
