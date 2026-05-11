# -*- coding: utf-8 -*-
"""
skill_encoder.py - Compress difficulty metrics into skill embeddings.

Architecture: Autoencoder  [4 -> 64 -> 16 -> 64 -> 4]
Training: MSE reconstruction loss on normalised metrics.
Output: 16-dim skill embedding per level, used to condition diffusion model.
"""

import os, sys
PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DATA_DIR   = os.path.join(PROJECT_DIR, "data_10x10")
MODEL_DIR  = os.path.join(PROJECT_DIR, "models_10x10")
os.makedirs(MODEL_DIR, exist_ok=True)

EMBED_DIM  = 16
HIDDEN_DIM = 64
EPOCHS     = 100
BATCH_SIZE = 256
LR         = 1e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

metrics = np.load(os.path.join(DATA_DIR, "metrics.npy"))
print(f"Loaded metrics: {metrics.shape} on {DEVICE}")

mean = metrics.mean(0)
std  = metrics.std(0) + 1e-8
metrics_norm = (metrics - mean) / std
np.save(os.path.join(MODEL_DIR, "scaler_stats.npy"), np.stack([mean, std]))
print(f"Mean: {mean.round(3)}  Std: {std.round(3)}")

tensor = torch.tensor(metrics_norm, dtype=torch.float32)
loader = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True)


class SkillAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM//2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM//2, EMBED_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM//2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM//2, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 4),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    def encode(self, x):
        return self.encoder(x)


model   = SkillAutoencoder().to(DEVICE)
opt     = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(f"Training for {EPOCHS} epochs...")
for epoch in tqdm(range(EPOCHS)):
    total = 0.0
    for (b,) in loader:
        b = b.to(DEVICE)
        recon, _ = model(b)
        loss = loss_fn(recon, b)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    if (epoch+1) % 25 == 0:
        tqdm.write(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.6f}")

enc_path = os.path.join(MODEL_DIR, "skill_encoder.pt")
torch.save({"model_state": model.state_dict(), "embed_dim": EMBED_DIM}, enc_path)
print(f"Encoder saved: {enc_path}")

model.eval()
all_z = []
with torch.no_grad():
    for (b,) in DataLoader(TensorDataset(tensor), batch_size=512):
        all_z.append(model.encode(b.to(DEVICE)).cpu().numpy())
embeddings = np.concatenate(all_z, axis=0)
emb_path   = os.path.join(DATA_DIR, "embeddings.npy")
np.save(emb_path, embeddings)
print(f"Embeddings saved: {emb_path}  shape={embeddings.shape}")
print("Skill encoder complete.")
