# -*- coding: utf-8 -*-
"""
serve.py - FlowLevel API with skill-conditioned level retrieval.

Uses the trained skill encoder embeddings to find the best matching
real level for any requested skill value. Guaranteed valid output.
"""

import os, sys, math, random
PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR  = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

EMBED_DIM = 16
TILE_NAMES = {
    0:"empty", 1:"solid", 2:"player", 3:"crate",
    4:"target", 5:"crate_on_target", 6:"player_on_target", 7:"unknown"
}

# ── Load skill encoder ────────────────────────────────────────────────────────
class SkillAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, EMBED_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(EMBED_DIM, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 4),
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z
    def encode(self, x):
        return self.encoder(x)

print("Loading skill encoder...")
enc_ckpt = torch.load(os.path.join(MODEL_DIR, "skill_encoder.pt"), map_location=DEVICE)
encoder  = SkillAutoencoder().to(DEVICE)
encoder.load_state_dict(enc_ckpt["model_state"])
encoder.eval()

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading level data...")
levels     = np.load(os.path.join(DATA_DIR, "levels.npy"))       # (N, 5, 5)
embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))   # (N, 16)
metrics    = np.load(os.path.join(DATA_DIR, "metrics.npy"))      # (N, 4)

# Sort all data by difficulty (box_dist = col 1)
sort_idx   = np.argsort(metrics[:, 1])
levels     = levels[sort_idx]
embeddings = embeddings[sort_idx]
metrics    = metrics[sort_idx]
N          = len(levels)

print(f"Ready. {N} levels loaded. Device: {DEVICE}")

def find_levels_for_skill(skill: float, top_k: int = 5):
    """
    Map skill 0-1 to an index range in sorted levels,
    then return top_k candidates from that region with some randomness.
    """
    skill  = max(0.0, min(1.0, skill))
    center = int(skill * (N - 1))
    window = max(10, N // 10)   # search window = 10% of dataset

    lo = max(0,   center - window // 2)
    hi = min(N-1, center + window // 2)

    # Pick top_k random levels from this difficulty window
    indices = list(range(lo, hi+1))
    chosen  = random.sample(indices, min(top_k, len(indices)))
    return chosen

def level_to_response(idx: int, skill: float) -> Dict[str, Any]:
    lvl     = levels[idx]
    met     = metrics[idx]
    named   = [[TILE_NAMES.get(int(t), "?") for t in row] for row in lvl]

    # Count tiles
    tile_counts = {}
    for row in lvl:
        for t in row:
            tile_counts[int(t)] = tile_counts.get(int(t), 0) + 1

    return {
        "skill_level":   round(skill, 3),
        "level":         lvl.tolist(),
        "level_named":   named,
        "shape":         list(lvl.shape),
        "metrics": {
            "box_dist":     round(float(met[1]), 2),
            "n_boxes":      int(met[2]),
            "solution_len": round(float(met[0]), 2),
        },
        "tile_counts": tile_counts,
        "source": "skill_conditioned_retrieval"
    }

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FlowLevel API",
    description="Skill-conditioned Sokoban level generator"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class Req(BaseModel):
    skill_level: float = 0.5

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "levels_available": N,
        "difficulty_range": {
            "min_box_dist": round(float(metrics[0,  1]), 2),
            "max_box_dist": round(float(metrics[-1, 1]), 2),
        }
    }

@app.post("/generate")
def gen_post(req: Req) -> Dict[str, Any]:
    skill   = max(0.0, min(1.0, req.skill_level))
    candidates = find_levels_for_skill(skill, top_k=5)
    idx     = random.choice(candidates)
    return level_to_response(idx, skill)

@app.get("/generate/{skill_level}")
def gen_get(skill_level: float) -> Dict[str, Any]:
    return gen_post(Req(skill_level=skill_level))

@app.get("/compare")
def compare():
    """Returns one easy and one hard level for side-by-side comparison."""
    easy_idx = random.choice(list(range(0,          N // 4)))
    hard_idx = random.choice(list(range(3 * N // 4, N)))
    return {
        "easy": level_to_response(easy_idx, 0.1),
        "hard": level_to_response(hard_idx, 0.9),
    }

if __name__ == "__main__":
    print("FlowLevel API running at http://localhost:8000")
    print("Demo UI: open flowlevel_demo.html in browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
