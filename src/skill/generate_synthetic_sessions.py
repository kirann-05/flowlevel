# -*- coding: utf-8 -*-
"""
generate_synthetic_sessions.py - Simulate player behavior using precomputed metrics.

Uses levels.npy + metrics.npy (already computed) to simulate player behavioral
signals at 20 skill levels. No A* calls needed — the solution_len and box_dist
columns in metrics.npy are used directly. Runs in ~2 seconds (numpy vectorised).

Run: python flowlevel/src/skill/generate_synthetic_sessions.py
Output: data/synthetic_sessions.npy  shape (2000, 6)
        data/synthetic_labels.npy    shape (2000,)
"""

import os, sys
PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import numpy as np

DATA_DIR = os.path.join(PROJECT_DIR, "data")
SEED     = 42
rng      = np.random.default_rng(SEED)

SKILL_LEVELS       = np.linspace(0.05, 1.0, 20)   # 20 skill levels
SESSIONS_PER_SKILL = 10                             # 10 sessions per skill
LEVELS_PER_SESSION = 10                             # 10 levels per session

# Load sorted levels and precomputed metrics
levels  = np.load(os.path.join(DATA_DIR, "levels.npy"))
metrics = np.load(os.path.join(DATA_DIR, "metrics.npy"))   # [sol_len, box_dist, n_boxes, solvable]
sort_idx = np.argsort(metrics[:, 1])                        # sort by box_dist
levels   = levels[sort_idx]
metrics  = metrics[sort_idx]
N        = len(levels)

print(f"Generating synthetic sessions (vectorised — no A*)...")
print(f"  {len(SKILL_LEVELS)} skill levels x {SESSIONS_PER_SKILL} sessions x {LEVELS_PER_SESSION} levels")

all_features = []
all_labels   = []

for skill in SKILL_LEVELS:
    center = int(skill * (N - 1))
    window = max(10, N // 10)
    lo     = max(0, center - window // 2)
    hi     = min(N - 1, center + window // 2)

    for _ in range(SESSIONS_PER_SKILL):
        idxs     = rng.integers(lo, hi + 1, size=LEVELS_PER_SESSION)
        met_sel  = metrics[idxs]                          # (10, 4)
        sol_len  = np.maximum(1, met_sel[:, 0])           # solution_len col
        solvable = met_sel[:, 3]                           # 1.0 / 0.0

        # Solve probability: stronger players solve more
        solve_prob  = float(np.clip(skill + 0.3, 0.0, 1.0))
        solved_mask = (rng.random(LEVELS_PER_SESSION) < solve_prob) & (solvable > 0.5)

        # Move count: optimal x inefficiency (weak players take ~4x more moves)
        inefficiency  = 1.0 + (1.0 - skill) * 3.0
        total_moves   = np.maximum(1, (sol_len * inefficiency * rng.uniform(0.8, 1.2, LEVELS_PER_SESSION)).astype(int))

        # Solve time in ms: weaker players are slower per move
        ms_per_move   = int(1500 + (1.0 - skill) * 2000)
        solve_time_ms = np.clip(
            total_moves * ms_per_move + rng.integers(-500, 500, LEVELS_PER_SESSION),
            0, 60000
        )

        # Undo count: inverse of skill
        undo_rate  = np.clip((1.0 - skill) * 0.3 + rng.uniform(-0.05, 0.05, LEVELS_PER_SESSION), 0, 1)
        undo_count = (total_moves * undo_rate).astype(int)

        # Time variance between moves: more hesitation for weaker players
        time_variance = ((1.0 - skill) * 2000 + rng.integers(0, 500, LEVELS_PER_SESSION)).astype(int)

        for i in range(LEVELS_PER_SESSION):
            all_features.append([
                float(solve_time_ms[i]),
                float(total_moves[i]),
                float(solved_mask[i]),
                float(undo_count[i]),
                float(time_variance[i]),
                float(round(skill, 4)),
            ])
            all_labels.append(float(skill))

features_arr = np.array(all_features, dtype=np.float32)
labels_arr   = np.array(all_labels,   dtype=np.float32)

np.save(os.path.join(DATA_DIR, "synthetic_sessions.npy"), features_arr)
np.save(os.path.join(DATA_DIR, "synthetic_labels.npy"),   labels_arr)

print(f"\nDone!")
print(f"  synthetic_sessions.npy  shape={features_arr.shape}")
print(f"  synthetic_labels.npy    shape={labels_arr.shape}")
print(f"  Skill range : {labels_arr.min():.2f} - {labels_arr.max():.2f}")
print(f"  Solve rate  : {features_arr[:, 2].mean():.2f}")
print("Run train_estimator.py next.")
