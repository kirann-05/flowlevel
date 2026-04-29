# -*- coding: utf-8 -*-
"""
player_model.py - A* behavioral solver for Sokoban difficulty measurement.

Replaces box_distance with true behavioral metrics derived from actually solving
each level. Produces 6 features per level vs the original 4.

Features produced:
  solution_len    : min moves to solve (-1 if unsolvable)
  backtracks      : states revisited during A* search
  dead_ends       : moves with no valid neighbor
  states_explored : total A* states expanded
  box_dist        : Manhattan box-target distance (kept as complement)
  n_boxes         : number of boxes

Run: python src/skill/player_model.py
Output: data/behavioral_metrics.npy  shape (N, 6)
        data/behavioral_embeddings.npy  shape (N, 16)  (re-encoded)
"""

import os, sys, heapq, json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATA_DIR  = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

EMPTY=0; SOLID=1; PLAYER=2; CRATE=3; TARGET=4; CRATE_ON_TARGET=5; PLAYER_ON_TARGET=6
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

MAX_STEPS = 300   # A* budget per level — keeps runtime fast on 5x5


def astar_solve(level: np.ndarray) -> dict:
    """
    A* solver for Sokoban 5x5. Returns behavioral difficulty metrics.
    Uses Manhattan distance heuristic. Timeout at MAX_STEPS node expansions.
    """
    player, boxes, goals = None, [], []
    for r in range(level.shape[0]):
        for c in range(level.shape[1]):
            t = int(level[r, c])
            if t in (PLAYER, PLAYER_ON_TARGET):
                player = (r, c)
                if t == PLAYER_ON_TARGET:
                    goals.append((r, c))
            elif t == CRATE:
                boxes.append((r, c))
            elif t == TARGET:
                goals.append((r, c))
            elif t == CRATE_ON_TARGET:
                boxes.append((r, c))
                goals.append((r, c))

    if not player or not boxes or not goals:
        return _fail_metrics()

    goal_set = frozenset(map(tuple, goals))
    solid    = set(map(tuple, zip(*np.where(level == SOLID)))) if np.any(level == SOLID) else set()

    def heuristic(box_set):
        return sum(
            min(abs(br-gr) + abs(bc-gc) for gr, gc in goal_set)
            for br, bc in box_set
        )

    box_dist_val = heuristic(frozenset(map(tuple, boxes)))

    init       = (player, frozenset(map(tuple, boxes)))
    g_score    = defaultdict(lambda: float('inf'))
    g_score[init] = 0
    open_heap  = [(heuristic(init[1]), 0, init)]
    visited    = set()
    backtracks = 0
    dead_ends  = 0

    while open_heap:
        _, g, state = heapq.heappop(open_heap)
        pos, box_set = state

        if state in visited:
            backtracks += 1
            continue
        visited.add(state)

        if g > MAX_STEPS:
            break

        if box_set == goal_set:
            return {
                'solution_len':    g,
                'backtracks':      backtracks,
                'dead_ends':       dead_ends,
                'states_explored': len(visited),
                'box_dist':        box_dist_val,
                'n_boxes':         len(boxes),
                'solvable':        1.0,
            }

        r, c = pos
        valid_moves = 0
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in solid:
                continue
            new_boxes = box_set
            if (nr, nc) in box_set:
                br, bc = nr + dr, nc + dc
                if (br, bc) in solid or (br, bc) in box_set:
                    continue
                new_boxes = (box_set - {(nr, nc)}) | {(br, bc)}
            ns = ((nr, nc), new_boxes)
            tg = g + 1
            if tg < g_score[ns]:
                g_score[ns] = tg
                heapq.heappush(open_heap, (tg + heuristic(new_boxes), tg, ns))
            valid_moves += 1

        if valid_moves == 0:
            dead_ends += 1

    return {
        'solution_len':    -1,
        'backtracks':      backtracks,
        'dead_ends':       dead_ends,
        'states_explored': len(visited),
        'box_dist':        box_dist_val,
        'n_boxes':         len(boxes),
        'solvable':        0.0,
    }


def _fail_metrics():
    return {'solution_len': -1, 'backtracks': 0, 'dead_ends': 0,
            'states_explored': 0, 'box_dist': 0.0, 'n_boxes': 0, 'solvable': 0.0}


FEATURE_KEYS = ['solution_len', 'backtracks', 'dead_ends',
                 'states_explored', 'box_dist', 'n_boxes']


if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    levels = np.load(os.path.join(DATA_DIR, 'levels.npy'))
    N = len(levels)
    print(f"Running A* on {N} levels (max_steps={MAX_STEPS})...")

    results = []
    solvable_count = 0
    for lvl in tqdm(levels, desc='A* solving'):
        m = astar_solve(lvl)
        results.append([m[k] for k in FEATURE_KEYS])
        if m['solvable'] > 0.5:
            solvable_count += 1

    behavioral = np.array(results, dtype=np.float32)
    out_path = os.path.join(DATA_DIR, 'behavioral_metrics.npy')
    np.save(out_path, behavioral)
    print(f"\nBehavioral metrics saved: {out_path}  shape={behavioral.shape}")
    print(f"Solvable: {solvable_count}/{N}  ({100*solvable_count/N:.1f}%)")

    # Print comparison table
    print("\n--- Behavioral Difficulty by Level Index Quintile ---")
    idxs = np.argsort(behavioral[:, 4])  # sort by box_dist
    q = N // 5
    for i, label in enumerate(['Easiest 20%', 'Easy 20%', 'Medium 20%', 'Hard 20%', 'Hardest 20%']):
        seg = behavioral[idxs[i*q:(i+1)*q]]
        print(f"  {label}: sol_len={seg[:,0].mean():.1f}  "
              f"backtracks={seg[:,1].mean():.1f}  "
              f"states={seg[:,3].mean():.1f}  "
              f"box_dist={seg[:,4].mean():.2f}")

    # Re-train a 6-feature skill encoder on behavioral metrics
    print("\nTraining 6-feature behavioral skill encoder...")
    DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBED_DIM = 16
    EPOCHS    = 100

    mean = behavioral.mean(0)
    std  = behavioral.std(0) + 1e-8
    norm = (behavioral - mean) / std
    np.save(os.path.join(MODEL_DIR, 'behavioral_scaler_stats.npy'), np.stack([mean, std]))

    tensor = torch.tensor(norm, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=256, shuffle=True)

    class BehavioralEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(6, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, EMBED_DIM),
            )
            self.decoder = nn.Sequential(
                nn.Linear(EMBED_DIM, 32), nn.ReLU(),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, 6),
            )
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z
        def encode(self, x):
            return self.encoder(x)

    enc = BehavioralEncoder().to(DEVICE)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(EPOCHS), desc='Encoder'):
        total = 0.0
        for (b,) in loader:
            b = b.to(DEVICE)
            recon, _ = enc(b)
            loss = loss_fn(recon, b)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (epoch+1) % 25 == 0:
            tqdm.write(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.6f}")

    enc.eval()
    all_z = []
    with torch.no_grad():
        for (b,) in DataLoader(TensorDataset(tensor), batch_size=512):
            all_z.append(enc.encode(b.to(DEVICE)).cpu().numpy())
    behavioral_embeddings = np.concatenate(all_z, 0)

    np.save(os.path.join(DATA_DIR, 'behavioral_embeddings.npy'), behavioral_embeddings)
    torch.save({'model_state': enc.state_dict(), 'embed_dim': EMBED_DIM,
                'n_features': 6, 'feature_keys': FEATURE_KEYS,
                'scaler_mean': mean.tolist(), 'scaler_std': std.tolist()},
               os.path.join(MODEL_DIR, 'behavioral_encoder.pt'))

    print(f"\nBehavioral encoder saved: models/behavioral_encoder.pt")
    print(f"Behavioral embeddings saved: data/behavioral_embeddings.npy  shape={behavioral_embeddings.shape}")
    print("Done. Use behavioral_embeddings.npy in serve.py for richer skill conditioning.")
