# -*- coding: utf-8 -*-
"""
metrics_runner.py - Quantitative evaluation for FlowLevel.

Generates levels at 5 skill levels and computes:
  Table 1: Skill-Difficulty Correlation (mean box_dist per skill group)
  Table 2: Diversity Score (pairwise Hamming distance within skill group)
  Table 3: Validity Rate (% levels with correct entity counts)

Run: python flowlevel/src/evaluation/metrics_runner.py
Results saved to: flowlevel/data/evaluation_results.json
"""

import os, sys, random, json
import numpy as np

PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATA_DIR  = os.path.join(PROJECT_DIR, "data")
OUT_FILE  = os.path.join(DATA_DIR, "evaluation_results.json")

N_SAMPLES_PER_SKILL = 100   # generate 100 levels per skill level
SKILL_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]

TILE = {0:"empty", 1:"solid", 2:"player", 3:"crate",
        4:"target", 5:"crate_on_target", 6:"player_on_target"}
PLAYER_TILES   = {2, 6}
CRATE_TILES    = {3, 5}
TARGET_TILES   = {4, 5, 6}

# ── Load data (sorted by difficulty, same as serve.py) ───────────────────────
print("Loading level data...")
levels   = np.load(os.path.join(DATA_DIR, "levels.npy"))    # (N, 5, 5)
metrics  = np.load(os.path.join(DATA_DIR, "metrics.npy"))   # (N, 4) [sol, box_dist, n_boxes, solvable]

sort_idx = np.argsort(metrics[:, 1])   # sort by box_dist col
levels   = levels[sort_idx]
metrics  = metrics[sort_idx]
N        = len(levels)
print(f"  {N} levels loaded. box_dist range: {metrics[0,1]:.2f} - {metrics[-1,1]:.2f}")


# ── Sampling (same logic as serve.py) ────────────────────────────────────────
def sample_levels_for_skill(skill: float, n: int = 100):
    """Sample n level indices from the difficulty window for this skill."""
    skill  = max(0.0, min(1.0, skill))
    center = int(skill * (N - 1))
    window = max(10, N // 10)
    lo = max(0, center - window // 2)
    hi = min(N - 1, center + window // 2)
    pool = list(range(lo, hi + 1))
    # Sample with replacement if pool < n
    if len(pool) >= n:
        chosen = random.sample(pool, n)
    else:
        chosen = random.choices(pool, k=n)
    return chosen


# ── Validity check ────────────────────────────────────────────────────────────
def is_valid_level(lvl: np.ndarray) -> bool:
    """Exactly 1 player, >= 1 crate, >= 1 target, n_crates == n_targets."""
    flat = lvl.flatten()
    n_player  = sum(1 for t in flat if t in PLAYER_TILES)
    n_crates  = sum(1 for t in flat if t in CRATE_TILES)
    n_targets = sum(1 for t in flat if t in TARGET_TILES)
    return n_player == 1 and n_crates >= 1 and n_targets >= 1 and n_crates == n_targets


# ── Diversity: pairwise Hamming distance ─────────────────────────────────────
def diversity_score(level_batch: np.ndarray) -> float:
    """
    Mean pairwise Hamming distance (fraction of tiles that differ)
    across all pairs in the batch.
    Higher = more diverse.
    """
    n = len(level_batch)
    if n < 2:
        return 0.0
    flat = level_batch.reshape(n, -1)  # (n, 25)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = float(np.sum(flat[i] != flat[j])) / flat.shape[1]
            total += diff
            count += 1
    return total / count if count > 0 else 0.0


# ── Baseline: random sampling (no skill conditioning) ────────────────────────
def sample_random_levels(n: int = 100):
    """Randomly sample n levels from the entire dataset — no skill targeting."""
    return random.sample(range(N), min(n, N))


# ── Main evaluation loop ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("FLOWLEVEL QUANTITATIVE EVALUATION")
print("="*60)

results = {
    "skill_difficulty_correlation": {},   # Table 1
    "diversity_scores":             {},   # Table 2
    "validity_rates":               {},   # Table 3
    "baseline_random": {}
}

print("\nTable 1+2+3: Skill-conditioned results")
print(f"{'Skill':>6}  {'Mean box_dist':>14}  {'Std':>6}  {'Diversity':>10}  {'Validity%':>10}")
print("-"*58)

for skill in SKILL_LEVELS:
    indices = sample_levels_for_skill(skill, N_SAMPLES_PER_SKILL)
    sampled_levels  = levels[indices]
    sampled_metrics = metrics[indices]

    box_dists     = sampled_metrics[:, 1]
    mean_bd       = float(np.mean(box_dists))
    std_bd        = float(np.std(box_dists))
    div_score     = diversity_score(sampled_levels)
    valid_count   = sum(1 for lvl in sampled_levels if is_valid_level(lvl))
    validity_rate = valid_count / len(sampled_levels) * 100.0

    results["skill_difficulty_correlation"][str(skill)] = {
        "mean_box_dist": round(mean_bd, 4),
        "std_box_dist":  round(std_bd, 4),
        "n_samples":     len(indices)
    }
    results["diversity_scores"][str(skill)] = {
        "diversity_score": round(div_score, 4),
        "n_pairs": (len(indices) * (len(indices) - 1)) // 2
    }
    results["validity_rates"][str(skill)] = {
        "valid_count":   valid_count,
        "total":         len(indices),
        "validity_pct":  round(validity_rate, 2)
    }

    print(f"  {skill:.1f}  {mean_bd:>14.4f}  {std_bd:>6.4f}  {div_score:>10.4f}  {validity_rate:>9.1f}%")

# ── Ablation: unconditioned model vs conditioned ──────────────────────────────
# If unconditioned diffusion model has been trained and DDIM sampled,
# use those samples as the ablation baseline. Otherwise fall back to random.
DDIM_SAMPLES_PATH = os.path.join(DATA_DIR, "ddim_samples.npy")
ddim_available    = os.path.exists(DDIM_SAMPLES_PATH)

print(f"\nTable 4: Ablation — {'DDIM unconditioned' if ddim_available else 'Random (fallback)'}")
if ddim_available:
    ddim_levels = np.load(DDIM_SAMPLES_PATH)
    print(f"  DDIM samples loaded: {len(ddim_levels)} levels")

for skill_target in SKILL_LEVELS:
    if ddim_available and len(ddim_levels) >= N_SAMPLES_PER_SKILL:
        # Use DDIM unconditioned samples — no skill targeting
        rand_indices  = random.sample(range(len(ddim_levels)),
                                      min(N_SAMPLES_PER_SKILL, len(ddim_levels)))
        base_levels   = ddim_levels[rand_indices]
        # Compute box_dist for ddim levels using bfs_checker
        from src.evaluation.bfs_checker import box_distances as _bd
        base_bd_vals  = np.array([_bd(l) for l in base_levels], dtype=np.float32)
        base_div      = diversity_score(base_levels)
        baseline_label = "ddim_unconditioned"
    else:
        # Fallback: random sampling from dataset
        rand_indices  = sample_random_levels(N_SAMPLES_PER_SKILL)
        base_levels   = levels[rand_indices]
        base_bd_vals  = metrics[rand_indices][:, 1]
        base_div      = diversity_score(base_levels)
        baseline_label = "random_sampling"

    results["baseline_random"][str(skill_target)] = {
        "mean_box_dist":   round(float(np.mean(base_bd_vals)), 4),
        "diversity_score": round(base_div, 4),
        "baseline_type":   baseline_label,
    }
    print(f"  skill={skill_target}  {baseline_label}  "
          f"mean_box_dist={np.mean(base_bd_vals):.4f}  diversity={base_div:.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {OUT_FILE}")

# ── Print LaTeX tables ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LaTeX TABLE 1 — Skill-Difficulty Correlation")
print("="*60)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Skill-Difficulty Correlation: Mean box distance per requested skill level}")
print(r"\begin{tabular}{ccc}")
print(r"\hline")
print(r"Skill Level & Mean Box Distance & Std Dev \\")
print(r"\hline")
for skill in SKILL_LEVELS:
    s = str(skill)
    bd  = results["skill_difficulty_correlation"][s]["mean_box_dist"]
    std = results["skill_difficulty_correlation"][s]["std_box_dist"]
    print(f"{skill:.1f} & {bd:.4f} & {std:.4f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n" + "="*60)
print("LaTeX TABLE 2 — Diversity Scores")
print("="*60)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Diversity Scores: Mean pairwise Hamming distance at each skill level}")
print(r"\begin{tabular}{cc}")
print(r"\hline")
print(r"Skill Level & Diversity Score \\")
print(r"\hline")
for skill in SKILL_LEVELS:
    s = str(skill)
    div = results["diversity_scores"][s]["diversity_score"]
    print(f"{skill:.1f} & {div:.4f} \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n" + "="*60)
print("LaTeX TABLE 3 — Validity Rates")
print("="*60)
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Level Validity Rate at each skill level}")
print(r"\begin{tabular}{ccc}")
print(r"\hline")
print(r"Skill Level & Valid Levels & Validity \% \\")
print(r"\hline")
for skill in SKILL_LEVELS:
    s = str(skill)
    vc = results["validity_rates"][s]["valid_count"]
    tt = results["validity_rates"][s]["total"]
    vp = results["validity_rates"][s]["validity_pct"]
    print(f"{skill:.1f} & {vc}/{tt} & {vp:.1f}\\% \\\\")
print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")

print("\nDone. Use data/evaluation_results.json for figures notebook.")
