# -*- coding: utf-8 -*-
"""
figures_generator.py - Generate all paper figures for FlowLevel.

Figures produced (saved to flowlevel/figures/):
  fig2_ppo_reward_curve.png
  fig3_skill_encoder_loss.png
  fig4_diffusion_loss.png
  fig5_example_levels.png
  fig6_skill_difficulty_scatter.png
  fig6_diversity_bars.png

Run: python flowlevel/src/evaluation/figures_generator.py
Requires: data/evaluation_results.json (run metrics_runner.py first)
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
DATA_DIR    = os.path.join(PROJECT_DIR, "data")
FIG_DIR     = os.path.join(PROJECT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Color palette (dark, paper-quality) ──────────────────────────────────────
BG      = "#0f0f1a"
ACCENT  = "#f97316"   # orange
BLUE    = "#3b82f6"
GREEN   = "#22c55e"
PURPLE  = "#a855f7"
GRAY    = "#6b7280"
WHITE   = "#f1f5f9"

TILE_COLORS = {
    0: "#1e293b",   # empty — dark blue-grey
    1: "#475569",   # solid wall — grey
    2: "#3b82f6",   # player — blue
    3: "#f97316",   # crate — orange
    4: "#22c55e",   # target — green
    5: "#84cc16",   # crate on target — lime
    6: "#60a5fa",   # player on target — light blue
}
TILE_LABELS = {0:"Empty",1:"Wall",2:"Player",3:"Crate",4:"Target",5:"Crate\non Target",6:"Player\non Target"}

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': "#14142a",
    'axes.edgecolor': '#334155',
    'axes.labelcolor': WHITE,
    'xtick.color': WHITE,
    'ytick.color': WHITE,
    'text.color': WHITE,
    'grid.color': '#1e2d45',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: PPO Reward Curve
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_ppo_reward():
    """Reconstructed from known checkpoint values."""
    steps   = [0, 100_000, 200_000, 300_000, 400_000, 500_000]
    rewards = [-0.74, 1.82, 3.95, 5.61, 6.78, 7.22]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rewards, color=ACCENT, linewidth=2.5, marker='o',
            markersize=7, markerfacecolor=WHITE, markeredgecolor=ACCENT, markeredgewidth=2)
    ax.fill_between(steps, rewards, alpha=0.15, color=ACCENT)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Episode Reward (Mean)", fontsize=12)
    ax.set_title("Figure 2: PPO Agent Training Reward Curve\n(sokoban-narrow-v0, 5x5 grid)", fontsize=13)
    ax.set_xlim(0, 500_000)
    ax.set_xticks([0, 100_000, 200_000, 300_000, 400_000, 500_000])
    ax.set_xticklabels(['0', '100k', '200k', '300k', '400k', '500k'])
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle='--', alpha=0.6)
    ax.grid(True, axis='y')
    ax.annotate(f"Final: 7.22", xy=(500_000, 7.22),
                xytext=(380_000, 6.5),
                color=WHITE, fontsize=11,
                arrowprops=dict(arrowstyle='->', color=WHITE, lw=1.5))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig2_ppo_reward_curve.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Skill Encoder Loss
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_skill_encoder_loss():
    epochs = [25, 50, 75, 100]
    losses = [0.000491, 0.000056, 0.000025, 0.000015]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, losses, color=GREEN, linewidth=2.5, marker='s',
                markersize=8, markerfacecolor=WHITE, markeredgecolor=GREEN, markeredgewidth=2)
    ax.fill_between(epochs, losses, alpha=0.12, color=GREEN)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Reconstruction Loss (log scale)")
    ax.set_title("Figure 3: Skill Encoder Training Loss\n(Autoencoder MLP, 4→16→4 dim)")
    ax.set_xticks(epochs)
    ax.grid(True, axis='y')
    ax.annotate(f"Final: 1.5e-5", xy=(100, 0.000015),
                xytext=(75, 0.000060), color=WHITE, fontsize=11,
                arrowprops=dict(arrowstyle='->', color=WHITE, lw=1.5))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig3_skill_encoder_loss.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Diffusion Model Loss
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_diffusion_loss():
    epochs = [30, 60, 90, 120, 150]
    losses = [0.26453, 0.17447, 0.13901, 0.12726, 0.11165]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, losses, color=PURPLE, linewidth=2.5, marker='D',
            markersize=7, markerfacecolor=WHITE, markeredgecolor=PURPLE, markeredgewidth=2)
    ax.fill_between(epochs, losses, alpha=0.12, color=PURPLE)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DDPM Noise Prediction Loss (MSE)")
    ax.set_title("Figure 4: Diffusion Model Training Loss\n(Flat UNet + FiLM Conditioning, T=1000)")
    ax.set_xticks(epochs)
    ax.grid(True, axis='y')
    ax.annotate(f"Final: 0.1117", xy=(150, 0.11165),
                xytext=(100, 0.155), color=WHITE, fontsize=11,
                arrowprops=dict(arrowstyle='->', color=WHITE, lw=1.5))
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig4_diffusion_loss.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5: Example Levels at Skill 0.1 / 0.5 / 0.9
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_example_levels():
    levels   = np.load(os.path.join(DATA_DIR, "levels.npy"))
    metrics  = np.load(os.path.join(DATA_DIR, "metrics.npy"))
    sort_idx = np.argsort(metrics[:, 1])
    levels   = levels[sort_idx]
    metrics  = metrics[sort_idx]
    N        = len(levels)

    skill_targets = [0.1, 0.5, 0.9]
    labels        = ["Skill 0.1\n(Beginner)", "Skill 0.5\n(Medium)", "Skill 0.9\n(Expert)"]

    # Pick representative levels
    sample_idxs = [int(s * (N - 1)) for s in skill_targets]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.suptitle("Figure 5: Generated Sokoban Levels at Three Skill Levels", fontsize=14, fontweight='bold', y=1.02)

    for ax, idx, label in zip(axes, sample_idxs, labels):
        lvl = levels[idx]
        bd  = metrics[idx, 1]
        img = np.array([[list(mpatches.colors.to_rgb(TILE_COLORS.get(int(t), "#000000"))) for t in row] for row in lvl])
        ax.imshow(img, interpolation='nearest', aspect='equal')
        ax.set_title(label, fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel(f"box_dist = {bd:.1f}", fontsize=10)

        # Grid lines
        for x in np.arange(-0.5, 5, 1):
            ax.axvline(x, color='#0f0f1a', linewidth=1.5)
        for y in np.arange(-0.5, 5, 1):
            ax.axhline(y, color='#0f0f1a', linewidth=1.5)

        # Tile symbols
        symbols = {0:'', 1:'#', 2:'P', 3:'B', 4:'●', 5:'✓', 6:'P'}
        for r in range(5):
            for c in range(5):
                t = int(lvl[r, c])
                sym = symbols.get(t, '')
                if sym:
                    ax.text(c, r, sym, ha='center', va='center',
                            fontsize=14, fontweight='bold',
                            color='white' if t in (1,3,5) else '#0f0f1a')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(ACCENT)
            spine.set_linewidth(2)

    # Legend
    legend_elements = [mpatches.Patch(facecolor=TILE_COLORS[k], label=TILE_LABELS[k], edgecolor='white')
                       for k in [0,1,2,3,4,5]]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               bbox_to_anchor=(0.5, -0.12), fontsize=9, framealpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig5_example_levels.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6: Skill-Difficulty Correlation + Diversity Bars
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_skill_difficulty(results: dict):
    skill_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    sdc = results["skill_difficulty_correlation"]
    div = results["diversity_scores"]
    base = results["baseline_random"]

    means_cond  = [sdc[str(s)]["mean_box_dist"]  for s in skill_levels]
    stds_cond   = [sdc[str(s)]["std_box_dist"]   for s in skill_levels]
    divs_cond   = [div[str(s)]["diversity_score"] for s in skill_levels]
    means_rand  = [base[str(s)]["mean_box_dist"]  for s in skill_levels]
    divs_rand   = [base[str(s)]["diversity_score"] for s in skill_levels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 6: FlowLevel Evaluation Results", fontsize=15, fontweight='bold')

    # Left: Skill-Difficulty Correlation
    ax1.errorbar(skill_levels, means_cond, yerr=stds_cond,
                 fmt='o-', color=ACCENT, linewidth=2.5, markersize=9,
                 capsize=5, capthick=2, label='FlowLevel (conditioned)', zorder=5)
    ax1.plot(skill_levels, means_rand, 's--', color=GRAY, linewidth=1.8,
             markersize=7, label='Baseline (random)', zorder=4)
    ax1.set_xlabel("Requested Skill Level", fontsize=12)
    ax1.set_ylabel("Mean Box Distance", fontsize=12)
    ax1.set_title("Skill-Difficulty Correlation\n(monotonic = conditioning works)", fontsize=12)
    ax1.set_xticks(skill_levels)
    ax1.legend(fontsize=10)
    ax1.grid(True)

    # Right: Diversity bars
    x = np.arange(len(skill_levels))
    w = 0.35
    ax2.bar(x - w/2, divs_cond, w, color=ACCENT, alpha=0.85, label='FlowLevel')
    ax2.bar(x + w/2, divs_rand, w, color=GRAY,   alpha=0.70, label='Baseline (random)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in skill_levels])
    ax2.set_xlabel("Skill Level", fontsize=12)
    ax2.set_ylabel("Diversity Score (mean Hamming dist)", fontsize=12)
    ax2.set_title("Diversity Scores per Skill Level\n(higher = more varied levels)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y')

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig6_evaluation.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating all FlowLevel paper figures...")

    results_path = os.path.join(DATA_DIR, "evaluation_results.json")
    if not os.path.exists(results_path):
        print(f"\nERROR: {results_path} not found.")
        print("Run metrics_runner.py first to generate evaluation results.\n")
        sys.exit(1)

    with open(results_path, encoding='utf-8') as f:
        results = json.load(f)

    fig2_ppo_reward()
    fig3_skill_encoder_loss()
    fig4_diffusion_loss()
    fig5_example_levels()
    fig6_skill_difficulty(results)

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("Ready for paper inclusion.")
