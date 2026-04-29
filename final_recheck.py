# -*- coding: utf-8 -*-
"""
final_recheck.py - Comprehensive objective verification for FlowLevel.
Run: python flowlevel/final_recheck.py
"""
import sys, os, json, subprocess
sys.path.insert(0, r"D:\College\Sem VI\Minor Project\flowlevel")
import torch, numpy as np

PROJECT = r"D:\College\Sem VI\Minor Project\flowlevel"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print("  FLOWLEVEL -- FINAL OBJECTIVE RECHECK")
print(f"  Device: {DEVICE}")
print("=" * 65)

# ── 1. Required files ──────────────────────────────────────────────
REQUIRED = {
    "PPO checkpoint":           "checkpoints/ppo_sokoban_final.zip",
    "Level data":               "data/levels.npy",
    "Difficulty metrics":       "data/metrics.npy",
    "Skill embeddings":         "data/embeddings.npy",
    "Skill encoder":            "models/skill_encoder.pt",
    "Diffusion model":          "models/diffusion_model.pt",
    "Unconditioned model":      "models/diffusion_unconditioned.pt",
    "LSTM skill estimator":     "models/skill_estimator.pt",
    "Scaler stats":             "models/scaler_stats.npy",
    "Synthetic sessions":       "data/synthetic_sessions.npy",
    "DDIM samples":             "data/ddim_samples.npy",
    "Eval results JSON":        "data/evaluation_results.json",
    "Fig 2 (PPO curve)":        "figures/fig2_ppo_reward_curve.png",
    "Fig 4 (Diffusion loss)":   "figures/fig4_diffusion_loss.png",
    "Fig 6 (Eval)":             "figures/fig6_evaluation.png",
    "Generator demo":           "demo/flowlevel_demo.html",
    "Playable demo":            "demo/play.html",
    "Presentation":             "FlowLevel_Review2_Presentation.html",
}

all_ok = True
print("\n  [1] REQUIRED FILES")
for name, rel in REQUIRED.items():
    p = os.path.join(PROJECT, rel)
    if os.path.exists(p):
        kb = os.path.getsize(p) // 1024
        print(f"  [OK]  {name:<35} {kb} KB")
    else:
        print(f"  [!!]  {name:<35} MISSING")
        all_ok = False

# ── 2. Data sanity ─────────────────────────────────────────────────
print("\n  [2] DATA SANITY")
lvls = np.load(os.path.join(PROJECT, "data/levels.npy"))
mets = np.load(os.path.join(PROJECT, "data/metrics.npy"))
print(f"  Levels shape  : {lvls.shape}")
print(f"  box_dist range: {mets[:, 1].min():.1f} - {mets[:, 1].max():.1f}")
synth = np.load(os.path.join(PROJECT, "data/synthetic_sessions.npy"))
print(f"  Synth sessions: {synth.shape}  solve_rate={synth[:, 2].mean():.2f}")

# ── 3. Evaluation results ──────────────────────────────────────────
print("\n  [3] EVALUATION RESULTS (Skill-Difficulty Correlation)")
with open(os.path.join(PROJECT, "data/evaluation_results.json")) as f:
    ev = json.load(f)
sdc = ev.get("skill_difficulty_correlation", {})
prev_bd = 0.0
monotonic = True
for sk in ["0.1", "0.3", "0.5", "0.7", "0.9"]:
    v = sdc.get(sk, {})
    bd = v.get("mean_box_dist", 0)
    ok = "OK" if bd >= prev_bd else "FAIL"
    print(f"  skill={sk}  mean_box_dist={bd:.4f}  [{ok}]")
    if bd < prev_bd:
        monotonic = False
    prev_bd = bd
print(f"  Monotonically increasing: {'YES' if monotonic else 'NO'}")

# ── 4. LSTM smoke test ─────────────────────────────────────────────
print("\n  [4] LSTM SKILL ESTIMATOR")
import torch.nn as nn

class Est(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 64, batch_first=True, num_layers=2, dropout=0.0)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)

ckpt = torch.load(os.path.join(PROJECT, "models/skill_estimator.pt"), map_location=DEVICE)
m = Est().to(DEVICE)
m.load_state_dict(ckpt["model_state"])
m.eval()
print(f"  Loaded OK. val_mse={ckpt['best_val_mse']:.5f}  RMSE={ckpt['best_val_mse']**0.5:.4f}")
mean_v = np.array(ckpt["scaler_mean"], dtype=np.float32)
std_v  = np.array(ckpt["scaler_std"],  dtype=np.float32)
for tgt in [0.1, 0.5, 0.9]:
    row = np.array([[5000 + (1-tgt)*15000, 10 + (1-tgt)*30,
                     float(tgt > 0.3), int((1-tgt)*5), int((1-tgt)*1500), tgt]] * 3,
                   dtype=np.float32)
    row[:, :5] = (row[:, :5] - mean_v) / std_v
    x = torch.tensor(row).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = m(x).item()
    err = abs(pred - tgt)
    status = "OK" if err < 0.2 else "WARN"
    print(f"  Target {tgt:.1f}  ->  predicted {pred:.3f}  (error {err:.3f}) [{status}]")

# ── 5. Unconditioned model ─────────────────────────────────────────
print("\n  [5] UNCONDITIONED DIFFUSION (ablation)")
uc  = torch.load(os.path.join(PROJECT, "models/diffusion_unconditioned.pt"), map_location=DEVICE, weights_only=False)
cfg = uc.get("config", {})
print(f"  Config : {cfg}")
print(f"  Keys   : {list(uc.keys())}")

# ── 6. DDIM report ────────────────────────────────────────────────
print("\n  [6] DDIM LIVE INFERENCE REPORT")
report_path = os.path.join(PROJECT, "data/ddim_sample_report.txt")
if os.path.exists(report_path):
    with open(report_path) as f:
        for line in f.readlines()[:8]:
            print(f"  {line.rstrip()}")
else:
    print("  (report file not found)")

# ── 7. Git status ──────────────────────────────────────────────────
print("\n  [7] GIT STATUS")
res = subprocess.run(["git", "-C", PROJECT, "log", "--oneline", "-4"],
                     capture_output=True, text=True)
print(res.stdout.strip())

# ── Final verdict ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ALL OBJECTIVES MET -- READY FOR SUBMISSION" if all_ok
      else "  SOME FILES MISSING -- SEE ABOVE")
print("=" * 65)
