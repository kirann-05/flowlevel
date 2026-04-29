# -*- coding: utf-8 -*-
"""
run_all.py - FlowLevel one-command demo launcher for review day.

Verifies all modules, prints status for each, then starts the API.
Run: python run_all.py
"""

import os, sys

PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

print("=" * 60)
print("  FLOWLEVEL — System Status Check")
print("=" * 60)

CHECKS = [
    ("PPO Checkpoint",       os.path.join(PROJECT_DIR, "checkpoints", "ppo_sokoban_final.zip")),
    ("Level Data",           os.path.join(PROJECT_DIR, "data", "levels.npy")),
    ("Difficulty Metrics",   os.path.join(PROJECT_DIR, "data", "metrics.npy")),
    ("Skill Embeddings",     os.path.join(PROJECT_DIR, "data", "embeddings.npy")),
    ("Skill Encoder Model",  os.path.join(PROJECT_DIR, "models", "skill_encoder.pt")),
    ("Diffusion Model",      os.path.join(PROJECT_DIR, "models", "diffusion_model.pt")),
    ("Scaler Stats",         os.path.join(PROJECT_DIR, "models", "scaler_stats.npy")),
    ("Evaluation Results",   os.path.join(PROJECT_DIR, "data", "evaluation_results.json")),
    ("Figure 2 (PPO curve)", os.path.join(PROJECT_DIR, "figures", "fig2_ppo_reward_curve.png")),
    ("Figure 6 (Eval)",      os.path.join(PROJECT_DIR, "figures", "fig6_evaluation.png")),
    ("Generator Demo",       os.path.join(PROJECT_DIR, "demo", "flowlevel_demo.html")),
    ("Playable Demo",        os.path.join(PROJECT_DIR, "demo", "play.html")),
    ("Presentation",         os.path.join(PROJECT_DIR, "FlowLevel_Review2_Presentation.html")),
]

# Optional behavioral files
OPTIONAL = [
    ("Behavioral Metrics",    os.path.join(PROJECT_DIR, "data",   "behavioral_metrics.npy")),
    ("Behavioral Encoder",    os.path.join(PROJECT_DIR, "models", "behavioral_encoder.pt")),
    ("Behavioral Embeddings", os.path.join(PROJECT_DIR, "data",   "behavioral_embeddings.npy")),
    ("LSTM Skill Estimator",  os.path.join(PROJECT_DIR, "models", "skill_estimator.pt")),
    ("Synthetic Sessions",    os.path.join(PROJECT_DIR, "data",   "synthetic_sessions.npy")),
    ("DDIM Samples",          os.path.join(PROJECT_DIR, "data",   "ddim_samples.npy")),
    ("Unconditioned Model",   os.path.join(PROJECT_DIR, "models", "diffusion_unconditioned.pt")),
]

all_ok = True
for name, path in CHECKS:
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) // 1024 if exists else 0
    status = f"OK ({size_kb} KB)" if exists else "MISSING"
    icon   = "[OK]" if exists else "[!!]"
    print(f"  {icon}  {name:<28} {status}")
    if not exists:
        all_ok = False

print()
print("  Optional (A* behavioral model):")
for name, path in OPTIONAL:
    exists = os.path.exists(path)
    size_kb = os.path.getsize(path) // 1024 if exists else 0
    icon = "[OK]" if exists else "[  ]"
    status = f"OK ({size_kb} KB)" if exists else "not yet run"
    print(f"  {icon}  {name:<28} {status}")

print()
if not all_ok:
    print("  [!!] Some required files are missing. Resolve before demo.")
    sys.exit(1)

# Quick import check
print("  Importing FlowLevel API...")
try:
    import numpy as np
    import torch
    from fastapi import FastAPI
    import uvicorn
    print("  [OK] All dependencies importable.")
except ImportError as e:
    print(f"  [!!] Import error: {e}")
    sys.exit(1)

# Load and verify levels
import numpy as np
levels = np.load(os.path.join(PROJECT_DIR, "data", "levels.npy"))
metrics = np.load(os.path.join(PROJECT_DIR, "data", "metrics.npy"))
print(f"\n  Levels loaded: {len(levels):,}  Shape: {levels.shape}")
print(f"  Difficulty range: box_dist {metrics[:,1].min():.1f} – {metrics[:,1].max():.1f}")

print("\n" + "=" * 60)
print("  ALL CHECKS PASSED — Starting FlowLevel API")
print("  API: http://localhost:8000")
print("  Docs: http://localhost:8000/docs")
print("  Demo: open demo/flowlevel_demo.html in browser")
print("  Play: open demo/play.html in browser")
print("=" * 60 + "\n")

import uvicorn
# Import the app from serve.py
serve_path = os.path.join(PROJECT_DIR, "src", "api", "serve.py")
exec(open(serve_path, encoding='utf-8').read())
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
