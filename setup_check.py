"""
FlowLevel — Setup Verification Script
Run this after pip install -r requirements.txt
It checks every dependency and tells you exactly what's working and what isn't.

Usage: python setup_check.py
"""

import sys

print("=" * 55)
print("  FlowLevel — Environment Check")
print("=" * 55)

errors = []

# Python version
print(f"\n[1] Python version: {sys.version.split()[0]}", end="  ")
if sys.version_info >= (3, 10):
    print("OK")
else:
    print("FAIL — need Python 3.10+")
    errors.append("Python version too old")

# PyTorch
print("[2] PyTorch...", end="  ")
try:
    import torch
    print(f"OK — version {torch.__version__}", end="  ")
    if torch.cuda.is_available():
        print(f"| GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("| No GPU detected — will use CPU (slower but works)")
except ImportError:
    print("FAIL — run: pip install torch")
    errors.append("PyTorch not installed")

# NumPy
print("[3] NumPy...", end="  ")
try:
    import numpy as np
    print(f"OK — version {np.__version__}")
except ImportError:
    print("FAIL — run: pip install numpy")
    errors.append("NumPy not installed")

# Stable-Baselines3
print("[4] Stable-Baselines3 (PPO)...", end="  ")
try:
    import stable_baselines3
    from stable_baselines3 import PPO
    print(f"OK — version {stable_baselines3.__version__}")
except ImportError:
    print("FAIL — run: pip install stable-baselines3[extra]")
    errors.append("Stable-Baselines3 not installed")

# Gymnasium
print("[5] Gymnasium...", end="  ")
try:
    import gymnasium
    print(f"OK — version {gymnasium.__version__}")
except ImportError:
    print("FAIL — run: pip install gymnasium")
    errors.append("Gymnasium not installed")

# Diffusers
print("[6] HuggingFace Diffusers...", end="  ")
try:
    import diffusers
    print(f"OK — version {diffusers.__version__}")
except ImportError:
    print("FAIL — run: pip install diffusers")
    errors.append("Diffusers not installed")

# Matplotlib
print("[7] Matplotlib...", end="  ")
try:
    import matplotlib
    print(f"OK — version {matplotlib.__version__}")
except ImportError:
    print("FAIL — run: pip install matplotlib")
    errors.append("Matplotlib not installed")

# FastAPI
print("[8] FastAPI...", end="  ")
try:
    import fastapi
    print(f"OK — version {fastapi.__version__}")
except ImportError:
    print("FAIL — run: pip install fastapi uvicorn")
    errors.append("FastAPI not installed")

# gym-pcgrl check
print("[9] gym-pcgrl...", end="  ")
try:
    import gym_pcgrl
    print("OK")
except ImportError:
    print("NOT INSTALLED YET — install separately (see below)")
    print("      Run this:")
    print("      pip install git+https://github.com/amidos2006/gym-pcgrl.git")

# W&B
print("[10] Weights & Biases...", end="  ")
try:
    import wandb
    print(f"OK — version {wandb.__version__}")
except ImportError:
    print("FAIL — run: pip install wandb")
    errors.append("wandb not installed")

# Final verdict
print("\n" + "=" * 55)
if not errors:
    print("  ALL CHECKS PASSED — you are ready to build FlowLevel")
else:
    print(f"  {len(errors)} issue(s) found:")
    for e in errors:
        print(f"    - {e}")
    print("\n  Fix the above then re-run this script.")
print("=" * 55)
