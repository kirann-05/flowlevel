# -*- coding: utf-8 -*-
"""
train_ppo.py - PPO training on sokoban-narrow-v0.
DummyVecEnv for Jupyter/Windows compatibility.
TensorBoard logging disabled to avoid file lock issues.
"""

import os, sys
PROJECT_DIR = r"D:\College\Sem VI\Minor Project\flowlevel"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import gym
import gym_pcgrl
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

ENV_ID         = "sokoban-narrow-v0"
N_ENVS         = 4
TOTAL_STEPS    = 500_000
SAVE_FREQ      = 100_000
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
FINAL_PATH     = os.path.join(CHECKPOINT_DIR, "ppo_sokoban_final")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device   : {device}")
print(f"Env      : {ENV_ID}")
print(f"Steps    : {TOTAL_STEPS:,}")
print(f"Parallel : {N_ENVS} envs")

def make_env(rank, seed=42):
    def _init():
        e = gym.make(ENV_ID)
        e.seed(seed + rank)
        return e
    set_random_seed(seed)
    return _init

vec_env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
vec_env = VecMonitor(vec_env)
print("Environments ready.")

resume_path = None
if os.path.exists(CHECKPOINT_DIR):
    ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR)
                    if f.startswith("rl_model_") and f.endswith(".zip")])
    if ckpts:
        resume_path = os.path.join(CHECKPOINT_DIR, ckpts[-1])
        print(f"Resuming from: {resume_path}")

if resume_path:
    model = PPO.load(resume_path, env=vec_env, device=device)
    remaining = max(0, TOTAL_STEPS - model.num_timesteps)
    print(f"Steps done: {model.num_timesteps:,} | Remaining: {remaining:,}")
else:
    remaining = TOTAL_STEPS
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=None,      # disabled — avoids file lock errors
    )

checkpoint_cb = CheckpointCallback(
    save_freq=SAVE_FREQ // N_ENVS,
    save_path=CHECKPOINT_DIR,
    name_prefix="rl_model",
    verbose=1,
)

if remaining > 0:
    print(f"Training for {remaining:,} steps...")
    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        reset_num_timesteps=(resume_path is None),
    )
    model.save(FINAL_PATH)
    print(f"Training complete. Model saved to {FINAL_PATH}.zip")
else:
    print("Already complete.")

vec_env.close()
