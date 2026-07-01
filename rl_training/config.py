"""
Central configuration for the competitive tennis PPO pipeline.
All hyperparameters live here; no magic numbers in other files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class TrainingConfig:
    # ── Environment ──────────────────────────────────────────────────────────
    env_path: str = "./Tennis.app"  # mlagents globs for Tennis.* and finds Tennis.app
    env_port: int = 5005                # base port; avoids conflicts on multi-run
    time_scale: float = 20.0            # physics speed-up (1.0 = real-time)
    obs_dim: int = 24                   # 3 stacked frames × 8 spatial vars
    act_dim: int = 2                    # [Δx-force, Δy-force], bounded [-1,1]

    # ── Network ───────────────────────────────────────────────────────────────
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # ── PPO / GAE ─────────────────────────────────────────────────────────────
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99                 # discount factor
    gae_lambda: float = 0.95            # GAE λ — bias-variance trade-off
    clip_epsilon: float = 0.2           # PPO surrogate clip ratio
    entropy_coef: float = 0.01          # exploration entropy bonus weight
    value_loss_coef: float = 0.5        # critic loss relative weight
    max_grad_norm: float = 0.5          # global gradient clip norm
    ppo_epochs: int = 10                # optimization passes per rollout buffer
    mini_batch_size: int = 256          # SGD mini-batch size
    rollout_steps: int = 2048           # environment steps before each update

    # ── Self-Play Gating ──────────────────────────────────────────────────────
    win_rate_threshold: float = 0.70    # freeze active policy when above this
    eval_window: int = 500              # rolling episode window for win-rate
    min_steps_between_freezes: int = 20_000  # hard cooldown between checkpoints
    recent_opponent_prob: float = 0.80  # prob of facing the most-recent ckpt
    checkpoint_dir: str = "./checkpoints"

    # ── Training Loop ─────────────────────────────────────────────────────────
    total_timesteps: int = 10_000_000
    save_every_steps: int = 100_000     # save Elo snapshot + checkpoint
    log_every_steps: int = 2_048        # console logging frequency

    # ── Infrastructure ────────────────────────────────────────────────────────
    seed: int = 42
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
