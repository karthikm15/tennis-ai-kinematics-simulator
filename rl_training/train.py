"""
Main competitive tennis PPO training loop (pure Python env — no Unity).

Flow per iteration
──────────────────
1. ROLLOUT  — Collect `rollout_steps` environment steps.
              Each step = one shot by the active policy (team 0 / AI side).
              Team 1's response is handled INSIDE TennisEnv via opponent_fn.

2. GAE      — Bootstrap the final value and compute λ-returns + advantages.

3. PPO UPDATE — ppo_epochs optimization passes over the buffer.

4. SELF-PLAY GATE — After each episode, check the rolling win-rate.
                    If > threshold: freeze active policy, sample new opponent.

5. LOGGING  — Scalar stats printed every rollout.

Usage
─────
    cd rl_training
    python3 train.py
"""
from __future__ import annotations

import os
import time
import random
from typing import Optional

import numpy as np
import torch

from buffer import RolloutBuffer
from config import TrainingConfig
from tennis_env import TennisEnv
from networks import ActorCritic
from ppo import PPOOptimizer
from self_play import SelfPlayManager


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(
    global_step: int,
    stats: dict,
    self_play: SelfPlayManager,
    t_start: float,
    ep_lens: list,
    outcomes: dict,
) -> None:
    elapsed = time.time() - t_start
    sps = global_step / elapsed if elapsed > 0 else 0

    # Episode / rally diagnostics
    n_eps = len(ep_lens)
    avg_len  = float(np.mean(ep_lens)) if ep_lens else 0.0
    pct_rally = sum(l > 1 for l in ep_lens) / n_eps if n_eps else 0.0

    # Compact outcome string: winner=W  net=N  out=O  unreach=U
    W = outcomes.get("winner", 0)
    N = outcomes.get("net_fault_0", 0)
    O = outcomes.get("out_of_bounds_0", 0)
    U = outcomes.get("unreachable_return", 0)
    on = outcomes.get("opponent_net", 0)
    oo = outcomes.get("opponent_out", 0)

    print(
        f"[{global_step:>10,}] "
        f"steps/s={sps:,.0f}  "
        f"win_rate={self_play.current_win_rate():.1%}  "
        f"elo={self_play.elo.active_elo:>7.1f}  "
        f"opp_elo={self_play.opponent_elo():>7.1f}  "
        f"pool={len(self_play.pool)}  "
        f"π_loss={stats['policy_loss']:+.4f}  "
        f"v_loss={stats['value_loss']:.4f}  "
        f"H={stats['entropy']:.3f}  "
        f"clip={stats['clip_fraction']:.2%}"
    )
    print(
        f"{'':>13}"
        f"eps={n_eps}  avg_len={avg_len:.1f}  rally%={pct_rally:.0%}  "
        f"[W:{W} N:{N} O:{O} U:{U} oN:{on} oO:{oo}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def run_training(cfg: Optional[TrainingConfig] = None) -> ActorCritic:
    if cfg is None:
        cfg = TrainingConfig()

    _set_seeds(cfg.seed)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"[train] Device:           {cfg.device}")
    print(f"[train] Total timesteps:  {cfg.total_timesteps:,}")
    print(f"[train] Rollout steps:    {cfg.rollout_steps}")
    print(f"[train] Env:              pure Python TennisEnv (no Unity)")

    # ── Components ────────────────────────────────────────────────────────────
    policy = ActorCritic(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        hidden_dims=cfg.hidden_dims,
    ).to(cfg.device)

    env        = TennisEnv()
    self_play  = SelfPlayManager(cfg, policy, env)
    buffer     = RolloutBuffer(cfg.rollout_steps, cfg.obs_dim, cfg.act_dim, cfg.device)
    ppo        = PPOOptimizer(policy, cfg)

    _training_loop(cfg, policy, buffer, ppo, self_play, env)
    return policy


def _training_loop(
    cfg:       TrainingConfig,
    policy:    ActorCritic,
    buffer:    RolloutBuffer,
    ppo:       PPOOptimizer,
    self_play: SelfPlayManager,
    env:       TennisEnv,
) -> None:
    global_step = 0
    t_start     = time.time()
    last_stats: dict = {}

    # Per-rollout episode diagnostics
    _rollout_ep_lens:  list = []
    _rollout_outcomes: dict = {}

    # Initial reset
    obs_np, _ = env.reset()
    obs = torch.tensor(obs_np, dtype=torch.float32, device=cfg.device).unsqueeze(0)  # (1, 24)
    _ep_len = 0

    while global_step < cfg.total_timesteps:

        # ── PHASE 1: Rollout ───────────────────────────────────────────────────
        policy.eval()
        buffer.clear()
        _rollout_ep_lens.clear()
        _rollout_outcomes.clear()

        for _ in range(cfg.rollout_steps):
            with torch.no_grad():
                action, x_pre, log_prob, value = policy.get_action(obs)

            obs_np_next, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            done = terminated or truncated

            buffer.add(
                obs       = obs.squeeze(0),
                x_pretanh = x_pre.squeeze(0),
                log_prob  = log_prob.squeeze(0),
                reward    = float(reward),
                value     = value.squeeze(0),
                done      = done,
            )

            global_step += 1
            _ep_len += 1

            if done:
                won = reward > 0
                self_play.record_episode_result(won, draw=(reward == 0.0))
                self_play.maybe_advance(global_step)

                outcome = info.get("outcome", "rally_trunc")
                _rollout_outcomes[outcome] = _rollout_outcomes.get(outcome, 0) + 1
                _rollout_ep_lens.append(_ep_len)
                _ep_len = 0
                obs_np_next, _ = env.reset()

            obs = torch.tensor(obs_np_next, dtype=torch.float32,
                               device=cfg.device).unsqueeze(0)

        # ── PHASE 2: GAE ───────────────────────────────────────────────────────
        with torch.no_grad():
            _, _, last_value = policy(obs)
            bootstrap = last_value.squeeze().item()

        buffer.compute_gae(bootstrap, cfg.gamma, cfg.gae_lambda)

        # ── PHASE 3: PPO update ────────────────────────────────────────────────
        policy.train()
        last_stats = ppo.update(buffer)

        # ── PHASE 4: Logging & checkpointing ──────────────────────────────────
        if global_step % cfg.log_every_steps < cfg.rollout_steps:
            _log(global_step, last_stats, self_play, t_start, _rollout_ep_lens, _rollout_outcomes)

        if global_step % cfg.save_every_steps < cfg.rollout_steps:
            elo_path = os.path.join(
                cfg.checkpoint_dir, f"elo_{global_step:010d}.json"
            )
            self_play.save_elo_snapshot(elo_path)

    print(
        f"\n[train] Done. {global_step:,} steps | "
        f"Final Elo: {self_play.elo.active_elo:.1f}"
    )
    # Final saves
    self_play.pool.save(
        model=policy, ckpt_id=0, elo=self_play.elo.active_elo,
        extra={"final": True, "global_step": global_step},
    )
    self_play.save_elo_snapshot(
        os.path.join(cfg.checkpoint_dir, "elo_final.json")
    )


if __name__ == "__main__":
    run_training(TrainingConfig())
