"""
PPO optimizer with clipped surrogate objective + clipped value loss.

Algorithm
─────────
For each rollout buffer:
    1.  Normalize advantages across the full buffer (mean=0, std=1).
    2.  For `ppo_epochs` passes:
        For each mini-batch:
            a.  Re-evaluate stored (obs, x_pretanh) under the CURRENT policy
                to get new_log_prob, entropy, value.
            b.  Compute importance-sampling ratio ρ = exp(new_lp − old_lp).
            c.  Clipped policy loss:
                    L_CLIP = −min(ρ·Â,  clip(ρ, 1±ε)·Â)
            d.  Clipped value loss (prevents large critic updates):
                    L_VF   =  max(MSE(V, G), MSE(V_clip, G))
            e.  Entropy bonus:  L_H = −H[π]
            f.  Total:  L = L_CLIP + c_v·L_VF − c_H·L_H
    3.  Clip gradients by global norm (0.5 is standard).
    4.  Step Adam with a linear learning-rate annealing schedule.

Diagnostic tracking
────────────────────
The update() method returns a dict of averaged scalars that the training
loop writes to the log:
    policy_loss, value_loss, entropy, kl_approx, clip_fraction
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from buffer import RolloutBuffer
from config import TrainingConfig
from networks import ActorCritic


class PPOOptimizer:
    """
    Encapsulates the PPO update step and its optimizer state.

    Separate learning rates for encoder+actor vs. critic are applied via
    Adam parameter groups — the critic often benefits from a larger step
    because it lacks the stabilizing entropy term.
    """

    def __init__(self, policy: ActorCritic, cfg: TrainingConfig) -> None:
        self.policy = policy
        self.cfg = cfg

        self.optimizer = torch.optim.Adam(
            [
                # Encoder and actor share a smaller lr to keep the policy stable.
                {
                    "params": list(policy.encoder.parameters())
                            + list(policy.actor.parameters()),
                    "lr": cfg.lr_actor,
                },
                # Critic can safely use a larger lr to track value quickly.
                {
                    "params": policy.critic.parameters(),
                    "lr": cfg.lr_critic,
                },
            ],
            eps=1e-5,
        )

        # Linear LR annealing from 1× to 0.1× over total training.
        n_updates = cfg.total_timesteps // cfg.rollout_steps
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=max(n_updates, 1),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Run ppo_epochs optimization passes over the filled rollout buffer.

        The buffer must have had compute_gae() called before this.

        Returns
        -------
        dict with keys: policy_loss, value_loss, entropy, kl_approx, clip_fraction
        """
        self._normalize_advantages(buffer)

        stats = {
            "policy_loss":   [],
            "value_loss":    [],
            "entropy":       [],
            "kl_approx":     [],
            "clip_fraction": [],
        }

        for _ in range(self.cfg.ppo_epochs):
            for obs_b, x_b, old_lp_b, adv_b, ret_b in buffer.get_minibatches(
                self.cfg.mini_batch_size
            ):
                # obs_b    (B, obs_dim)
                # x_b      (B, act_dim)   — pre-tanh actions from buffer
                # old_lp_b (B,)           — log π_old(a|s)
                # adv_b    (B,)
                # ret_b    (B,)

                new_lp, entropy, value = self.policy.evaluate_actions(obs_b, x_b)
                # new_lp:  (B,)
                # entropy: (B,)  (summed over act_dim inside evaluate_actions)
                # value:   (B,)

                # ── Policy loss (clipped surrogate) ───────────────────────────

                # Importance-sampling ratio ρ_t = π_θ(a|s) / π_θ_old(a|s)
                ratio = torch.exp(new_lp - old_lp_b)               # (B,)

                policy_loss_raw   = -ratio * adv_b                  # (B,)
                policy_loss_clipped = -(
                    torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon,
                                       1.0 + self.cfg.clip_epsilon) * adv_b
                )                                                    # (B,)
                policy_loss = torch.max(policy_loss_raw, policy_loss_clipped).mean()

                # ── Value loss (clipped to prevent large critic swings) ────────

                # Reconstruct approximate old values from stored returns/advantages.
                # v_old ≈ G_t − Â_t  (since G = V + Â)
                v_old = ret_b - adv_b                               # (B,)

                value_unclipped = F.mse_loss(value, ret_b)
                value_clipped   = v_old + torch.clamp(
                    value - v_old,
                    -self.cfg.clip_epsilon,
                     self.cfg.clip_epsilon,
                )                                                    # (B,)
                value_loss = torch.max(
                    value_unclipped,
                    F.mse_loss(value_clipped, ret_b),
                )

                # ── Entropy bonus ─────────────────────────────────────────────

                entropy_loss = entropy.mean()                        # scalar

                # ── Combined loss ─────────────────────────────────────────────

                loss = (
                    policy_loss
                    + self.cfg.value_loss_coef * value_loss
                    - self.cfg.entropy_coef   * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                # ── Diagnostics (no grad needed) ──────────────────────────────
                with torch.no_grad():
                    # Approximate KL: E[(ratio−1) − log(ratio)] (Schulman 2017)
                    log_ratio = new_lp - old_lp_b
                    kl_approx = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = (
                        (torch.abs(ratio - 1) > self.cfg.clip_epsilon)
                        .float().mean().item()
                    )

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy_loss.item())
                stats["kl_approx"].append(kl_approx)
                stats["clip_fraction"].append(clip_frac)

        self.lr_scheduler.step()
        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize_advantages(buffer: RolloutBuffer) -> None:
        """Normalize advantages to zero-mean / unit-variance over the full buffer."""
        N = buffer._size
        adv = buffer.advantages[:N]
        buffer.advantages[:N] = (adv - adv.mean()) / (adv.std() + 1e-8)
