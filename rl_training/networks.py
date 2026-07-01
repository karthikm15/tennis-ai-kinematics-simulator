"""
Actor-Critic network for continuous tennis control.

Policy family: TanhNormal — samples from a diagonal Gaussian in pre-tanh
space, then squashes through tanh to enforce the [-1, 1] action bounds
mandated by the Unity action spec. The log-probability is corrected for
the Jacobian of the tanh transformation, which is critical for unbiased
PPO ratio computation.

Architecture:
    Shared encoder  →  Actor head (mean + state-independent log_std)
                    →  Critic head (scalar V(s))

Orthogonal initialization with gain √2 for hidden layers (standard RL
practice per Andrychowicz 2021), small gain 0.01 for the actor output
head to start near-deterministic, and gain 1.0 for the critic head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ortho_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Apply orthogonal weight init + zero bias to Linear / Conv layers."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    """
    Multi-layer MLP trunk shared by actor and critic.

    Interleaves Linear → LayerNorm → Tanh so gradient magnitude stays
    bounded independent of depth, without the instability of BatchNorm
    across small RL mini-batches.
    """

    def __init__(self, obs_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [
                _ortho_init(nn.Linear(in_dim, h), gain=np.sqrt(2)),
                nn.LayerNorm(h),
                nn.Tanh(),
            ]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs:    (B, obs_dim)
        # return: (B, out_dim)
        return self.net(obs)


class ActorHead(nn.Module):
    """
    Outputs (mean, log_std) for a TanhNormal policy.

    log_std is a learnable state-independent parameter vector (not a network
    head), which provides a clean exploration/exploitation dial and avoids
    the instability of predicting heteroscedastic variance from features.
    """

    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0

    def __init__(self, in_dim: int, act_dim: int) -> None:
        super().__init__()
        self.mean_layer = _ortho_init(nn.Linear(in_dim, act_dim), gain=0.01)
        # Shared log_std across the batch; learned independently of state.
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # features: (B, in_dim)
        mean = self.mean_layer(features)                           # (B, act_dim)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        log_std = log_std.expand_as(mean)                          # (B, act_dim)
        return mean, log_std


class CriticHead(nn.Module):
    """Scalar state-value estimator V(s)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.layer = _ortho_init(nn.Linear(in_dim, 1), gain=1.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, in_dim)
        # return:   (B, 1)
        return self.layer(features)


# ─────────────────────────────────────────────────────────────────────────────
# Main module
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic with a TanhNormal policy.

    The three public methods map to three distinct phases:
        forward()           — raw forward pass (mean, log_std, value)
        get_action()        — rollout collection (sample + log_prob + value)
        evaluate_actions()  — PPO optimization (re-evaluate stored actions)
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        self.encoder = SharedEncoder(obs_dim, hidden_dims)
        self.actor   = ActorHead(self.encoder.out_dim, act_dim)
        self.critic  = CriticHead(self.encoder.out_dim)

    # ── Core forward ──────────────────────────────────────────────────────────

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs    → (B, obs_dim)
        return → mean (B, act_dim), log_std (B, act_dim), value (B, 1)
        """
        features = self.encoder(obs)                    # (B, hidden_dim)
        mean, log_std = self.actor(features)            # (B, act_dim) each
        value = self.critic(features)                   # (B, 1)
        return mean, log_std, value

    # ── Rollout collection ────────────────────────────────────────────────────

    def get_action(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a stochastic action from the TanhNormal policy.

        obs        → (B, obs_dim)
        returns    → action      (B, act_dim)  — tanh-squashed ∈ [-1, 1]
                     x_pretanh  (B, act_dim)  — pre-tanh sample (store in buffer)
                     log_prob   (B,)
                     value      (B,)
        """
        mean, log_std, value = self(obs)                # (B, act_dim), …, (B, 1)
        dist = Normal(mean, log_std.exp())
        x = dist.rsample()                              # (B, act_dim) — reparameterized
        action = torch.tanh(x)                          # (B, act_dim) — env action
        log_prob = self._tanh_log_prob(dist, x)         # (B,)
        return action, x, log_prob, value.squeeze(-1)   # …, (B,)

    # ── PPO optimization ──────────────────────────────────────────────────────

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        x_pretanh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate stored actions under the *current* policy parameters.
        Called once per mini-batch during each PPO epoch.

        obs        → (B, obs_dim)
        x_pretanh  → (B, act_dim)  — pre-tanh actions from the buffer
        returns    → log_prob  (B,)
                     entropy   (B,)  — summed over action dimensions
                     value     (B,)
        """
        mean, log_std, value = self(obs)                   # (B, act_dim), …, (B, 1)
        dist = Normal(mean, log_std.exp())
        log_prob = self._tanh_log_prob(dist, x_pretanh)   # (B,)
        entropy   = dist.entropy().sum(-1)                 # (B,)  sum over act_dim
        return log_prob, entropy, value.squeeze(-1)        # (B,), (B,), (B,)

    # ── Evaluation (no sampling) ──────────────────────────────────────────────

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy deterministic action — use only during evaluation, not training."""
        mean, _, _ = self(obs)
        return torch.tanh(mean)                             # (B, act_dim)

    # ── Internal: TanhNormal log-prob ─────────────────────────────────────────

    @staticmethod
    def _tanh_log_prob(dist: Normal, x: torch.Tensor) -> torch.Tensor:
        """
        log π(a|s) under a TanhNormal distribution.

        The change-of-variables formula for a = tanh(x), x ~ N(μ,σ):

            log π(a|s) = log p(x|s) − Σ_i log(1 − tanh²(xᵢ))

        The Jacobian term is computed in a numerically stable form:
            log(1 − tanh²(x)) = 2·[log 2 − x − softplus(−2x)]

        x    → (B, act_dim)
        return (B,)   — summed over action dimensions
        """
        log_prob = dist.log_prob(x)                             # (B, act_dim)
        # Stable Jacobian correction (avoids log(0) when x is large)
        jacobian = 2.0 * (
            np.log(2.0) - x - F.softplus(-2.0 * x)
        )                                                       # (B, act_dim)
        return (log_prob - jacobian).sum(-1)                    # (B,)
