"""
Fixed-length rollout buffer with Generalized Advantage Estimation (GAE).

All tensors are pre-allocated on the target device at init time to avoid
repeated device transfers during hot-path rollout collection.

Buffer layout (N = rollout_steps):
    obs          (N, obs_dim)   — raw observations from env
    x_pretanh    (N, act_dim)   — pre-tanh Gaussian samples (NOT tanh(x))
    log_probs    (N,)           — log π_old(a|s) at collection time
    rewards      (N,)           — competitive ±1 / 0 rewards from wrapper
    values       (N,)           — V(s_t) from critic at collection time
    dones        (N,)           — 1.0 at episode terminal, 0.0 otherwise
    advantages   (N,)           — computed by compute_gae()
    returns      (N,)           — λ-returns = advantages + values

Note: x_pretanh (not action) is stored so that evaluate_actions() can
re-compute the exact TanhNormal log-prob under the updated policy.
"""
from __future__ import annotations

import torch
from typing import Iterator, Tuple


class RolloutBuffer:
    """
    Single-actor rollout buffer for on-policy PPO.

    Stores exactly `capacity` timesteps, then exposes randomized
    mini-batches for PPO optimization. Does NOT handle multiple parallel
    environments — that would require an extra leading dimension and a
    different flatten strategy.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        device: str,
    ) -> None:
        self.capacity  = capacity
        self.obs_dim   = obs_dim
        self.act_dim   = act_dim
        self.device    = device

        # Storage (pre-allocated; avoids per-step memory allocation)
        self.obs        = torch.zeros(capacity, obs_dim,  device=device)
        self.x_pretanh  = torch.zeros(capacity, act_dim,  device=device)
        self.log_probs  = torch.zeros(capacity,            device=device)
        self.rewards    = torch.zeros(capacity,            device=device)
        self.values     = torch.zeros(capacity,            device=device)
        self.dones      = torch.zeros(capacity,            device=device)
        self.advantages = torch.zeros(capacity,            device=device)
        self.returns    = torch.zeros(capacity,            device=device)

        self._ptr  = 0   # insertion pointer (never wraps; cleared after update)
        self._size = 0   # valid entries

    # ── Write path ────────────────────────────────────────────────────────────

    def add(
        self,
        obs:       torch.Tensor,   # (obs_dim,)
        x_pretanh: torch.Tensor,   # (act_dim,)
        log_prob:  torch.Tensor,   # scalar
        reward:    float,
        value:     torch.Tensor,   # scalar
        done:      bool,
    ) -> None:
        """Append one timestep. Raises if the buffer is already full."""
        if self._ptr >= self.capacity:
            raise RuntimeError(
                f"RolloutBuffer overflow: capacity={self.capacity}. "
                "Call clear() before adding new data."
            )
        i = self._ptr
        self.obs[i]        = obs.detach()
        self.x_pretanh[i]  = x_pretanh.detach()
        self.log_probs[i]  = log_prob.detach()
        self.rewards[i]    = reward
        self.values[i]     = value.detach()
        self.dones[i]      = float(done)
        self._ptr  += 1
        self._size  = self._ptr

    # ── GAE computation ───────────────────────────────────────────────────────

    def compute_gae(
        self,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute Generalized Advantage Estimates (Schulman 2016) in-place.

        GAE recurrence (backwards through time):

            δ_t  = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
            Â_t  = δ_t + (γλ) · (1 − done_t) · Â_{t+1}

        λ-returns are then:  G_t = Â_t + V(s_t)

        last_value: V(s_T) — critic bootstrap for the final truncated step.
        """
        N = self._size
        gae = 0.0

        for t in reversed(range(N)):
            non_terminal = 1.0 - self.dones[t].item()

            # Next-step value: use last_value for the very last step.
            next_val = self.values[t + 1].item() if t < N - 1 else last_value

            # TD residual
            delta = (
                self.rewards[t].item()
                + gamma * next_val * non_terminal
                - self.values[t].item()
            )

            # GAE recursion
            gae = delta + gamma * gae_lambda * non_terminal * gae
            self.advantages[t] = gae

        # λ-returns
        self.returns[:N] = self.advantages[:N] + self.values[:N]

    # ── Read path (PPO optimization) ──────────────────────────────────────────

    def get_minibatches(
        self, mini_batch_size: int
    ) -> Iterator[Tuple[torch.Tensor, ...]]:
        """
        Yield randomly-shuffled mini-batches for PPO update passes.

        Each batch is a tuple:
            obs        (B, obs_dim)
            x_pretanh  (B, act_dim)
            log_probs  (B,)
            advantages (B,)
            returns    (B,)

        Advantages are read *after* external normalization in PPOOptimizer.
        """
        N = self._size
        indices = torch.randperm(N, device=self.device)

        for start in range(0, N, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            yield (
                self.obs[idx],          # (B, obs_dim)
                self.x_pretanh[idx],    # (B, act_dim)
                self.log_probs[idx],    # (B,)
                self.advantages[idx],   # (B,)
                self.returns[idx],      # (B,)
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Reset insertion pointer without zeroing memory (faster)."""
        self._ptr  = 0
        self._size = 0

    def is_full(self) -> bool:
        return self._size >= self.capacity

    def __len__(self) -> int:
        return self._size
