"""
CompetitiveTennisEnv — custom Python wrapper around the ML-Agents
Unity Tennis binary that enforces strict competitive zero-sum rules.

Design contract
───────────────
• STRIPS all Unity-native reward signals (the cooperative +0.1 for net
  clearance and time-penalty −0.01/step) from both teams.
• INJECTS a competitive reward from terminal event semantics:
      Team 0 faulted  →  r₀ = −1.0,  r₁ = +1.0
      Team 1 faulted  →  r₀ = +1.0,  r₁ = −1.0
      Simultaneous / interrupted episode → r₀ = r₁ = 0.0
• Returns separate (obs_0, obs_1) each step so each policy observes its
  own local reference frame as the Unity scene provides.
• Exposes get_obs_{0,1}() for the training loop to fetch the current
  observations without triggering a step.

Unity ML-Agents API used
─────────────────────────
  mlagents_envs >= 0.27 (Python API only — no ML-Agents trainer needed)
  Behavior names are resolved dynamically after reset(); the wrapper
  assumes sorted order maps to [team_0, team_1].

Fault detection heuristic
──────────────────────────
  The Unity Tennis.cs script writes reward = −1.0 into the TerminalStep
  of whichever agent dropped the ball (the "fault") and reward = 0.0 for
  the surviving agent. We read the sign of that reward only as a boolean
  fault flag, then completely discard its magnitude and replace it with
  our own competitive ±1.0 signal.
"""
from __future__ import annotations

import os

import numpy as np
from typing import Dict, Optional, Tuple

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)

from config import TrainingConfig


class CompetitiveTennisEnv:
    """
    Thin stateful adapter between the Unity binary and the PPO training loop.

    Public interface:
        reset()         → (obs_0, obs_1, info)
        step(a0, a1)    → (obs_0, obs_1, reward_0, terminated, truncated, info)
        close()         → None
        current_score   → (int, int)

    Parameters
    ----------
    cfg : TrainingConfig
        Provides env_path, env_port, time_scale, obs_dim, act_dim.
    worker_id : int
        Offset added to env_port so multiple training runs do not clash.
    """

    # These are resolved in __init__ after env.reset(); kept as instance attrs.
    _behavior_0: str
    _behavior_1: str

    def __init__(self, cfg: TrainingConfig, worker_id: int = 0) -> None:
        self.cfg = cfg

        # Resolve to an absolute path so mlagents_envs glob logic finds a bare
        # executable (no extension) via its exact-match fallback rather than the
        # extension-based glob that would miss a file named just "Tennis".
        env_abs_path = os.path.abspath(cfg.env_path)

        engine_channel = EngineConfigurationChannel()
        self._env = UnityEnvironment(
            file_name=env_abs_path,
            base_port=cfg.env_port + worker_id,
            side_channels=[engine_channel],
            no_graphics=True,
        )
        # Accelerate physics for faster sample collection during training.
        engine_channel.set_configuration_parameters(time_scale=cfg.time_scale)

        self._env.reset()
        behavior_names = sorted(self._env.behavior_specs.keys())
        if len(behavior_names) < 2:
            raise RuntimeError(
                f"Expected ≥2 behavior specs (one per team), found: {behavior_names}. "
                "Ensure the Tennis executable is configured for two-agent competitive play."
            )
        self._behavior_0 = behavior_names[0]
        self._behavior_1 = behavior_names[1]

        self._validate_specs()

        # Episode-level counters
        self._score_0: int = 0    # points won by team 0
        self._score_1: int = 0    # points won by team 1
        self._ep_steps: int = 0

        # Global stats
        self.episodes_completed: int = 0
        self.total_points_played: int = 0

        # Cached current obs (updated by reset/step)
        self._obs_0: np.ndarray = np.zeros((1, cfg.obs_dim), dtype=np.float32)
        self._obs_1: np.ndarray = np.zeros((1, cfg.obs_dim), dtype=np.float32)

    # ── Spec validation ───────────────────────────────────────────────────────

    def _validate_specs(self) -> None:
        for bname in (self._behavior_0, self._behavior_1):
            spec = self._env.behavior_specs[bname]
            assert spec.action_spec.continuous_size == self.cfg.act_dim, (
                f"Behavior '{bname}' has continuous_size="
                f"{spec.action_spec.continuous_size}, expected {self.cfg.act_dim}."
            )
            total_obs = sum(s.shape[0] for s in spec.observation_specs)
            assert total_obs == self.cfg.obs_dim, (
                f"Behavior '{bname}' observation total dim {total_obs} "
                f"≠ cfg.obs_dim {self.cfg.obs_dim}."
            )

    # ── Observation helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_obs(steps: DecisionSteps, obs_dim: int) -> np.ndarray:
        """
        Flatten all ObservationSpec arrays for each agent into one vector.

        steps.obs is a list of arrays, one per ObservationSpec in the behavior.
        For Tennis (single flat spec), it is [array(n_agents, 24)].
        Returns shape (n_agents, obs_dim) or (0, obs_dim) if no agents need decisions.
        """
        if len(steps) == 0:
            return np.zeros((0, obs_dim), dtype=np.float32)
        # Concatenate across observation specs along the feature axis
        return np.concatenate(
            [o.astype(np.float32) for o in steps.obs], axis=-1
        )  # (n_agents, obs_dim)

    def _zero_actions(self, n: int) -> ActionTuple:
        return ActionTuple(
            continuous=np.zeros((n, self.cfg.act_dim), dtype=np.float32)
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Reset the Unity episode.

        Returns
        -------
        obs_0 : (n_agents, obs_dim)  — initial observation for team 0
        obs_1 : (n_agents, obs_dim)  — initial observation for team 1
        info  : dict
        """
        self._env.reset()
        dec_0, _ = self._env.get_steps(self._behavior_0)
        dec_1, _ = self._env.get_steps(self._behavior_1)

        # Unity requires actions to be set before the first step.
        if len(dec_0) > 0:
            self._env.set_actions(self._behavior_0, self._zero_actions(len(dec_0)))
        if len(dec_1) > 0:
            self._env.set_actions(self._behavior_1, self._zero_actions(len(dec_1)))

        self._score_0 = 0
        self._score_1 = 0
        self._ep_steps = 0

        self._obs_0 = self._extract_obs(dec_0, self.cfg.obs_dim)
        self._obs_1 = self._extract_obs(dec_1, self.cfg.obs_dim)

        return self._obs_0, self._obs_1, {"score": (0, 0)}

    def step(
        self,
        action_0: np.ndarray,   # (n_agents, act_dim) — active policy actions
        action_1: np.ndarray,   # (n_agents, act_dim) — frozen policy actions
    ) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, Dict]:
        """
        Advance the simulation by one physics step.

        Returns
        -------
        obs_0       : (n_agents, obs_dim)
        obs_1       : (n_agents, obs_dim)
        reward_0    : float — competitive reward for team 0 this step
        terminated  : bool  — True when a point concludes (natural terminal)
        truncated   : bool  — True when interrupted by env timeout
        info        : dict
        """
        self._env.set_actions(
            self._behavior_0,
            ActionTuple(continuous=action_0.astype(np.float32)),
        )
        self._env.set_actions(
            self._behavior_1,
            ActionTuple(continuous=action_1.astype(np.float32)),
        )
        self._env.step()

        dec_0, term_0 = self._env.get_steps(self._behavior_0)
        dec_1, term_1 = self._env.get_steps(self._behavior_1)

        reward_0, terminated, truncated, point_label = self._resolve_reward(
            term_0, term_1
        )
        self._ep_steps += 1

        # After a terminal, Unity re-enqueues fresh initial obs in DecisionSteps.
        # Fall back to zeros for the one-step gap where the env re-initialises.
        if len(dec_0) > 0:
            self._obs_0 = self._extract_obs(dec_0, self.cfg.obs_dim)
        if len(dec_1) > 0:
            self._obs_1 = self._extract_obs(dec_1, self.cfg.obs_dim)

        if terminated or truncated:
            self.episodes_completed += 1

        info: Dict = {
            "score":        (self._score_0, self._score_1),
            "episode_steps": self._ep_steps,
            "point_label":  point_label,  # "win" | "loss" | "draw" | "ongoing"
        }
        return self._obs_0, self._obs_1, reward_0, terminated, truncated, info

    # ── Reward resolution ─────────────────────────────────────────────────────

    def _resolve_reward(
        self,
        term_0: TerminalSteps,
        term_1: TerminalSteps,
    ) -> Tuple[float, bool, bool, str]:
        """
        Translate Unity terminal events into a competitive ±1 / 0 reward.

        Unity semantic:  reward < 0 in TerminalSteps ⟹ that agent faulted.
        We IGNORE reward magnitude; sign is used only as a fault indicator.

        Returns (reward_0, terminated, truncated, label).
        """
        if len(term_0) == 0 and len(term_1) == 0:
            return 0.0, False, False, "ongoing"

        self.total_points_played += 1

        # interrupted=True means the episode hit a max-step cutoff (truncation).
        ep0_interrupted = len(term_0) > 0 and bool(term_0.interrupted.any())
        ep1_interrupted = len(term_1) > 0 and bool(term_1.interrupted.any())
        truncated = ep0_interrupted or ep1_interrupted

        if truncated:
            return 0.0, False, True, "truncated"

        # Fault detection: negative Unity terminal reward ⟹ that team dropped the ball.
        team_0_faulted = (
            len(term_0) > 0 and bool((term_0.reward < -1e-6).any())
        )
        team_1_faulted = (
            len(term_1) > 0 and bool((term_1.reward < -1e-6).any())
        )

        if team_0_faulted and not team_1_faulted:
            self._score_1 += 1
            return -1.0, True, False, "loss"

        if team_1_faulted and not team_0_faulted:
            self._score_0 += 1
            return +1.0, True, False, "win"

        # Simultaneous fault (e.g. both out-of-bounds at the same tick) → draw.
        return 0.0, True, False, "draw"

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def current_score(self) -> Tuple[int, int]:
        return self._score_0, self._score_1

    @property
    def obs_0(self) -> np.ndarray:
        return self._obs_0

    @property
    def obs_1(self) -> np.ndarray:
        return self._obs_1

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._env.close()

    def __enter__(self) -> "CompetitiveTennisEnv":
        return self

    def __exit__(self, *_) -> None:
        self.close()
