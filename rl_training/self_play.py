"""
Adversarial self-play infrastructure for competitive tennis training.

Three cooperating components:

EloTracker
    Maintains Elo ratings for every saved checkpoint and for the live
    active policy. After every episode the tracker updates both participants'
    ratings via the standard FIDE formula:
        E_A = 1 / (1 + 10^((R_B − R_A) / 400))
        R_A ← R_A + K·(S_A − E_A)
    Elo lets us measure absolute skill progression across millions of
    environment steps even when opponents change frequently.

CheckpointPool
    Stores frozen copies of the active policy as it evolves. Implements
    the recency-biased sampling strategy:
        P(most-recent opponent) = recent_opponent_prob  (default 0.80)
        P(random older checkpoint) = 1 − recent_opponent_prob

SelfPlayManager
    Orchestrates the full loop:
        1. Holds a reference to TennisEnv and sets env.opponent_fn to point
           at the current frozen policy's get_action method.
        2. Active policy trains against the current frozen opponent.
        3. Episode outcomes are recorded in a rolling window.
        4. When win_rate > threshold over eval_window episodes,
           the active policy is frozen as a new checkpoint.
        5. A new opponent is sampled from the pool and the env is updated.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from config import TrainingConfig
from networks import ActorCritic

if TYPE_CHECKING:
    from tennis_env import TennisEnv


# ─────────────────────────────────────────────────────────────────────────────
# Elo tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckpointRecord:
    ckpt_id:  int
    path:     str
    elo:      float = 1200.0
    wins:     int   = 0
    losses:   int   = 0
    draws:    int   = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games > 0 else 0.0

    def as_dict(self) -> Dict:
        return {
            "ckpt_id":  self.ckpt_id,
            "elo":      round(self.elo, 1),
            "win_rate": round(self.win_rate, 4),
            "wins":     self.wins,
            "losses":   self.losses,
            "draws":    self.draws,
            "games":    self.games,
        }


class EloTracker:
    K: float           = 8.0    # reduced from 32 — fewer games per window than tournament play
    INITIAL_ELO: float = 1200.0

    def __init__(self) -> None:
        self.checkpoints: Dict[int, CheckpointRecord] = {}
        self.active_elo: float = self.INITIAL_ELO
        self._active_wins:   int = 0
        self._active_losses: int = 0
        self._active_draws:  int = 0

    def register_checkpoint(
        self, ckpt_id: int, path: str, inherit_elo: Optional[float] = None
    ) -> None:
        self.checkpoints[ckpt_id] = CheckpointRecord(
            ckpt_id=ckpt_id,
            path=path,
            elo=inherit_elo if inherit_elo is not None else self.INITIAL_ELO,
        )

    def record_match(self, active_won: bool, opponent_id: int, draw: bool = False) -> None:
        opp = self.checkpoints[opponent_id]
        E_active = self._expected(self.active_elo, opp.elo)
        E_opp    = 1.0 - E_active

        if draw:
            S_active, S_opp = 0.5, 0.5
            self._active_draws += 1
            opp.draws          += 1
        elif active_won:
            S_active, S_opp = 1.0, 0.0
            self._active_wins += 1
            opp.losses        += 1
        else:
            S_active, S_opp = 0.0, 1.0
            self._active_losses += 1
            opp.wins            += 1

        self.active_elo += self.K * (S_active - E_active)
        opp.elo         += self.K * (S_opp    - E_opp)

    @property
    def active_win_rate(self) -> float:
        total = self._active_wins + self._active_losses + self._active_draws
        return self._active_wins / total if total > 0 else 0.0

    @staticmethod
    def _expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    def snapshot(self) -> Dict:
        return {
            "active": {
                "elo":      round(self.active_elo, 1),
                "win_rate": round(self.active_win_rate, 4),
                "wins":     self._active_wins,
                "losses":   self._active_losses,
                "draws":    self._active_draws,
            },
            "checkpoints": {
                str(cid): rec.as_dict()
                for cid, rec in sorted(self.checkpoints.items())
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint pool
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointPool:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self._records: List[CheckpointRecord] = []
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    def save(
        self,
        model:    ActorCritic,
        ckpt_id:  int,
        elo:      float,
        extra:    Optional[Dict] = None,
    ) -> CheckpointRecord:
        path = os.path.join(self.cfg.checkpoint_dir, f"ckpt_{ckpt_id:07d}.pt")
        torch.save(
            {"model_state": model.state_dict(), "ckpt_id": ckpt_id, "elo": elo, **(extra or {})},
            path,
        )
        rec = CheckpointRecord(ckpt_id=ckpt_id, path=path, elo=elo)
        self._records.append(rec)
        return rec

    def load_into(self, model: ActorCritic, record: CheckpointRecord) -> None:
        state = torch.load(record.path, map_location=self.cfg.device, weights_only=True)
        model.load_state_dict(state["model_state"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    def sample(self) -> CheckpointRecord:
        if not self._records:
            raise RuntimeError("CheckpointPool is empty.")
        if len(self._records) == 1:
            return self._records[-1]
        if random.random() < self.cfg.recent_opponent_prob:
            return self._records[-1]
        return random.choice(self._records)

    def __len__(self) -> int:
        return len(self._records)


# ─────────────────────────────────────────────────────────────────────────────
# Self-play manager
# ─────────────────────────────────────────────────────────────────────────────

class SelfPlayManager:
    """
    Coordinates the active policy, the frozen opponent pool, and the env.

    Key design: holds a direct reference to TennisEnv and sets
    `env.opponent_fn` whenever the frozen opponent changes, so the env
    always calls the current checkpoint without any external wiring in
    the training loop.
    """

    def __init__(
        self,
        cfg:           TrainingConfig,
        active_policy: ActorCritic,
        env:           "TennisEnv",
    ) -> None:
        self.cfg           = cfg
        self.active_policy = active_policy
        self.env           = env
        self.pool          = CheckpointPool(cfg)
        self.elo           = EloTracker()
        self._counter      = 0
        self._window: List[Optional[bool]] = []

        # Frozen policy — a separate model instance, never aliased with active
        self.frozen_policy = ActorCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dims=cfg.hidden_dims,
        ).to(cfg.device)

        self._current_opponent: Optional[CheckpointRecord] = None
        self._step_at_last_freeze: int = 0

        # Seed the pool with the initial random policy
        self._do_freeze(seeding=True)

    # ── Frozen-policy inference (called by env via opponent_fn) ───────────────

    @torch.no_grad()
    def get_frozen_action(self, obs: torch.Tensor) -> np.ndarray:
        """
        obs : (1, obs_dim) tensor — team 1's observation
        Returns (act_dim,) numpy array
        """
        action, _, _, _ = self.frozen_policy.get_action(obs)
        return action.squeeze(0).cpu().numpy()   # (act_dim,)

    # ── Episode result tracking ───────────────────────────────────────────────

    def record_episode_result(self, active_won: bool, draw: bool = False) -> None:
        if self._current_opponent is not None:
            self.elo.record_match(
                active_won=active_won,
                opponent_id=self._current_opponent.ckpt_id,
                draw=draw,
            )
        self._window.append(None if draw else active_won)
        if len(self._window) > self.cfg.eval_window:
            self._window.pop(0)

    def maybe_advance(self, global_step: int) -> bool:
        if global_step - self._step_at_last_freeze < self.cfg.min_steps_between_freezes:
            return False
        if len(self._window) < self.cfg.eval_window:
            return False
        win_rate = self._current_win_rate()
        if win_rate < self.cfg.win_rate_threshold:
            return False
        print(
            f"[SelfPlay] Win rate {win_rate:.1%} ≥ threshold "
            f"{self.cfg.win_rate_threshold:.1%} — freezing policy "
            f"(Elo={self.elo.active_elo:.1f})."
        )
        self._step_at_last_freeze = global_step
        self._do_freeze(seeding=False)
        return True

    # ── Accessors ─────────────────────────────────────────────────────────────

    def current_win_rate(self) -> float:
        return self._current_win_rate()

    def opponent_elo(self) -> float:
        if self._current_opponent is None:
            return EloTracker.INITIAL_ELO
        return self.elo.checkpoints[self._current_opponent.ckpt_id].elo

    def save_elo_snapshot(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.elo.snapshot(), fh, indent=2)
        print(f"[EloTracker] Snapshot → {path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _do_freeze(self, seeding: bool = False) -> None:
        self._counter += 1
        ckpt_id = self._counter
        rec = self.pool.save(
            model=self.active_policy,
            ckpt_id=ckpt_id,
            elo=self.elo.active_elo,   # stored for logging only
            extra={"seeding": seeding},
        )
        # New checkpoints always start at INITIAL_ELO — inheriting the parent's
        # inflated rating causes unbounded Elo growth because every match is
        # equal-Elo, the active always wins, and ratings compound each cycle.
        self.elo.register_checkpoint(ckpt_id, rec.path, inherit_elo=None)

        # Sample next opponent and wire it into the env
        new_opp = self.pool.sample()
        self.pool.load_into(self.frozen_policy, new_opp)
        self._current_opponent = new_opp

        # This is the key line — env.opponent_fn is now the frozen checkpoint
        self.env.opponent_fn = self.get_frozen_action

        self._window.clear()
        label = "seed" if seeding else "new checkpoint"
        print(
            f"[SelfPlay] {label} #{ckpt_id} frozen. "
            f"Next opponent: #{new_opp.ckpt_id} (Elo={new_opp.elo:.1f})."
        )

    def _current_win_rate(self) -> float:
        definite = [r for r in self._window if r is not None]
        return sum(definite) / len(definite) if definite else 0.0
