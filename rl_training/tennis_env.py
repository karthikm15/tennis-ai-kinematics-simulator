"""
Pure Python 2D tennis environment — no Unity dependency.

Court coordinates match court.ts exactly so trained weights integrate
directly into the browser UI without any coordinate remapping.

    x ∈ [0, 8.23]   court width
    y ∈ [0, 23.77]  court length
    net at y = 11.885
    player half  y ∈ [0,      11.885]   (team 1 / "player" side)
    AI half      y ∈ [11.885, 23.77]   (team 0 / "AI"     side)

Turn-based episode structure
─────────────────────────────
Each call to step() represents team 0 (the active policy) taking ONE shot.
Internally the env simulates team 1's response via `opponent_fn` before
returning control. This means the training loop only ever passes ONE action
per step, matching the buffer's single-actor design.

Episode flow:
    reset()         → team 0 starts at their baseline, ball at net, obs returned
    step(action_0)  → team 0 hits, team 1 responds internally, new obs returned
    ...             → repeat until someone can't reach or hits out/into net
    terminated=True → return ±1 reward, episode is over

Observation (24-dim = 3 stacked frames × 8):
    [self_x, self_y, self_vx, self_vy, opp_x, opp_y, ball_x, ball_y]
    All normalized to [-1, 1]. team 1's obs is y-mirrored so a single
    policy (same weights) plays equally well from either side of the net.

Action (2-dim, tanh-squashed ∈ [-1, 1]):
    action[0] → landing_x on opponent's half (same formula both sides)
    action[1] → landing_y on opponent's half (remapped per which side is shooting)

UI integration
──────────────
After training, call `action_to_court_coords(action, shooting_from_ai_side=True)`
to get the {x, y} Vec2 that replaces `generateAiShot(...).landing` in
useGameState.ts. No other changes to the UI are needed.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

# ── Court constants (exact match to court.ts) ─────────────────────────────────

COURT_WIDTH         = 8.23
COURT_LENGTH        = 23.77
NET_Y               = 11.885   # COURT.netYM
NET_HEIGHT          = 0.914    # COURT.netHeightCenterM
SERVICE_BOX_DEPTH   = 6.40    # COURT.serviceBoxDepthM
PLAYER_BASELINE_Y   = 0.0     # COURT.playerBaselineY
AI_BASELINE_Y       = 23.77   # COURT.aiBaselineY

# Minimum gap from sidelines / baselines when choosing a landing spot
MARGIN = 0.3

# Shots must land at least NET_MARGIN metres from the net on each side.
# Without this buffer, the parabolic arc physics break down for near-net shots:
# MIN_TRAVEL_TIME forces a flat trajectory that cannot clear the 0.914 m net.
NET_MARGIN = 2.0

# ── Movement constants (matching kinematics.ts) ───────────────────────────────

AI_MAX_SPEED  = 5.5   # AI_MAX_SPEED_MS
OPP_MAX_SPEED = 6.0   # PLAYER_MAX_SPEED_MS

# ── Shot physics ──────────────────────────────────────────────────────────────

SHOT_SPEED      = 10.0  # m/s  (fixed rally speed — matches mid-range in ai.ts)
MIN_TRAVEL_TIME = 0.8   # seconds  (matching validateReturn)
RALLY_CONTINUE_REWARD = 0.025
HIT_AWAY_BONUS_SCALE = 0.05
DEPTH_BONUS_SCALE = 0.03
CENTER_SPAM_PENALTY = 0.025
EXTREME_CORNER_PENALTY = 0.06
CHEAP_WINNER_PENALTY = 0.25
LONG_RALLY_SHOT_COUNT = 4

# ── Observation ───────────────────────────────────────────────────────────────

N_FRAMES    = 3
N_FEATURES  = 8         # per frame
OBS_DIM     = N_FRAMES * N_FEATURES   # 24  ← must match TrainingConfig.obs_dim
ACT_DIM     = 2                       # ← must match TrainingConfig.act_dim

# Normalization denominators
_X_NORM = COURT_WIDTH / 2.0    # normalise x: divide, then subtract 1
_Y_NORM = COURT_LENGTH / 2.0   # normalise y
_V_NORM = max(AI_MAX_SPEED, OPP_MAX_SPEED)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _norm_x(x: float) -> float:
    return x / _X_NORM - 1.0   # [0, W] → [-1, 1]

def _norm_y(y: float) -> float:
    return y / _Y_NORM - 1.0   # [0, L] → [-1, 1]

def _norm_v(v: float) -> float:
    return np.clip(v / _V_NORM, -1.0, 1.0)


def action_to_court_coords(
    action: np.ndarray,
    shooting_from_ai_side: bool = True,
) -> Dict[str, float]:
    """
    Convert a 2D tanh-squashed action → court landing coordinate.

    This is the KEY integration function. The returned dict {x, y} is the
    `landing` field you pass into the Shot constructor in useGameState.ts,
    directly replacing generateAiShot(...).landing.

    Parameters
    ----------
    action              (2,) array, values ∈ [-1, 1]
    shooting_from_ai_side  True if the AI (upper half) is hitting toward the
                            player's half (y < NET_Y). False for the reverse.
    """
    ax, ay = float(action[0]), float(action[1])
    # Map [-1, 1] → [MARGIN, WIDTH - MARGIN]
    land_x = MARGIN + (ax + 1.0) / 2.0 * (COURT_WIDTH - 2.0 * MARGIN)
    # Map [-1, 1] → the valid depth range in the opponent's half.
    # Shots must stay NET_MARGIN metres from the net: the parabolic arc
    # physics produce a too-flat trajectory for near-net shots that uses
    # MIN_TRAVEL_TIME, causing spurious "opponent hits net" outcomes.
    if shooting_from_ai_side:
        # Target: player half, y ∈ [MARGIN, NET_Y - NET_MARGIN]
        y_lo = MARGIN
        y_hi = NET_Y - NET_MARGIN          # 9.885 m
        land_y = y_lo + (ay + 1.0) / 2.0 * (y_hi - y_lo)
    else:
        # Target: AI half, y ∈ [NET_Y + NET_MARGIN, COURT_LENGTH - MARGIN]
        y_lo = NET_Y + NET_MARGIN          # 13.885 m
        y_hi = COURT_LENGTH - MARGIN       # 23.47 m
        land_y = y_lo + (ay + 1.0) / 2.0 * (y_hi - y_lo)

    land_x = float(np.clip(land_x, 0.0, COURT_WIDTH))
    land_y = float(np.clip(land_y, 0.0, COURT_LENGTH))
    return {"x": land_x, "y": land_y}


# ── Internal physics helpers ──────────────────────────────────────────────────

def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def _can_reach(
    from_x: float, from_y: float,
    to_x:   float, to_y:   float,
    max_speed: float,
    travel_time: float,
) -> bool:
    """Matches canReach() in kinematics.ts exactly."""
    return _dist(from_x, from_y, to_x, to_y) <= max_speed * travel_time


def _travel_time(origin_x: float, origin_y: float,
                 land_x: float, land_y: float) -> float:
    d = _dist(origin_x, origin_y, land_x, land_y)
    return max(d / SHOT_SPEED, MIN_TRAVEL_TIME)


def _in_player_half(y: float) -> bool:
    return PLAYER_BASELINE_Y <= y <= NET_Y


def _in_ai_half(y: float) -> bool:
    return NET_Y <= y <= AI_BASELINE_Y


def _in_court_x(x: float) -> bool:
    return 0.0 <= x <= COURT_WIDTH


def _net_clearance(
    from_x: float, from_y: float,
    to_x:   float, to_y:   float,
) -> bool:
    """
    Simplified net clearance check matching checkNetClearance() in kinematics.ts.
    The ball must cross above NET_HEIGHT when it passes y = NET_Y.
    Uses the same parabolic arc with HIT_HEIGHT = 1.0m contact height.
    """
    HIT_HEIGHT = 1.0
    GRAVITY    = 9.81

    total_dist = _dist(from_x, from_y, to_x, to_y)
    if total_dist < 0.01:
        return False

    t_land = total_dist / SHOT_SPEED
    t_land = max(t_land, MIN_TRAVEL_TIME)

    # vz that puts ball at ground level at t_land (from HIT_HEIGHT)
    vz = 0.5 * GRAVITY * t_land - HIT_HEIGHT / t_land

    # Fraction of travel where ball crosses the net
    min_y, max_y = min(from_y, to_y), max(from_y, to_y)
    if NET_Y < min_y or NET_Y > max_y:
        return True   # shot doesn't cross the net line

    frac_to_net = abs(NET_Y - from_y) / abs(to_y - from_y + 1e-9)
    t_net = frac_to_net * t_land

    ball_z_at_net = HIT_HEIGHT + vz * t_net - 0.5 * GRAVITY * t_net ** 2
    return ball_z_at_net >= NET_HEIGHT


# ── Heuristic opponent policy (mirrors ai.ts generateAiShot logic) ────────────

def _heuristic_action(
    self_x: float, self_y: float,
    opp_x:  float, opp_y:  float,
    shooting_from_ai_side: bool,
) -> np.ndarray:
    """
    Deterministic heuristic: 70% aim to the side the opponent is NOT on.
    Returns a raw [-1, 1] action array (same format as the policy network output).
    """
    mid_x = COURT_WIDTH / 2.0
    if np.random.random() < 0.70:
        if opp_x < mid_x:
            # Opponent is left → aim right
            target_x = mid_x + MARGIN + np.random.random() * (mid_x - 2 * MARGIN)
        else:
            # Opponent is right → aim left
            target_x = MARGIN + np.random.random() * (mid_x - 2 * MARGIN)
    else:
        target_x = MARGIN + np.random.random() * (COURT_WIDTH - 2 * MARGIN)

    # y: aim to the middle 60% of the target half
    y_range = NET_Y - 2 * MARGIN
    target_y_half = MARGIN + 0.2 * y_range + np.random.random() * 0.6 * y_range

    # Convert back to [-1, 1] action space
    ax = (target_x - MARGIN) / (COURT_WIDTH - 2 * MARGIN) * 2.0 - 1.0
    ay = (target_y_half - MARGIN) / (y_range - MARGIN) * 2.0 - 1.0

    return np.array([ax, ay], dtype=np.float32)


# ── Main environment class ────────────────────────────────────────────────────

class TennisEnv:
    """
    Pure Python competitive tennis environment.

    The active policy (team 0) controls the AI racket (upper half of court).
    The frozen opponent policy (team 1) controls the player racket (lower half).

    Set `opponent_fn` to a callable (obs_tensor → np.ndarray action) to drive
    team 1 with the frozen PPO policy. When None, the built-in heuristic is used.
    """

    def __init__(self) -> None:
        # Mutable — updated by SelfPlayManager when a new checkpoint is loaded
        self.opponent_fn: Optional[Callable[[torch.Tensor], np.ndarray]] = None

        # Internal state
        self._ai_x: float = 0.0
        self._ai_y: float = 0.0
        self._ai_vx: float = 0.0
        self._ai_vy: float = 0.0
        self._opp_x: float = 0.0
        self._opp_y: float = 0.0
        self._opp_vx: float = 0.0
        self._opp_vy: float = 0.0
        self._last_ball_x: float = 0.0
        self._last_ball_y: float = 0.0

        # 3-frame observation buffers (one per team)
        self._frame_buf: deque = deque(maxlen=N_FRAMES)
        self._opp_frame_buf: deque = deque(maxlen=N_FRAMES)

        # Stats
        self.episodes_completed: int = 0
        self.total_shots: int = 0
        self._rally_len: int = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def reset(self) -> Tuple[np.ndarray, Dict]:
        """
        Reset to start of a new point. AI starts near its baseline,
        player starts near theirs.

        Returns: obs (24,), info dict
        """
        # Randomise full starting positions within each player's half.
        # This prevents the agent exploiting a fixed "aim to unreachable corner"
        # strategy — sometimes the opponent is near the net (making drop shots
        # reachable), sometimes at the baseline, so the agent must learn to aim
        # based on WHERE the opponent actually is.
        self._ai_x  = float(np.random.uniform(MARGIN, COURT_WIDTH - MARGIN))
        self._ai_y  = float(np.random.uniform(NET_Y + 1.0, AI_BASELINE_Y - MARGIN))
        self._ai_vx = 0.0
        self._ai_vy = 0.0

        self._opp_x  = float(np.random.uniform(MARGIN, COURT_WIDTH - MARGIN))
        self._opp_y  = float(np.random.uniform(PLAYER_BASELINE_Y + MARGIN, NET_Y - 1.0))
        self._opp_vx = 0.0
        self._opp_vy = 0.0

        # Ball starts at mid-court (serve is first action)
        self._last_ball_x = COURT_WIDTH / 2.0
        self._last_ball_y = NET_Y
        self._rally_len = 0

        # Fill both frame buffers with the initial frame (3× to avoid zero-padding)
        initial_frame = self._build_frame_ai()
        self._frame_buf.clear()
        for _ in range(N_FRAMES):
            self._frame_buf.append(initial_frame)

        initial_opp_frame = self._build_frame_opp()
        self._opp_frame_buf.clear()
        for _ in range(N_FRAMES):
            self._opp_frame_buf.append(initial_opp_frame)

        return self._get_obs(), {"score": (0, 0)}

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self, action_0: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Advance one full shot exchange.

        1. Team 0 (AI) hits using action_0.
        2. Check if team 1 (player) can reach; if not → AI wins point.
        3. Team 1 responds using opponent_fn (or heuristic).
        4. Check if team 0 (AI) can reach the return; if not → player wins point.
        5. Return updated obs for next AI shot.

        Returns: obs(24,), reward_0, terminated, truncated, info
        """
        self.total_shots += 1

        # ── Team 0 (AI) shoots toward player's half ───────────────────────────
        ai_landing = action_to_court_coords(action_0, shooting_from_ai_side=True)
        land_x_0, land_y_0 = ai_landing["x"], ai_landing["y"]

        # Validity checks
        if not (_in_court_x(land_x_0) and _in_player_half(land_y_0)):
            return self._terminal(-1.0, "out_of_bounds_0")

        t_fly_0 = _travel_time(self._ai_x, self._ai_y, land_x_0, land_y_0)

        if not _net_clearance(self._ai_x, self._ai_y, land_x_0, land_y_0):
            return self._terminal(-1.0, "net_fault_0")

        shaped_reward = self._shot_quality_reward(
            land_x_0,
            land_y_0,
            self._opp_x,
            self._opp_y,
        )

        # Did player reach in time?
        opp_reached = _can_reach(
            self._opp_x, self._opp_y,
            land_x_0, land_y_0,
            OPP_MAX_SPEED, t_fly_0,
        )
        if not opp_reached:
            # AI wins the point
            self._update_positions_0(land_x_0, land_y_0, t_fly_0)
            winner_reward = 1.0 + shaped_reward
            if self._rally_len < LONG_RALLY_SHOT_COUNT:
                winner_reward -= CHEAP_WINNER_PENALTY
            return self._terminal(winner_reward, "winner")

        # Player reaches ball — update their position
        self._update_positions_0(land_x_0, land_y_0, t_fly_0)

        # ── Team 1 (player / frozen) shoots back toward AI half ───────────────
        # Update ball position so team 1's obs reflects where the ball just landed
        self._last_ball_x = land_x_0
        self._last_ball_y = land_y_0
        obs_1 = self._build_obs_1()
        obs_1_tensor = torch.tensor(obs_1, dtype=torch.float32).unsqueeze(0)

        if self.opponent_fn is not None:
            with torch.no_grad():
                action_1_np = self.opponent_fn(obs_1_tensor)
        else:
            action_1_np = _heuristic_action(
                self._opp_x, self._opp_y,
                self._ai_x,  self._ai_y,
                shooting_from_ai_side=False,
            )

        ai_return = action_to_court_coords(
            action_1_np.flatten(), shooting_from_ai_side=False
        )
        land_x_1, land_y_1 = ai_return["x"], ai_return["y"]

        if not (_in_court_x(land_x_1) and _in_ai_half(land_y_1)):
            return self._terminal(+1.0, "opponent_out")

        t_fly_1 = _travel_time(land_x_0, land_y_0, land_x_1, land_y_1)

        if not _net_clearance(land_x_0, land_y_0, land_x_1, land_y_1):
            return self._terminal(+1.0, "opponent_net")

        # Did AI reach the return in time?
        ai_reached = _can_reach(
            self._ai_x, self._ai_y,
            land_x_1, land_y_1,
            AI_MAX_SPEED, t_fly_1,
        )
        if not ai_reached:
            self._update_positions_1(land_x_1, land_y_1, t_fly_1)
            return self._terminal(-1.0, "unreachable_return")

        # Both reached — rally continues. Update AI position for next shot.
        self._update_positions_1(land_x_1, land_y_1, t_fly_1)
        self._last_ball_x = land_x_1
        self._last_ball_y = land_y_1
        self._rally_len += 1

        # Push new frame
        self._frame_buf.append(self._build_frame_ai())

        return self._get_obs(), RALLY_CONTINUE_REWARD + shaped_reward, False, False, {
            "shot_outcome": "rally",
            "ai_pos": (self._ai_x, self._ai_y),
        }

    def _shot_quality_reward(
        self,
        land_x: float,
        land_y: float,
        opp_x: float,
        opp_y: float,
    ) -> float:
        lateral_pressure = min(abs(land_x - opp_x) / COURT_WIDTH, 1.0)
        depth = 1.0 - min(max(land_y - PLAYER_BASELINE_Y, 0.0) / NET_Y, 1.0)
        center_distance = abs(land_x - COURT_WIDTH / 2.0)
        center_penalty = CENTER_SPAM_PENALTY if center_distance < 0.9 else 0.0
        extreme_corner = (land_x < 0.9 or land_x > COURT_WIDTH - 0.9) and land_y < 2.0
        corner_penalty = EXTREME_CORNER_PENALTY if extreme_corner else 0.0
        return (
            HIT_AWAY_BONUS_SCALE * lateral_pressure
            + DEPTH_BONUS_SCALE * depth
            - center_penalty
            - corner_penalty
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_positions_0(
        self, land_x: float, land_y: float, t: float
    ) -> None:
        """AI hit — player moves to the ball; AI stays put after hitting."""
        # AI velocity (just hit — no movement this shot)
        self._ai_vx = 0.0
        self._ai_vy = 0.0

        # Player moves toward landing
        dx = land_x - self._opp_x
        dy = land_y - self._opp_y
        d  = _dist(self._opp_x, self._opp_y, land_x, land_y)
        if d > 1e-6:
            frac = min(1.0, OPP_MAX_SPEED * t / d)
            new_x = self._opp_x + dx * frac
            new_y = self._opp_y + dy * frac
            self._opp_vx = (new_x - self._opp_x) / t
            self._opp_vy = (new_y - self._opp_y) / t
            self._opp_x  = new_x
            self._opp_y  = new_y

    def _update_positions_1(
        self, land_x: float, land_y: float, t: float
    ) -> None:
        """Player hit — AI moves to the return landing."""
        dx = land_x - self._ai_x
        dy = land_y - self._ai_y
        d  = _dist(self._ai_x, self._ai_y, land_x, land_y)
        if d > 1e-6:
            frac = min(1.0, AI_MAX_SPEED * t / d)
            new_x = self._ai_x + dx * frac
            new_y = self._ai_y + dy * frac
            self._ai_vx = (new_x - self._ai_x) / t
            self._ai_vy = (new_y - self._ai_y) / t
            self._ai_x  = new_x
            self._ai_y  = new_y

        # Player stops after hitting
        self._opp_vx = 0.0
        self._opp_vy = 0.0

    def _build_frame_ai(self) -> np.ndarray:
        """
        Build one 8-dim observation frame from AI's perspective.
        y-coords are expressed as distance-from-AI-baseline so the observation
        is symmetric: both agents see "I am at ~y=0, opponent is far away."
        """
        # From AI's POV: AI baseline is at y = COURT_LENGTH (23.77)
        # Flip y: y_ai = COURT_LENGTH - actual_y  (AI baseline → 0, net → 11.885)
        ai_y_flipped  = COURT_LENGTH - self._ai_y
        opp_y_flipped = COURT_LENGTH - self._opp_y
        ball_y_flipped = COURT_LENGTH - self._last_ball_y

        return np.array([
            _norm_x(self._ai_x),
            _norm_y(ai_y_flipped),
            _norm_v(self._ai_vx),
            _norm_v(-self._ai_vy),       # flip vy direction to match flipped y
            _norm_x(self._opp_x),
            _norm_y(opp_y_flipped),
            _norm_x(self._last_ball_x),
            _norm_y(ball_y_flipped),
        ], dtype=np.float32)

    def _build_frame_opp(self) -> np.ndarray:
        """
        One 8-dim observation frame from team 1 (player) perspective.

        Team 1's baseline is at y=0, so raw y values are already in the correct
        orientation: player at y≈1.5 (near baseline) and AI at y≈22.27 (far).
        No y-flip is needed — the same normalisation as team 0 produces symmetric
        feature values:  self_y ≈ -0.87 (near own baseline), opp_y ≈ +0.87 (far).
        """
        return np.array([
            _norm_x(self._opp_x),
            _norm_y(self._opp_y),          # raw: player baseline at y=0
            _norm_v(self._opp_vx),
            _norm_v(self._opp_vy),
            _norm_x(self._ai_x),
            _norm_y(self._ai_y),           # raw: AI is far at y≈22.27 → norm ≈ +0.87
            _norm_x(self._last_ball_x),
            _norm_y(self._last_ball_y),
        ], dtype=np.float32)

    def _build_obs_1(self) -> np.ndarray:
        """
        Build team 1's 24-dim observation using a proper 3-frame rolling history.
        Appends the current frame to the buffer then stacks.
        """
        self._opp_frame_buf.append(self._build_frame_opp())
        return np.concatenate(list(self._opp_frame_buf), axis=0)

    def _get_obs(self) -> np.ndarray:
        """Stack the 3 frames in the buffer into a 24-dim vector."""
        return np.concatenate(list(self._frame_buf), axis=0)   # (24,)

    def _terminal(
        self, reward: float, reason: str
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.episodes_completed += 1
        # Push the final frame before returning
        self._frame_buf.append(self._build_frame_ai())
        return self._get_obs(), reward, True, False, {"outcome": reason}

    def close(self) -> None:
        pass   # nothing to clean up

    def __enter__(self) -> "TennisEnv":
        return self

    def __exit__(self, *_) -> None:
        self.close()
