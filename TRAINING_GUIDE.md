# Tennis AI — Reinforcement Learning Training Guide

This document explains every decision made in the RL pipeline, what went wrong during setup, why we pivoted to a pure Python environment, and exactly how the trained agent plugs back into the browser UI.

---

## Table of Contents

1. [What We're Building](#1-what-were-building)
2. [Architecture Overview](#2-architecture-overview)
3. [The Environment](#3-the-environment)
4. [The Neural Network](#4-the-neural-network)
5. [The Training Algorithm (PPO + GAE)](#5-the-training-algorithm-ppo--gae)
6. [Adversarial Self-Play](#6-adversarial-self-play)
7. [Elo Rating System](#7-elo-rating-system)
8. [Why We Dropped Unity](#8-why-we-dropped-unity)
9. [Running Training](#9-running-training)
10. [Integrating into the Browser UI](#10-integrating-into-the-browser-ui)
11. [File Reference](#11-file-reference)

---

## 1. What We're Building

The existing UI has an AI bot (`src/engine/ai.ts`) that picks shot landing coordinates using a simple heuristic: aim to the side the player is **not** on, 70% of the time. We want to replace this with a policy network trained through **competitive self-play** — an AI that learns by repeatedly playing against earlier versions of itself and improving until it's genuinely hard to beat.

The trained model:
- Takes the current game state as input (positions + recent history)
- Outputs a **landing coordinate** `{ x, y }` on the player's half of the court
- This drops directly into the existing `Shot` object in `useGameState.ts`

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Python Training Pipeline  (rl_training/)                           │
│                                                                     │
│  TennisEnv  ──obs──►  ActorCritic  ──action──►  TennisEnv          │
│   (pure                 (PPO policy)             (simulates         │
│   Python)                                         opponent too)     │
│                              │                                      │
│                         RolloutBuffer                               │
│                              │                                      │
│                         PPOOptimizer                                │
│                              │                                      │
│                       SelfPlayManager  ──Elo──►  checkpoints/       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                               │
                         export_model.py
                               │
              ┌────────────────┴────────────────┐
              │                                 │
        public/rl_weights.json        src/engine/rl_agent.ts
              │                                 │
              └────────────────┬────────────────┘
                               │
                    Browser (useGameState.ts)
                    replaces generateAiShot()
```

---

## 3. The Environment

**File:** `rl_training/tennis_env.py`

### Court Coordinate System

The Python simulation uses the **exact same coordinate system** as `src/engine/court.ts`:

```
x ∈ [0, 8.23]   — court width (metres)
y ∈ [0, 23.77]  — court length (metres)

Player half:  y ∈ [0,      11.885]
Net:          y = 11.885
AI half:      y ∈ [11.885, 23.77]

Player baseline: y = 0
AI baseline:     y = 23.77
```

This means the observation and action values coming out of the trained model use the same units and reference frame as the TypeScript code, so no coordinate remapping is needed during integration.

### Turn-Based Episodes

Unlike continuous physics simulators, this environment is **turn-based**: each `step()` represents one complete shot exchange.

```
reset()         → AI at baseline, player at baseline, frame buffer initialised

step(action_0)
    1. AI hits ball using action_0 → landing coordinate on player's half
    2. Check: does player reach the ball in time?
       NO  → reward = +1, terminated = True  (AI wins point)
    3. Frozen policy chooses a return landing (on AI's half)
    4. Check: does AI reach the return in time?
       NO  → reward = -1, terminated = True  (player wins point)
    5. Both reach → reward = 0, terminated = False, rally continues
```

This keeps the training loop clean: the PPO training code sees a standard single-agent interface, and all of the two-player coordination happens inside `env.step()`.

### Observation Space (24-dim)

Three frames are stacked to give the agent temporal context — it can infer ball trajectory and opponent movement patterns from the history.

Each frame contains 8 features (all normalized to `[-1, 1]`):

| Index | Feature | Formula |
|---|---|---|
| 0 | AI x-position | `x / 4.115 - 1` |
| 1 | AI y-position (flipped) | `(23.77 - y) / 11.885 - 1` |
| 2 | AI x-velocity | `vx / 5.5` |
| 3 | AI y-velocity (flipped) | `-vy / 5.5` |
| 4 | Opponent x-position | `x / 4.115 - 1` |
| 5 | Opponent y-position (flipped) | `(23.77 - y) / 11.885 - 1` |
| 6 | Last ball x | `x / 4.115 - 1` |
| 7 | Last ball y (flipped) | `(23.77 - y) / 11.885 - 1` |

**Why y-flip?** The AI lives on the upper half (`y > 11.885`). Flipping the y-axis makes the AI's baseline appear at `y=0` in observation space, so the observation is symmetric — a single policy trained on one side generalises immediately to the other side during self-play without any extra logic.

### Action Space (2-dim)

The output of the policy network after tanh squashing:

```
action[0] ∈ [-1, 1]  →  landing_x ∈ [0.3, 7.93]  (player's court x)
action[1] ∈ [-1, 1]  →  landing_y ∈ [0.3, 11.58]  (player's court y < net)
```

Mapping (for AI shooting toward player's half):
```python
landing_x = 0.3 + (action[0] + 1) / 2 * (8.23 - 0.6)
landing_y = 0.3 + (action[1] + 1) / 2 * (11.585 - 0.3)
```

### Physics (matching `kinematics.ts` exactly)

```python
# canReach() — identical to TypeScript
def can_reach(from_pos, to_pos, max_speed, travel_time):
    return distance(from_pos, to_pos) <= max_speed * travel_time

# AI max speed:     5.5 m/s  (AI_MAX_SPEED_MS in kinematics.ts)
# Player max speed: 6.0 m/s  (PLAYER_MAX_SPEED_MS in kinematics.ts)
# Shot speed:       10.0 m/s (mid-range of the 8–12 m/s range in ai.ts)
# Min travel time:  0.8s
```

The net clearance check uses the same parabolic arc formula from `checkNetClearance()` in `kinematics.ts` — `HIT_HEIGHT = 1.0m`, `GRAVITY = 9.81 m/s²`.

### Reward Structure

| Event | Reward |
|---|---|
| Ball lands unreachable for player (winner) | **+1.0** |
| AI hits out of bounds or into net | **-1.0** |
| Player's return is unreachable for AI | **-1.0** |
| Player's return goes out or into net | **+1.0** |
| Rally continues (both reach the ball) | **0.0** |

This is a strict **zero-sum competitive** reward — exactly the opposite of the default Unity Tennis cooperative reward (`+0.1` for net clearance).

---

## 4. The Neural Network

**File:** `rl_training/networks.py`

### Architecture

```
Input (24,)
    │
    ▼
SharedEncoder
    Linear(24 → 256) + LayerNorm + Tanh
    Linear(256 → 256) + LayerNorm + Tanh
    │
    ├──► ActorHead
    │        Linear(256 → 2)  →  mean (μ)
    │        log_std param    →  log σ  (state-independent)
    │
    └──► CriticHead
             Linear(256 → 1)  →  V(s)
```

### TanhNormal Policy

The policy is a **TanhNormal distribution**: samples `x ~ N(μ, σ)` then squashes via `tanh` to get actions in `[-1, 1]`. This is **not** just clipping — the log-probability must be corrected for the change of variables:

```
log π(a|s) = log p(x|s) − Σᵢ 2·[log2 − xᵢ − softplus(−2xᵢ)]
```

The numerically stable `softplus` form avoids `log(0)` when `|x|` is large.

**Why store `x` (pre-tanh) in the buffer, not `tanh(x)`?**  
During the PPO update, we re-evaluate stored actions under the *current* policy. We need `x` to compute `log π_new(tanh(x)|s)`. Storing `tanh(x)` would require `atanh` which is undefined at ±1.

### Initialization

- Hidden layers: orthogonal init, gain `√2` (standard RL practice)
- Actor output: gain `0.01` — policy starts near-deterministic, entropy anneals it open
- Critic output: gain `1.0`

---

## 5. The Training Algorithm (PPO + GAE)

**Files:** `rl_training/ppo.py`, `rl_training/buffer.py`

### Why PPO?

PPO (Proximal Policy Optimization) was chosen over alternatives because:

- **DDPG/SAC** — Off-policy methods struggle with non-stationarity from self-play (the opponent changes between updates)
- **Vanilla policy gradient** — High variance; can collapse when switching from cooperative to competitive reward
- **PPO** — On-policy, naturally handles the changing opponent distribution since it only uses fresh rollouts; the clip ratio prevents catastrophic policy updates when reward changes sign

### Generalized Advantage Estimation (GAE)

After collecting `rollout_steps = 2048` transitions, the buffer computes advantages backwards through time:

```
δₜ = rₜ + γ·V(sₜ₊₁)·(1 − doneₜ) − V(sₜ)    ← TD residual
Âₜ = δₜ + (γλ)·(1 − doneₜ)·Âₜ₊₁             ← GAE recursion
Gₜ = Âₜ + V(sₜ)                                ← λ-returns
```

Key parameters:
- `γ = 0.99` — discount factor (points won 100 steps later still matter)
- `λ = 0.95` — bias-variance tradeoff (closer to 1 = lower bias, more variance)

### PPO Update

For each mini-batch from the buffer:

```
ratio = exp(log π_new(a|s) − log π_old(a|s))      ← importance sampling

L_CLIP = −min(ratio·Â,  clip(ratio, 1±ε)·Â)       ← clipped surrogate
L_VF   =  max(MSE(V, G), MSE(V_clipped, G))        ← clipped value loss
L_H    = −entropy[π]                                ← entropy bonus

L_total = L_CLIP + 0.5·L_VF − 0.01·L_H
```

**Why clip the value loss?** Without clipping, the critic can take arbitrarily large steps, destabilizing the advantage baseline. The clip mirrors the policy clip.

**Why normalize advantages before the PPO epoch loop, not per mini-batch?**  
Per-mini-batch normalization changes the gradient scale inconsistently across mini-batches within the same epoch, introducing noise. Buffer-level normalization is stable.

---

## 6. Adversarial Self-Play

**File:** `rl_training/self_play.py`

### The Problem with Naive Self-Play

If you train a policy against *its own current weights*, the opponent distribution changes every gradient step. PPO's on-policy assumption breaks because the "opponent" at collection time ≠ "opponent" at update time. This causes policy collapse in competitive settings.

### Solution: Frozen Checkpoint Pool

```
Active Policy (trains via PPO)
         │
         │ every 100 episodes, if win_rate > 55%:
         ▼
    freeze → save to CheckpointPool
         │
         │ sample new opponent:
         │   80% → most-recent checkpoint
         │   20% → random historical checkpoint
         ▼
Frozen Policy (weights fixed, drives env.opponent_fn)
```

The **80/20 split** is critical:
- **80% recent** ensures the agent primarily fights its current skill level, not a random ancestor
- **20% random** prevents the agent from finding strategies that beat *only* the last checkpoint. Without this, agents can get trapped in cycles (A beats B beats C beats A)

### Win-Rate Gate

The gate only counts wins and losses, not draws (truncated episodes). This prevents a high draw-rate game from falsely triggering the gate.

```
win_rate = wins / (wins + losses)   # over last 100 episodes
if win_rate > 0.55:
    freeze active policy
    sample new opponent
```

---

## 7. Elo Rating System

**File:** `rl_training/self_play.py` — `EloTracker` class

Every checkpoint and the active policy maintain an Elo rating. After each episode:

```
E_active = 1 / (1 + 10^((R_opponent − R_active) / 400))
R_active ← R_active + 32·(S − E_active)     where S = 1 (win), 0 (loss), 0.5 (draw)
```

This gives you a single number that tracks **absolute skill level** over millions of steps, regardless of how many opponent pool changes occur. You can plot `elo_XXXXXXXXXX.json` files over time to see the skill curve.

A new checkpoint inherits the active policy's current Elo at freeze time, so the rating scale is continuous.

---

## 8. Why We Dropped Unity

### Original Plan

The pipeline was originally built around `mlagents_envs` talking to a Unity `Tennis.app` binary. The plan:

1. Unity Tennis binary provides physics simulation and observations
2. Python intercepts the cooperative reward and replaces it with competitive ±1

### What Went Wrong (in order)

| Step | Problem |
|---|---|
| `mlagents-envs==0.30.0` | Doesn't exist on PyPI. Latest is 0.28.0 |
| System Python 3.11 | Conda env wasn't activating due to a `python3` alias from Python.org installation |
| `protobuf` conflict | `mlagents_envs` needs `protobuf<4.0` but system had 4.x |
| Old Tennis binary (2018) | Communicator Protocol 0.3 vs mlagents_envs Protocol 1.5 — incompatible |
| Tennis removed from ML-Agents | Tennis environment was removed from ML-Agents examples around release 16. No compatible pre-built binary exists for Python 3.9+ |

### Why Pure Python Is Better

1. **Zero dependencies on Unity, binary compatibility, or gRPC**
2. **Court physics exactly match the TypeScript UI** (same constants, same `canReach` formula)
3. **Fast** — no socket communication overhead; each step is microseconds vs milliseconds
4. **Debuggable** — the entire environment is readable Python, not a compiled binary
5. **Direct integration** — the action space (`landing_x`, `landing_y`) is exactly what `generateAiShot` returns

---

## 9. Running Training

### Setup

```bash
# Create conda env (must be Python 3.9)
conda create -n tennis-rl python=3.9.13 -c conda-forge -y
conda activate tennis-rl

# Resolve the python alias issue if which python3 shows wrong version:
alias python3="/Users/karthikmittal/opt/anaconda3/envs/tennis-rl/bin/python3"

# Install (numpy<2 required for torch 2.2.2 compatibility)
cd rl_training
python3 -m pip install torch>=2.1.0 "numpy>=1.24.0,<2.0.0"
```

No other packages needed — `mlagents_envs` is no longer a dependency.

### Start Training

```bash
cd rl_training
python3 train.py
```

Training runs for 10 million steps by default. On a MacBook CPU, expect ~500–1000 steps/second, so roughly 3–6 hours for a full run. You can stop early with Ctrl+C — the policy is saved to `checkpoints/` after every 100,000 steps.

### Monitor Progress

Each rollout prints a line like:
```
[   204,800] steps/s=847  win_rate=52.3%  elo=1223.4  opp_elo=1200.0  pool=1  π_loss=-0.0234  v_loss=0.1823  H=1.234  KL≈0.0041  clip=8.23%
```

| Field | Meaning |
|---|---|
| `win_rate` | Fraction of last 100 episodes won against current frozen opponent |
| `elo` | Active policy's Elo (starts at 1200, should rise over time) |
| `opp_elo` | Current frozen opponent's Elo |
| `pool` | Number of frozen checkpoints accumulated |
| `π_loss` | Policy gradient loss (should stay near 0, negative = improving) |
| `v_loss` | Critic MSE loss (should decrease over time) |
| `H` | Policy entropy (should stay > 0.5; if it drops to 0, policy has collapsed) |
| `KL≈` | Approximate KL divergence from old policy (should stay < 0.02) |
| `clip` | Fraction of ratios hitting the PPO clip bound (healthy: 5–20%) |

### Reducing Training Time

Edit `rl_training/config.py`:
```python
total_timesteps = 2_000_000   # 2M steps ≈ 30–60 min on CPU
eval_window = 50               # faster gating
win_rate_threshold = 0.60      # slightly stricter
```

---

## 10. Integrating into the Browser UI

### Export After Training

```bash
cd rl_training

# Export JSON weights + generate TypeScript agent file
python3 export_model.py \
    --checkpoint checkpoints/ckpt_final.pt \
    --out-dir ../public \
    --ts-out ../src/engine/rl_agent.ts
```

This creates:
- `public/rl_weights.json` — weight matrices (served statically by Vite)
- `src/engine/rl_agent.ts` — self-contained TypeScript inference engine

### Wire into useGameState.ts

**Step 1** — Add init call in `App.tsx`:

```typescript
import { initRLAgent } from './engine/rl_agent';

// Add this inside App():
useEffect(() => {
  initRLAgent('/rl_weights.json').catch(console.error);
}, []);
```

**Step 2** — In `useGameState.ts`, find the `PLAYER_SHOT_LANDED` case (line ~191):

```typescript
// BEFORE (random heuristic):
const nextShot = generateAiShot(newAiPos, state.playerPos);

// AFTER (RL agent):
import { rlAgentShot, resetRLFrameBuffer } from '../engine/rl_agent';

// Replace the generateAiShot call with:
const nextShot = rlAgentShot(newAiPos, state.playerPos, currentShot.landing);
```

**Step 3** — Reset the frame buffer at the start of each point (in `RESET_POINT` case):

```typescript
import { resetRLFrameBuffer } from '../engine/rl_agent';

// In the RESET_POINT reducer case, add:
resetRLFrameBuffer();
```

### What the RL Agent Returns

`rlAgentShot()` returns a standard `Shot` object — exactly the same type as `generateAiShot()`. The shape, speed, travel time, and arc are computed identically. Only the **landing coordinate** is different: it comes from the trained neural network instead of the heuristic.

---

## 11. File Reference

```
rl_training/
├── config.py         All hyperparameters in one dataclass
├── networks.py       ActorCritic (shared trunk + actor + critic heads)
│                       TanhNormal distribution with Jacobian correction
├── buffer.py         RolloutBuffer with GAE computation
├── tennis_env.py     Pure Python 2D tennis simulation
│                       Matches court.ts coordinates exactly
│                       action_to_court_coords() is the integration bridge
├── ppo.py            PPOOptimizer (clipped surrogate + clipped value loss)
├── self_play.py      EloTracker + CheckpointPool + SelfPlayManager
├── train.py          Main training loop
├── export_model.py   Exports weights → JSON + TypeScript rl_agent.ts
└── requirements.txt  torch, numpy<2
```
