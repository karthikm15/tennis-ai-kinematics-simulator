"""
diagnose.py — Inspect what a trained checkpoint is actually doing.

Usage (run while training is paused, or after it finishes):
    cd rl_training
    python3 diagnose.py                         # uses latest checkpoint
    python3 diagnose.py checkpoints/ckpt_N.pt  # specific checkpoint

Reports:
  • Episode length distribution  — are rallies happening?
  • Outcome breakdown            — what's causing wins/losses?
  • Landing position heatmap     — is the agent hitting the same spot every time?
  • Action clustering            — is it exploiting a degenerate corner?
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import torch

from config import TrainingConfig
from tennis_env import TennisEnv, action_to_court_coords, NET_Y, COURT_WIDTH, COURT_LENGTH
from networks import ActorCritic


# ── Load checkpoint ────────────────────────────────────────────────────────────

def load_latest(ckpt_dir: str) -> str:
    pts = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt files in {ckpt_dir}")
    return os.path.join(ckpt_dir, pts[-1])


cfg = TrainingConfig()
ckpt_path = sys.argv[1] if len(sys.argv) > 1 else load_latest(cfg.checkpoint_dir)
print(f"Loading: {ckpt_path}")

state  = torch.load(ckpt_path, map_location="cpu", weights_only=True)
policy = ActorCritic(cfg.obs_dim, cfg.act_dim, cfg.hidden_dims)
policy.load_state_dict(state["model_state"])
policy.eval()
print(f"Checkpoint Elo (at freeze): {state.get('elo', '?')}")
print()


# ── Run episodes ───────────────────────────────────────────────────────────────

N_EPISODES = 2000

env = TennisEnv()
outcomes:   dict = defaultdict(int)
ep_lengths: list = []
landing_ys: list = []
landing_xs: list = []
actions_x:  list = []
actions_y:  list = []

obs_np, _ = env.reset()
obs = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
ep_len = 0

for _ in range(N_EPISODES * 20):
    with torch.no_grad():
        action = policy.act_deterministic(obs)     # greedy (no noise)
    a_np = action.squeeze(0).cpu().numpy()
    actions_x.append(float(a_np[0]))
    actions_y.append(float(a_np[1]))

    coord = action_to_court_coords(a_np, shooting_from_ai_side=True)
    landing_xs.append(coord["x"])
    landing_ys.append(coord["y"])

    obs_np_next, reward, terminated, truncated, info = env.step(a_np)
    ep_len += 1

    if terminated or truncated:
        outcomes[info.get("outcome", "?")] += 1
        ep_lengths.append(ep_len)
        ep_len = 0
        obs_np_next, _ = env.reset()
        if len(ep_lengths) >= N_EPISODES:
            break

    obs = torch.tensor(obs_np_next, dtype=torch.float32).unsqueeze(0)


# ── Report ─────────────────────────────────────────────────────────────────────

total = sum(outcomes.values())
wins  = outcomes["winner"] + outcomes["opponent_out"] + outcomes["opponent_net"]
print(f"━━━ {total} episodes ━━━")
print(f"Win rate (vs frozen opponent / heuristic): {wins / total:.1%}")
print()

# Rally / episode length
print("Episode length:")
print(f"  mean = {np.mean(ep_lengths):.2f}   median = {int(np.median(ep_lengths))}   max = {max(ep_lengths)}")
print(f"  1 step  (instant winner or fault): {sum(l == 1 for l in ep_lengths) / total:.1%}")
print(f"  2 steps (one rally exchange):      {sum(l == 2 for l in ep_lengths) / total:.1%}")
print(f"  3-5 steps:                         {sum(3 <= l <= 5 for l in ep_lengths) / total:.1%}")
print(f"  6+ steps (sustained rally):        {sum(l >= 6 for l in ep_lengths) / total:.1%}")
print()

# Outcomes
print("Outcomes  (W=winner, N=net_fault, O=out_of_bounds, U=unreachable_return, oN/oO=opponent net/out):")
for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
    bar = "█" * (v * 40 // total)
    print(f"  {k:<28} {v:>5} ({v / total:.1%})  {bar}")
print()

# Landing position distribution
print(f"Landing Y  (0=player baseline  {NET_Y:.1f}=net — IDEAL: spread across mid-court):")
buckets = [
    ("near net  (>10.0 m)",          sum(y > 10.0 for y in landing_ys)),
    ("mid-court (6–10 m)",           sum(6.0 <= y <= 10.0 for y in landing_ys)),
    ("mid-deep  (3–6 m)",            sum(3.0 <= y < 6.0 for y in landing_ys)),
    ("deep      (<3 m)",             sum(y < 3.0 for y in landing_ys)),
]
for label, cnt in buckets:
    bar = "█" * (cnt * 40 // len(landing_ys))
    print(f"  {label}  {cnt:>5} ({cnt / len(landing_ys):.1%})  {bar}")
print(f"  mean y = {np.mean(landing_ys):.2f}  std = {np.std(landing_ys):.2f}")
print()

print(f"Landing X  (0=left  {COURT_WIDTH:.2f}=right — IDEAL: spread across court):")
buckets_x = [
    ("left sideline  (<1.5 m)",       sum(x < 1.5 for x in landing_xs)),
    ("left-center    (1.5–3.5 m)",    sum(1.5 <= x < 3.5 for x in landing_xs)),
    ("center         (3.5–4.75 m)",   sum(3.5 <= x < 4.75 for x in landing_xs)),
    ("right-center   (4.75–6.75 m)",  sum(4.75 <= x < 6.75 for x in landing_xs)),
    ("right sideline (>6.75 m)",      sum(x >= 6.75 for x in landing_xs)),
]
for label, cnt in buckets_x:
    bar = "█" * (cnt * 40 // len(landing_xs))
    print(f"  {label}  {cnt:>5} ({cnt / len(landing_xs):.1%})  {bar}")
print(f"  mean x = {np.mean(landing_xs):.2f}  std = {np.std(landing_xs):.2f}")
print()

# Action space clustering — key exploit indicator
print("Action space clustering (CONCERN: std near 0 = agent always hits same spot):")
print(f"  action[0] (→x):  mean={np.mean(actions_x):+.3f}  std={np.std(actions_x):.3f}")
print(f"  action[1] (→y):  mean={np.mean(actions_y):+.3f}  std={np.std(actions_y):.3f}")
if np.std(actions_x) < 0.15 or np.std(actions_y) < 0.15:
    print("  ⚠  Low action diversity — possible exploit (both std should be > 0.3)")
else:
    print("  ✓  Action diversity looks healthy")
