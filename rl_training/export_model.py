"""
Export a trained ActorCritic checkpoint for use in the TypeScript browser UI.

Two export formats:

1. ONNX  (export_onnx)
   Standard interchange format. Can be loaded with onnxruntime-node or
   onnxruntime-web. Requires `pip install onnx`.

2. JSON weights  (export_json_weights)
   Lightweight: exports raw weight matrices as nested arrays.
   The TypeScript file `src/engine/rl_agent.ts` (created here) contains
   a self-contained 50-line forward pass that loads this JSON and runs
   inference entirely in the browser — zero extra dependencies.

After training:
    python3 export_model.py --checkpoint checkpoints/ckpt_final.pt

Then in useGameState.ts, replace:
    const nextShot = generateAiShot(newAiPos, state.playerPos);
with:
    const nextShot = await rlAgentShot(newAiPos, state);
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import torch

from config import TrainingConfig
from networks import ActorCritic
from tennis_env import (
    COURT_WIDTH, NET_Y, MARGIN, NET_MARGIN, action_to_court_coords
)


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(model: ActorCritic, out_path: str, obs_dim: int = 24) -> None:
    """Export the actor mean head to ONNX (deterministic / greedy policy)."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        print("[export] 'onnx' not installed — skipping ONNX export. pip install onnx")
        return

    model.eval()
    dummy = torch.zeros(1, obs_dim)

    # Wrap forward to output only the deterministic action
    class ActorOnly(torch.nn.Module):
        def __init__(self, policy: ActorCritic) -> None:
            super().__init__()
            self.policy = policy

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.policy.act_deterministic(obs)   # (1, act_dim)

    actor_only = ActorOnly(model)

    torch.onnx.export(
        actor_only,
        dummy,
        out_path,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=14,
    )
    print(f"[export] ONNX saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# JSON weights export + TypeScript agent file
# ─────────────────────────────────────────────────────────────────────────────

def export_json_weights(
    model:    ActorCritic,
    json_out: str,
    ts_out:   str,
) -> None:
    """
    Dump network weights to JSON and write a self-contained TypeScript agent.

    The TypeScript forward pass is a plain nested-array matrix multiply +
    LayerNorm + Tanh — no ML framework needed in the browser.
    """
    model.eval()

    # ── Collect weights ───────────────────────────────────────────────────────
    layers_data = []

    # Encoder layers (Linear → LayerNorm → Tanh blocks)
    # Iterate through net children: Linear, LayerNorm, Tanh, Linear, LayerNorm, Tanh
    enc = model.encoder.net
    i = 0
    layer_idx = 0
    children = list(enc.children())
    while i < len(children):
        if isinstance(children[i], torch.nn.Linear):
            lin = children[i]
            ln  = children[i + 1] if i + 1 < len(children) and isinstance(children[i + 1], torch.nn.LayerNorm) else None

            layer_info: dict = {
                "type":   "linear_layernorm_tanh",
                "W":      lin.weight.detach().cpu().tolist(),  # (out, in)
                "b":      lin.bias.detach().cpu().tolist(),    # (out,)
            }
            if ln is not None:
                layer_info["ln_weight"] = ln.weight.detach().cpu().tolist()
                layer_info["ln_bias"]   = ln.bias.detach().cpu().tolist()
                layer_info["ln_eps"]    = ln.eps
                i += 1  # skip the LayerNorm
            i += 1      # skip the Tanh (if present)

            layers_data.append(layer_info)
            layer_idx += 1
        i += 1

    # Actor head (mean layer only — deterministic policy)
    actor_lin = model.actor.mean_layer
    layers_data.append({
        "type": "linear_tanh",
        "W":    actor_lin.weight.detach().cpu().tolist(),
        "b":    actor_lin.bias.detach().cpu().tolist(),
    })

    weights: dict = {
        "obs_dim":    24,
        "act_dim":    2,
        "actor_log_std": model.actor.log_std.detach().cpu().clamp(
            model.actor.LOG_STD_MIN,
            model.actor.LOG_STD_MAX,
        ).tolist(),
        "court": {
            "width":      COURT_WIDTH,
            "net_y":      NET_Y,
            "margin":     MARGIN,
            "net_margin": NET_MARGIN,
        },
        "layers": layers_data,
    }

    with open(json_out, "w") as fh:
        json.dump(weights, fh)
    print(f"[export] JSON weights saved → {json_out}")

    # ── Write TypeScript agent ────────────────────────────────────────────────
    ts_code = _typescript_agent(json_out)
    with open(ts_out, "w") as fh:
        fh.write(ts_code)
    print(f"[export] TypeScript agent saved → {ts_out}")


def _typescript_agent(json_weights_path: str) -> str:
    """
    Generate the TypeScript rl_agent.ts file.

    Drop this into src/engine/ and call rlAgentShot() in useGameState.ts
    wherever generateAiShot() is currently called.
    """
    # Use just the filename for the import path (assume it ends up in public/)
    weights_filename = os.path.basename(json_weights_path)

    return f'''\
/**
 * rl_agent.ts — Browser-side inference for the trained PPO tennis agent.
 *
 * Auto-generated by export_model.py. Do not edit manually.
 *
 * Usage in useGameState.ts:
 *   import {{ initRLAgent, rlAgentShot }} from './rl_agent';
 *
 *   // Call once on app init:
 *   await initRLAgent();
 *
 *   // Replace generateAiShot(...):
 *   const nextShot = rlAgentShot(newAiPos, state.playerPos, gameState);
 */

import {{ Vec2, Shot }} from '../types';
import {{ COURT }} from './court';
import {{ computeVz }} from './kinematics';

// ── Types ─────────────────────────────────────────────────────────────────────

interface LayerWeights {{
  type: string;
  W: number[][];
  b: number[];
  ln_weight?: number[];
  ln_bias?: number[];
  ln_eps?: number;
}}

interface ModelWeights {{
  obs_dim: number;
  act_dim: number;
  court: {{ width: number; net_y: number; margin: number; net_margin: number }};
  layers: LayerWeights[];
  actor_log_std?: number[];
}}

// ── Module state ──────────────────────────────────────────────────────────────

let _weights: ModelWeights | null = null;
const N_FRAMES = 3;
const N_FEATURES = 8;
const STOCHASTIC_POLICY_TEMPERATURE = 0.75;
const STOCHASTIC_MEAN_CLAMP = 0.95;
// Ring buffer for frame stacking
const _frameBuffer: number[][] = [];

// ── Init ──────────────────────────────────────────────────────────────────────

export async function initRLAgent(weightsUrl = '/{weights_filename}'): Promise<void> {{
  const res = await fetch(weightsUrl);
  _weights = (await res.json()) as ModelWeights;
  _frameBuffer.length = 0;
  console.log('[RL Agent] Weights loaded.');
}}

// ── Forward pass ──────────────────────────────────────────────────────────────

function matVec(W: number[][], x: number[]): number[] {{
  return W.map(row => row.reduce((s, w, j) => s + w * x[j], 0));
}}

function addBias(x: number[], b: number[]): number[] {{
  return x.map((v, i) => v + b[i]);
}}

function tanh(x: number[]): number[] {{
  return x.map(v => Math.tanh(v));
}}

function layerNorm(x: number[], w: number[], b: number[], eps: number): number[] {{
  const mean = x.reduce((s, v) => s + v, 0) / x.length;
  const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
  const std = Math.sqrt(variance + eps);
  return x.map((v, i) => w[i] * ((v - mean) / std) + b[i]);
}}

function forward(obs: number[]): number[] {{
  if (!_weights) throw new Error('RL Agent not initialised — call initRLAgent() first.');
  let h = obs;
  for (const layer of _weights.layers) {{
    h = addBias(matVec(layer.W, h), layer.b);
    if (layer.ln_weight && layer.ln_bias && layer.ln_eps !== undefined) {{
      h = layerNorm(h, layer.ln_weight, layer.ln_bias, layer.ln_eps);
    }}
    if (layer.type === 'linear_layernorm_tanh' || layer.type === 'linear_tanh') {{
      h = tanh(h);
    }}
  }}
  return h; // [action_0, action_1] ∈ [-1, 1]
}}

function randn(): number {{
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}}

function sampleTanhNormal(meanAction: number[]): number[] {{
  if (!_weights?.actor_log_std) return meanAction;
  return meanAction.map((mean, i) => {{
    const clippedMean = Math.max(-STOCHASTIC_MEAN_CLAMP, Math.min(STOCHASTIC_MEAN_CLAMP, mean));
    const preTanhMean = 0.5 * Math.log((1 + clippedMean) / (1 - clippedMean));
    const std = Math.exp(_weights!.actor_log_std![i] ?? -2) * STOCHASTIC_POLICY_TEMPERATURE;
    return Math.tanh(preTanhMean + randn() * std);
  }});
}}

// ── Observation builder ───────────────────────────────────────────────────────

function normX(x: number): number {{ return x / (COURT.widthM / 2) - 1; }}
function normY(y: number): number {{ return y / (COURT.lengthM / 2) - 1; }}
function normV(v: number): number {{ return Math.max(-1, Math.min(1, v / 6)); }}

function buildFrame(
  aiPos: Vec2, aiVel: Vec2,
  playerPos: Vec2,
  lastBall: Vec2,
): number[] {{
  // y-flip: AI sees court from its own perspective (baseline = y=0 in obs)
  const aiYFlip      = COURT.lengthM - aiPos.y;
  const oppYFlip     = COURT.lengthM - playerPos.y;
  const ballYFlip    = COURT.lengthM - lastBall.y;
  return [
    normX(aiPos.x),    normY(aiYFlip),
    normV(aiVel.x),    normV(-aiVel.y),
    normX(playerPos.x), normY(oppYFlip),
    normX(lastBall.x), normY(ballYFlip),
  ];
}}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Call this once per new point (when the point resets) to clear frame history.
 */
export function resetRLFrameBuffer(): void {{
  _frameBuffer.length = 0;
}}

/**
 * Replaces generateAiShot(). Returns a Shot for the AI to play.
 *
 * Parameters match what's available in the PLAYER_SHOT_LANDED branch of
 * the reducer in useGameState.ts.
 */
export function rlAgentShot(
  aiPos:     Vec2,
  playerPos: Vec2,
  lastBall:  Vec2,
  aiVel:     Vec2 = {{ x: 0, y: 0 }},
): Shot {{
  if (!_weights) {{
    throw new Error('RL Agent not initialised — call initRLAgent() first.');
  }}

  const frame = buildFrame(aiPos, aiVel, playerPos, lastBall);

  // Maintain rolling frame buffer (most-recent 3 frames)
  _frameBuffer.push(frame);
  if (_frameBuffer.length > N_FRAMES) _frameBuffer.shift();

  // Zero-pad if we don't have 3 frames yet (start of episode)
  const padded: number[][] = [];
  for (let i = 0; i < N_FRAMES; i++) {{
    padded.push(_frameBuffer[_frameBuffer.length - N_FRAMES + i] ?? new Array(N_FEATURES).fill(0));
  }}
  const obs = padded.flat();   // length 24

  const meanAction = forward(obs);   // [ax, ay] ∈ [-1, 1]
  const action = sampleTanhNormal(meanAction);

  // Map action → court landing coordinate (AI shoots toward player's half)
  // y-range matches training: [margin, net_y - net_margin] (NET_MARGIN buffer from net)
  const {{ margin, net_y, width, net_margin }} = _weights.court;
  const landX = margin + ((action[0] + 1) / 2) * (width - 2 * margin);
  const landY = margin + ((action[1] + 1) / 2) * (net_y - net_margin - margin);

  const landing: Vec2 = {{
    x: Math.max(0, Math.min(width,  landX)),
    y: Math.max(0, Math.min(net_y - 0.01, landY)),
  }};

  const dx = landing.x - aiPos.x;
  const dy = landing.y - aiPos.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const speed = 10.0;
  const travelTime = Math.max(dist / speed, 0.8);

  return {{
    origin:     aiPos,
    landing,
    speed,
    spinType:   'flat',
    travelTime,
    vz:         computeVz(travelTime),
  }};
}}
'''


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained PPO agent for browser use.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--out-dir", default="../public", help="Output directory for weights JSON")
    parser.add_argument("--ts-out", default="../src/engine/rl_agent.ts", help="TypeScript output path")
    parser.add_argument("--onnx", action="store_true", help="Also export ONNX file")
    args = parser.parse_args()

    cfg    = TrainingConfig()
    policy = ActorCritic(cfg.obs_dim, cfg.act_dim, cfg.hidden_dims)
    state  = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(state["model_state"])
    print(f"[export] Loaded checkpoint: {args.checkpoint}  (Elo={state.get('elo', '?'):.1f})")

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "rl_weights.json")

    export_json_weights(policy, json_path, args.ts_out)

    if args.onnx:
        onnx_path = os.path.join(args.out_dir, "rl_agent.onnx")
        export_onnx(policy, onnx_path, cfg.obs_dim)


if __name__ == "__main__":
    main()
