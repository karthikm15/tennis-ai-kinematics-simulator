import { COURT } from './court';
import { ValidationReason } from '../types';

const W   = COURT.widthM;    // 8.23 m
const NET = COURT.netYM;     // 11.885 m
const L   = COURT.lengthM;   // 23.77 m
const AI_HALF = L - NET;     // 11.885 m

// ── Shot record ───────────────────────────────────────────────────────────────

export interface ShotRecord {
  // Where the incoming AI ball landed (= where the player was standing to hit)
  ballLandX: number;
  ballLandY: number;
  // Where the player aimed their return
  targetX: number;
  targetY: number;
  // Outcome
  valid: boolean;
  reason?: ValidationReason;
  deflectionAngle?: number; // radians
  returnSpeed?: number;     // m/s
}

// ── Zone helpers ──────────────────────────────────────────────────────────────
// Divide the court width into 3 equal columns: 0=Left, 1=Center, 2=Right

export function xZone(x: number): 0 | 1 | 2 {
  if (x < W / 3)         return 0;
  if (x < (2 * W) / 3)  return 1;
  return 2;
}

// Divide AI half into 3 depth rows: 0=Short (net side), 1=Mid, 2=Deep (baseline)
export function yDepthZone(y: number): 0 | 1 | 2 {
  const d = y - NET;
  if (d < AI_HALF / 3)         return 0;
  if (d < (2 * AI_HALF) / 3)  return 1;
  return 2;
}

// ── Profile ───────────────────────────────────────────────────────────────────

export interface DirectionalRow {
  label: string;      // "Ball left", "Ball center", "Ball right"
  crossCount: number;
  sameCount:  number;
  total:      number;
}

export interface PlayerProfile {
  totalAttempts:  number;
  validCount:     number;
  faultRate:      number;   // 0–1
  faults: Record<ValidationReason, number>;
  // 3×3 [xCol][yRow] counts of valid targets in AI half
  heatmap: [[number,number,number],[number,number,number],[number,number,number]];
  // Conditioned on which side the ball arrived from
  directional: DirectionalRow[];
  avgDepth:       number;   // 0=short, 1=deep
  avgReturnSpeed: number;   // m/s
  // Last ≤10 valid shot speeds for sparkline
  recentSpeeds:   number[];
}

// ── Compute ───────────────────────────────────────────────────────────────────

export function computeProfile(history: ShotRecord[]): PlayerProfile {
  const heatmap: [[number,number,number],[number,number,number],[number,number,number]] =
    [[0,0,0],[0,0,0],[0,0,0]];

  const faults: Record<ValidationReason, number> = {
    out_of_bounds:    0,
    net_fault:        0,
    impossible_angle: 0,
    unreachable:      0,
  };

  // dirCounts[ballXZone][0=sameDir, 1=crossCourt]
  const dirCounts: [[number,number],[number,number],[number,number]] = [[0,0],[0,0],[0,0]];

  let totalDepth  = 0;
  let totalSpeed  = 0;
  let validCount  = 0;
  const recentSpeeds: number[] = [];

  for (const s of history) {
    if (!s.valid) {
      if (s.reason) faults[s.reason]++;
      continue;
    }

    // Heatmap
    const col = xZone(s.targetX);
    const row = yDepthZone(s.targetY);
    heatmap[col][row]++;

    // Directional tendency
    // Cross-court: ball left → target right (or center), ball right → target left (or center)
    const ballCol   = xZone(s.ballLandX);
    const targetCol = xZone(s.targetX);
    const isCross =
      (ballCol === 0 && targetCol >= 1) ||   // ball left  → aimed center or right
      (ballCol === 2 && targetCol <= 1);      // ball right → aimed center or left
    // ball center: call it cross if going to either outer column
    const isCrossFromCenter = ballCol === 1 && (targetCol === 0 || targetCol === 2);
    dirCounts[ballCol][isCross || isCrossFromCenter ? 1 : 0]++;

    // Depth & speed
    const depth = Math.max(0, Math.min(1, (s.targetY - NET) / AI_HALF));
    totalDepth += depth;
    if (s.returnSpeed) {
      totalSpeed += s.returnSpeed;
      recentSpeeds.push(s.returnSpeed);
    }
    validCount++;
  }

  const directional: DirectionalRow[] = [
    { label: 'Ball left',   crossCount: dirCounts[0][1], sameCount: dirCounts[0][0], total: dirCounts[0][0]+dirCounts[0][1] },
    { label: 'Ball center', crossCount: dirCounts[1][1], sameCount: dirCounts[1][0], total: dirCounts[1][0]+dirCounts[1][1] },
    { label: 'Ball right',  crossCount: dirCounts[2][1], sameCount: dirCounts[2][0], total: dirCounts[2][0]+dirCounts[2][1] },
  ];

  return {
    totalAttempts:  history.length,
    validCount,
    faultRate:      history.length > 0 ? (history.length - validCount) / history.length : 0,
    faults,
    heatmap,
    directional,
    avgDepth:       validCount > 0 ? totalDepth  / validCount : 0.5,
    avgReturnSpeed: validCount > 0 ? totalSpeed  / validCount : 0,
    recentSpeeds:   recentSpeeds.slice(-10),
  };
}
