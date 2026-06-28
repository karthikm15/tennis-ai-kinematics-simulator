import { Vec2, Shot, ShotValidation } from '../types';
import { COURT, isInBounds } from './court';

// ── Vector helpers ────────────────────────────────────────────────────────────

function sub(a: Vec2, b: Vec2): Vec2 { return { x: a.x - b.x, y: a.y - b.y }; }
function mag(v: Vec2): number { return Math.sqrt(v.x * v.x + v.y * v.y); }
function norm(v: Vec2): Vec2 {
  const m = mag(v);
  return m < 1e-9 ? { x: 0, y: 1 } : { x: v.x / m, y: v.y / m };
}
function dot(a: Vec2, b: Vec2): number { return a.x * b.x + a.y * b.y; }

// ── Physics constants ─────────────────────────────────────────────────────────

export const PLAYER_MAX_SPEED_MS = 6.0;  // human player sprint speed (m/s)
export const AI_MAX_SPEED_MS     = 5.5;  // AI opponent sprint speed (m/s)

const MIN_DEFLECTION_ANGLE_RAD = (30 * Math.PI) / 180;
const BASE_SWING_POWER = 8.0;
const TRANSFER_COEFFICIENT = 0.6;
export const GRAVITY = 9.81;
export const HIT_HEIGHT = 1.0; // m — racket contact height above court

// ── Arc helpers (exported for visualization) ──────────────────────────────────

/**
 * Vertical launch velocity (m/s) for a shot that starts at HIT_HEIGHT and
 * lands on the ground (z=0) after travelTime seconds.
 * Derived from: HIT_HEIGHT + Vz*T − 0.5*g*T² = 0
 */
export function computeVz(travelTime: number): number {
  return 0.5 * GRAVITY * travelTime - HIT_HEIGHT / travelTime;
}

/**
 * Ball height (m) at elapsed time t seconds into a shot with the given vz.
 * Clamped to zero so the ball never clips through the court.
 */
export function ballHeightAt(t: number, vz: number): number {
  return Math.max(0, HIT_HEIGHT + vz * t - 0.5 * GRAVITY * t * t);
}

// ── Individual validation checks ──────────────────────────────────────────────

export function isReachable(ballLanding: Vec2, playerPos: Vec2, ballTravelTime: number): boolean {
  const dist = mag(sub(ballLanding, playerPos));
  return dist / PLAYER_MAX_SPEED_MS <= ballTravelTime;
}

export function canReach(from: Vec2, to: Vec2, maxSpeed: number, travelTime: number): boolean {
  return mag(sub(to, from)) <= maxSpeed * travelTime;
}

export function reachablePos(from: Vec2, to: Vec2, maxSpeed: number, travelTime: number): Vec2 {
  const d = sub(to, from);
  const dist = mag(d);
  if (dist < 0.001) return from;
  const maxDist = maxSpeed * travelTime;
  if (dist <= maxDist) return to;
  const frac = maxDist / dist;
  return { x: from.x + d.x * frac, y: from.y + d.y * frac };
}

export function computeDeflectionAngle(incomingDir: Vec2, outgoingDir: Vec2): number {
  const d = Math.max(-1, Math.min(1, dot(norm(incomingDir), norm(outgoingDir))));
  return Math.acos(d);
}

export function computeReturnSpeed(incomingSpeed: number, deflectionAngle: number): number {
  const cosFactor = Math.sin(deflectionAngle); // peaks at 90°
  return Math.max(BASE_SWING_POWER, BASE_SWING_POWER + incomingSpeed * TRANSFER_COEFFICIENT * cosFactor);
}

export function checkNetClearance(hitPoint: Vec2, targetPoint: Vec2, speed: number): boolean {
  const NET_Y = COURT.netYM;
  const totalHorizDist = mag(sub(targetPoint, hitPoint));
  if (totalHorizDist < 0.01) return false;

  const minY = Math.min(hitPoint.y, targetPoint.y);
  const maxY = Math.max(hitPoint.y, targetPoint.y);
  if (NET_Y < minY || NET_Y > maxY) return true;

  const tLand  = totalHorizDist / speed;
  const vz     = computeVz(tLand);
  const distToNet = Math.abs(NET_Y - hitPoint.y);
  const tNet   = (distToNet / Math.abs(targetPoint.y - hitPoint.y)) * tLand;

  return ballHeightAt(tNet, vz) >= COURT.netHeightCenterM;
}

// ── Master validation ─────────────────────────────────────────────────────────

export function validateReturn(
  incoming: Shot,
  playerPos: Vec2,
  target: Vec2
): ShotValidation {
  if (!isInBounds(target, 'ai_half')) {
    return { valid: false, reason: 'out_of_bounds' };
  }

  if (!isReachable(incoming.landing, playerPos, incoming.travelTime)) {
    return { valid: false, reason: 'unreachable' };
  }

  const incomingDir = sub(incoming.landing, incoming.origin);
  const outgoingDir = sub(target, incoming.landing);
  const deflAngle   = computeDeflectionAngle(incomingDir, outgoingDir);

  if (deflAngle < MIN_DEFLECTION_ANGLE_RAD) {
    return { valid: false, reason: 'impossible_angle', deflectionAngle: deflAngle };
  }

  const returnSpeed = computeReturnSpeed(incoming.speed, deflAngle);

  if (!checkNetClearance(incoming.landing, target, returnSpeed)) {
    return { valid: false, reason: 'net_fault', deflectionAngle: deflAngle, returnSpeed };
  }

  const dist     = mag(sub(target, incoming.landing));
  const travelTime = Math.max(dist / returnSpeed, 0.5);

  const returnShot: Shot = {
    origin:     incoming.landing,
    landing:    target,
    speed:      returnSpeed,
    spinType:   'flat',
    travelTime,
    vz:         computeVz(travelTime),
  };

  return { valid: true, deflectionAngle: deflAngle, returnSpeed, returnShot };
}
