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
// With these values, from standstill the player reaches max speed in 0.5s
// and covers ~7.5m in 1.5s — enough for most groundstrokes but not extreme corners.
export const PLAYER_ACCEL_MS2    = 12.0;
export const AI_ACCEL_MS2        = 10.0;
const PLAYER_REACTION_TIME_S     = 0.25;
const AI_REACTION_TIME_S         = 0.18;

const MIN_DEFLECTION_ANGLE_RAD = (20 * Math.PI) / 180;
const BASE_SWING_POWER = 8.0;
const TRANSFER_COEFFICIENT = 0.6;
export const GRAVITY = 9.81;
export const HIT_HEIGHT       = 1.0; // m — groundstroke contact height
export const SERVE_HIT_HEIGHT = 2.4; // m — overhead serve contact height

// ── Arc helpers (exported for visualization) ──────────────────────────────────

export function computeVzFromHeight(startHeight: number, travelTime: number): number {
  return 0.5 * GRAVITY * travelTime - startHeight / travelTime;
}

export function computeVz(travelTime: number): number {
  return computeVzFromHeight(HIT_HEIGHT, travelTime);
}

export function ballHeightAt(t: number, vz: number, startHeight = HIT_HEIGHT): number {
  return Math.max(0, startHeight + vz * t - 0.5 * GRAVITY * t * t);
}

// ── Movement reach (game logic — fixed max speed) ─────────────────────────────

export function canReach(from: Vec2, to: Vec2, maxSpeed: number, travelTime: number): boolean {
  return mag(sub(to, from)) <= maxSpeed * travelTime;
}

export function reachablePos(from: Vec2, to: Vec2, maxSpeed: number, travelTime: number): Vec2 {
  const d    = sub(to, from);
  const dist = mag(d);
  if (dist < 0.001) return from;
  const maxDist = maxSpeed * travelTime;
  if (dist <= maxDist) return to;
  const frac = maxDist / dist;
  return { x: from.x + d.x * frac, y: from.y + d.y * frac };
}

// ── Visual-only acceleration helper (used by TennisCourt animation) ───────────

// Distance covered starting from rest after `time` seconds under acceleration.
export function reachableDistAccel(maxSpeed: number, accel: number, time: number): number {
  const tToMax = maxSpeed / accel;
  if (time <= tToMax) return 0.5 * accel * time * time;
  return 0.5 * accel * tToMax * tToMax + maxSpeed * (time - tToMax);
}

export function canReachWithAcceleration(
  from: Vec2,
  to: Vec2,
  maxSpeed: number,
  accel: number,
  travelTime: number,
  reactionTime = 0,
): boolean {
  const availableTime = Math.max(0, travelTime - reactionTime);
  return mag(sub(to, from)) <= reachableDistAccel(maxSpeed, accel, availableTime);
}

export function reachablePosWithAcceleration(
  from: Vec2,
  to: Vec2,
  maxSpeed: number,
  accel: number,
  travelTime: number,
  reactionTime = 0,
): Vec2 {
  const d = sub(to, from);
  const dist = mag(d);
  if (dist < 0.001) return from;
  const availableTime = Math.max(0, travelTime - reactionTime);
  const maxDist = reachableDistAccel(maxSpeed, accel, availableTime);
  if (dist <= maxDist) return to;
  const frac = maxDist / dist;
  return { x: from.x + d.x * frac, y: from.y + d.y * frac };
}

export function canPlayerReach(from: Vec2, to: Vec2, travelTime: number): boolean {
  return canReachWithAcceleration(
    from, to, PLAYER_MAX_SPEED_MS, PLAYER_ACCEL_MS2, travelTime, PLAYER_REACTION_TIME_S,
  );
}

export function reachablePlayerPos(from: Vec2, to: Vec2, travelTime: number): Vec2 {
  return reachablePosWithAcceleration(
    from, to, PLAYER_MAX_SPEED_MS, PLAYER_ACCEL_MS2, travelTime, PLAYER_REACTION_TIME_S,
  );
}

export function canAiReach(from: Vec2, to: Vec2, travelTime: number): boolean {
  return canReachWithAcceleration(
    from, to, AI_MAX_SPEED_MS, AI_ACCEL_MS2, travelTime, AI_REACTION_TIME_S,
  );
}

export function reachableAiPos(from: Vec2, to: Vec2, travelTime: number): Vec2 {
  return reachablePosWithAcceleration(
    from, to, AI_MAX_SPEED_MS, AI_ACCEL_MS2, travelTime, AI_REACTION_TIME_S,
  );
}

// ── isReachable (used inside validateReturn — always true since playerPos == landing) ──

export function isReachable(ballLanding: Vec2, playerPos: Vec2, ballTravelTime: number): boolean {
  const dist = mag(sub(ballLanding, playerPos));
  return dist / PLAYER_MAX_SPEED_MS <= ballTravelTime;
}

export function computeDeflectionAngle(incomingDir: Vec2, outgoingDir: Vec2): number {
  const d = Math.max(-1, Math.min(1, dot(norm(incomingDir), norm(outgoingDir))));
  return Math.acos(d);
}

export function computeReturnSpeed(incomingSpeed: number, deflectionAngle: number): number {
  const cosFactor = Math.sin(deflectionAngle); // peaks at 90°
  return Math.max(BASE_SWING_POWER, BASE_SWING_POWER + incomingSpeed * TRANSFER_COEFFICIENT * cosFactor);
}

export function checkNetClearance(
  hitPoint: Vec2, targetPoint: Vec2, speed: number, startHeight = HIT_HEIGHT,
): boolean {
  const NET_Y = COURT.netYM;
  const totalHorizDist = mag(sub(targetPoint, hitPoint));
  if (totalHorizDist < 0.01) return false;

  const minY = Math.min(hitPoint.y, targetPoint.y);
  const maxY = Math.max(hitPoint.y, targetPoint.y);
  if (NET_Y < minY || NET_Y > maxY) return true;

  const tLand     = totalHorizDist / speed;
  const vz        = computeVzFromHeight(startHeight, tLand);
  const distToNet = Math.abs(NET_Y - hitPoint.y);
  const tNet      = (distToNet / Math.abs(targetPoint.y - hitPoint.y)) * tLand;

  return ballHeightAt(tNet, vz, startHeight) >= COURT.netHeightCenterM;
}

// ── Serve validation ──────────────────────────────────────────────────────────

function isInAiServiceBox(point: Vec2): boolean {
  const { widthM, netYM, serviceBoxDepthM } = COURT;
  return point.x >= 0 && point.x <= widthM
    && point.y >= netYM && point.y <= netYM + serviceBoxDepthM;
}

export function validateServe(playerPos: Vec2, target: Vec2): ShotValidation {
  if (!isInAiServiceBox(target)) {
    return { valid: false, reason: 'out_of_bounds' };
  }

  const speed = 20 + Math.random() * 8; // 20–28 m/s

  // Serve contact is overhead — use SERVE_HIT_HEIGHT for net clearance check
  if (!checkNetClearance(playerPos, target, speed, SERVE_HIT_HEIGHT)) {
    return { valid: false, reason: 'net_fault' };
  }

  const dist       = mag(sub(target, playerPos));
  const travelTime = Math.max(dist / speed, 0.8);

  const returnShot: Shot = {
    origin:     playerPos,
    landing:    target,
    speed,
    spinType:   'flat',
    travelTime,
    hitHeight:  SERVE_HIT_HEIGHT,
    vz:         computeVzFromHeight(SERVE_HIT_HEIGHT, travelTime),
  };

  return { valid: true, returnShot };
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

  const dist       = mag(sub(target, incoming.landing));
  const travelTime = Math.max(dist / returnSpeed, 0.8);

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
