import { Vec2, Shot } from '../types';
import { COURT } from './court';
import { computeVz, computeVzFromHeight, SERVE_HIT_HEIGHT } from './kinematics';

function sub(a: Vec2, b: Vec2): Vec2 { return { x: a.x - b.x, y: a.y - b.y }; }
function mag(v: Vec2): number { return Math.sqrt(v.x * v.x + v.y * v.y); }

const MARGIN = 0.3; // m from sidelines/baseline

export function generateAiShot(aiPos: Vec2, playerPos: Vec2): Shot {
  const mid  = COURT.widthM / 2;

  // 70% of the time aim to the side the player is NOT on; 30% random
  let landX: number;
  if (Math.random() < 0.70) {
    if (playerPos.x < mid) {
      // Player left of center → aim to right side
      landX = mid + MARGIN + Math.random() * (mid - 2 * MARGIN);
    } else {
      // Player right of center → aim to left side
      landX = MARGIN + Math.random() * (mid - 2 * MARGIN);
    }
  } else {
    landX = MARGIN + Math.random() * (COURT.widthM - 2 * MARGIN);
  }

  // Y: keep in middle 70% of player's half so shots are physically reachable
  const landY = COURT.netYM * 0.2 + Math.random() * (COURT.netYM * 0.7);
  const landing: Vec2 = { x: landX, y: landY };

  const speed      = 8 + Math.random() * 4; // 8–12 m/s — comfortable rally pace
  const dist       = mag(sub(landing, aiPos));
  const travelTime = Math.max(dist / speed, 1.5); // min 1.5s so ball stays in air

  return {
    origin:     aiPos,
    landing,
    speed,
    spinType:   'flat',
    travelTime,
    vz:         computeVz(travelTime),
  };
}

// Serve: faster shot landing in the player's service box (between service line and net)
export function generateAiServe(aiPos: Vec2): Shot {
  const serviceNearY = COURT.netYM - COURT.serviceBoxDepthM; // ~5.485
  const landX = MARGIN + Math.random() * (COURT.widthM - 2 * MARGIN);
  const landY = serviceNearY + MARGIN + Math.random() * (COURT.serviceBoxDepthM - 2 * MARGIN);
  const landing: Vec2 = { x: landX, y: landY };

  const speed      = 18 + Math.random() * 6; // 18–24 m/s — fast serve, visible arc
  const dist       = mag(sub(landing, aiPos));
  const travelTime = Math.max(dist / speed, 0.8); // min 0.8s so animation is clear

  return {
    origin:     aiPos,
    landing,
    speed,
    spinType:   'flat',
    travelTime,
    hitHeight:  SERVE_HIT_HEIGHT,
    vz:         computeVzFromHeight(SERVE_HIT_HEIGHT, travelTime),
  };
}

export function defaultAiPos(): Vec2 {
  return { x: COURT.widthM / 2, y: COURT.aiBaselineY - 1.5 };
}

export function defaultPlayerPos(): Vec2 {
  return { x: COURT.widthM / 2, y: COURT.playerBaselineY + 1.5 };
}
