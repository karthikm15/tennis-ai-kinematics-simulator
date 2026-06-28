import { Vec2, Shot } from '../types';
import { COURT } from './court';
import { computeVz } from './kinematics';

function sub(a: Vec2, b: Vec2): Vec2 { return { x: a.x - b.x, y: a.y - b.y }; }
function mag(v: Vec2): number { return Math.sqrt(v.x * v.x + v.y * v.y); }

const MARGIN = 0.3; // m from sidelines/baseline

export function generateAiShot(aiPos: Vec2): Shot {
  const landX = MARGIN + Math.random() * (COURT.widthM - 2 * MARGIN);
  // Keep shots in the middle 60% of the player's half so they're reachable
  const landY = COURT.netYM * 0.3 + Math.random() * (COURT.netYM * 0.6);
  const landing: Vec2 = { x: landX, y: landY };

  const speed      = 12 + Math.random() * 6; // 12–18 m/s (allows more reaction time)
  const dist       = mag(sub(landing, aiPos));
  const travelTime = Math.max(dist / speed, 0.8); // min 0.8s travel time

  return {
    origin:     aiPos,
    landing,
    speed,
    spinType:   'flat',
    travelTime,
    vz:         computeVz(travelTime),
  };
}

export function defaultAiPos(): Vec2 {
  return { x: COURT.widthM / 2, y: COURT.aiBaselineY - 1.5 };
}

export function defaultPlayerPos(): Vec2 {
  return { x: COURT.widthM / 2, y: COURT.playerBaselineY + 1.5 };
}
