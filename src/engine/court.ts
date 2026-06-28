import { Vec2 } from '../types';

export const COURT = {
  lengthM: 23.77,
  widthM: 8.23,
  netYM: 11.885,
  netHeightCenterM: 0.914,
  netHeightPostM: 1.07,
  serviceBoxDepthM: 6.4,
  playerBaselineY: 0,
  aiBaselineY: 23.77,
} as const;

// ── Canvas ────────────────────────────────────────────────────────────────────

export const CANVAS_W = 860;
export const CANVAS_H = 370;

// ── Perspective camera (umpire / sideline view) ───────────────────────────────
//
// Camera is 6 m outside the right sideline, at mid-court height, 4.5 m up.
// It looks at the court centre, giving a true sideline view:
//   player half (Y = 0)          → LEFT side of canvas
//   AI half    (Y = courtLength) → RIGHT side of canvas
//
// Camera axes in world space
//   Right   = (0, +1, 0)           world Y → screen X (right = +Y)
//   Forward = (FWD_X,  0, FWD_Z)   points from camera to court centre
//   Up      = (FWD_Z,  0, -FWD_X)  = Right × Forward

const CAM_X = COURT.widthM + 6;  // ≈ 14.23 m — outside right sideline
const CAM_Y = COURT.netYM;        // 11.885 m — at net midpoint along length
const CAM_Z = 4.5;                // 4.5 m above ground

const _DX = COURT.widthM / 2 - CAM_X;  // ≈ -10.115
const _DZ = -CAM_Z;                     // = -4.5
const _H  = Math.sqrt(_DX * _DX + _DZ * _DZ);

export const FWD_X = _DX / _H;   // ≈ -0.9142
export const FWD_Z = _DZ / _H;   // ≈ -0.4067

export const FOCAL     = 220;
export const PP_X      = 430;
export const PP_Y      = 218;
// y-coordinate of the vanishing horizon (z=0 ground plane at infinity)
export const HORIZON_Y = Math.round(PP_Y - FOCAL * FWD_Z / FWD_X);  // ≈ 182

// ── Projection ────────────────────────────────────────────────────────────────

export interface ProjectedPoint {
  x: number;
  y: number;
  depth: number;
}

/**
 * Projects a 3D world point (court metres) → 2D canvas pixels.
 * World axes: X = court width (0–8.23 m), Y = court length (0–23.77 m), Z = height.
 */
export function project3D(wx: number, wy: number, wz: number): ProjectedPoint {
  const dx = wx - CAM_X;
  const dy = wy - CAM_Y;
  const dz = wz - CAM_Z;

  const camRight = dy;                   // dot(d, Right=(0,1,0))
  const camFwd   = dx * FWD_X + dz * FWD_Z;
  const camUp    = dx * FWD_Z - dz * FWD_X;  // dot(d, Up=(FWD_Z,0,-FWD_X))

  if (camFwd < 0.01) return { x: -9999, y: -9999, depth: 0.01 };

  return {
    x: PP_X + FOCAL * camRight / camFwd,
    y: PP_Y - FOCAL * camUp    / camFwd,
    depth: camFwd,
  };
}

// ── Ray-cast click → court ground ─────────────────────────────────────────────

/**
 * Maps a canvas click (pixels) to a court ground position (z = 0, metres).
 * Returns { x: -1, y: -1 } when the ray misses the ground plane
 * (click above the horizon line).
 */
export function canvasClickToCourt(sx: number, sy: number): Vec2 {
  const camRight = (sx - PP_X) / FOCAL;
  const camUp    = (PP_Y - sy) / FOCAL;

  // Ray direction in world = camRight*Right + camUp*Up + Forward
  const ray_x = FWD_X + camUp * FWD_Z;
  const ray_y = camRight;
  const ray_z = FWD_Z - camUp * FWD_X;

  if (ray_z >= -0.001) return { x: -1, y: -1 };  // ray points upward

  const t = -CAM_Z / ray_z;
  return {
    x: CAM_X + t * ray_x,
    y: CAM_Y + t * ray_y,
  };
}

// ── Bounds check ──────────────────────────────────────────────────────────────

export type CourtZone = 'full' | 'player_half' | 'ai_half';

export function isInBounds(point: Vec2, zone: CourtZone = 'full'): boolean {
  if (point.x < 0 || point.x > COURT.widthM) return false;
  switch (zone) {
    case 'full':        return point.y >= 0 && point.y <= COURT.lengthM;
    case 'player_half': return point.y >= 0 && point.y <= COURT.netYM;
    case 'ai_half':     return point.y >= COURT.netYM && point.y <= COURT.lengthM;
  }
}
