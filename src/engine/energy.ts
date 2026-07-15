// ── Energy / fatigue model ────────────────────────────────────────────────────
//
// Each player has one stamina tank (0–100). Shots drain it, time between
// shots refills it, and the ~25s rest between points refills a big chunk.
//
// What makes a shot "difficult" (expensive), modeled on real tennis:
//   1. Running  — cost grows with distance and quadratically with sprint
//                 intensity (how close to flat-out you ran to arrive in time).
//   2. Power    — swing cost grows quadratically with ball speed above a
//                 relaxed rally pace (~kinetic energy put into the ball).
//   3. Redirect — sharply changing the ball's direction (deflection near 90°)
//                 costs extra racquet work; blocking it straight back rides
//                 the ball's own momentum and is cheap.
//   4. Serving  — flat overhead cost plus a quadratic surcharge for pace.
//
// Fatigue shows up the way it does on a real court:
//   • Legs go first — sprint speed drops with the tank, so wide balls become
//     unreachable and the player simply loses the point.
//   • One swing can only spend a fraction of what's left in the tank, so a
//     tired player can always block the ball back but can't hit big.
//   • A hard hit leaves you "winded": until you play an easy shot or rest,
//     the next swing's budget collapses — two consecutive bombs are
//     physically impossible without regenerating.

export const MAX_ENERGY = 100;

// Recovery
const RECOVER_RESTING_PER_S  = 5.0;  // your shot flying away — reset, breathe
const RECOVER_TRACKING_PER_S = 2.0;  // ball incoming — recovering but on the move
export const POINT_REST_RECOVERY = 40; // between-points rest

// Running
const RUN_BASE_PER_M    = 0.5;
const SPRINT_COST_PER_M = 2.6;       // × intensity²

// Groundstroke swing
const SWING_BASE          = 2.0;
export const COMFORT_SPEED = 10.0;   // m/s — relaxed rally pace, near-free
const POWER_COST_K        = 0.55;    // × (speed − comfort)²
const ANGLE_COST_K        = 3.0;     // × sin(deflection), peaks at 90° redirect

// Serve
const SERVE_BASE       = 5.0;
const SERVE_EASY_SPEED = 18.0;       // m/s — a spun-in second serve
const SERVE_POWER_K    = 0.25;       // × (speed − easy)²

// Burst limits
export const HARD_SHOT_COST  = 16;   // a swing this costly leaves you winded
const BURST_FRAC             = 0.60; // one swing may spend ≤ 60% of the tank
const WINDED_BURST_FRAC      = 0.25; // …only 25% while winded

// Floors — fatigue takes your legs and your power, never the ability to
// put a racquet on the ball.
export const MIN_RETURN_SPEED  = 8.0;
export const MIN_SERVE_SPEED   = 14.0;
export const MIN_AI_SHOT_SPEED = 6.0;

const FULL_SPRINT_ENERGY = 65;       // below this the legs start going

export function clampEnergy(e: number): number {
  return Math.max(0, Math.min(MAX_ENERGY, e));
}

/** Sprint speed multiplier: 1.0 above 65 energy, fading to 0.55 when empty. */
export function energySpeedFactor(energy: number): number {
  if (energy >= FULL_SPRINT_ENERGY) return 1;
  return 0.55 + 0.45 * (Math.max(0, energy) / FULL_SPRINT_ENERGY);
}

export function recoverWhileResting(energy: number, seconds: number): number {
  return clampEnergy(energy + RECOVER_RESTING_PER_S * seconds);
}

export function recoverWhileTracking(energy: number, seconds: number): number {
  return clampEnergy(energy + RECOVER_TRACKING_PER_S * seconds);
}

export function recoverBetweenPoints(energy: number): number {
  return clampEnergy(energy + POINT_REST_RECOVERY);
}

/** Cost of running `distanceM` to the ball with `availableTimeS` to get there. */
export function runCost(distanceM: number, availableTimeS: number, maxSpeedMS: number): number {
  if (distanceM < 0.05) return 0;
  const required  = distanceM / Math.max(availableTimeS, 0.1);
  const intensity = Math.min(1.2, required / Math.max(maxSpeedMS, 0.1));
  return distanceM * (RUN_BASE_PER_M + SPRINT_COST_PER_M * intensity * intensity);
}

/** Cost of a groundstroke at `ballSpeedMS` with the given redirect angle. */
export function swingCost(ballSpeedMS: number, deflectionAngleRad = 0): number {
  const over = Math.max(0, ballSpeedMS - COMFORT_SPEED);
  // sin peaks at 90° — a full perpendicular redirect — and falls off toward
  // 180°, where the return rides the incoming ball's rebound.
  const angle = Math.sin(Math.max(0, Math.min(Math.PI, deflectionAngleRad)));
  return SWING_BASE + POWER_COST_K * over * over + ANGLE_COST_K * angle;
}

export function serveCost(speedMS: number): number {
  const over = Math.max(0, speedMS - SERVE_EASY_SPEED);
  return SERVE_BASE + SERVE_POWER_K * over * over;
}

function swingBudget(energy: number, winded: boolean): number {
  return Math.max(0, energy) * (winded ? WINDED_BURST_FRAC : BURST_FRAC);
}

/** Fastest groundstroke the tank can currently pay for. */
export function maxAffordableReturnSpeed(energy: number, winded: boolean): number {
  const avail = swingBudget(energy, winded) - SWING_BASE;
  const v = COMFORT_SPEED + Math.sqrt(Math.max(0, avail) / POWER_COST_K);
  return Math.max(MIN_RETURN_SPEED, v);
}

/** Fastest serve the tank can currently pay for. */
export function maxAffordableServeSpeed(energy: number, winded: boolean): number {
  const avail = swingBudget(energy, winded) - SERVE_BASE;
  const v = SERVE_EASY_SPEED + Math.sqrt(Math.max(0, avail) / SERVE_POWER_K);
  return Math.max(MIN_SERVE_SPEED, v);
}

export function isHardShot(cost: number): boolean {
  return cost >= HARD_SHOT_COST;
}

export function effortLabel(cost: number): string {
  if (cost >= HARD_SHOT_COST) return 'Big effort';
  if (cost >= 8) return 'Moderate';
  return 'Easy';
}
