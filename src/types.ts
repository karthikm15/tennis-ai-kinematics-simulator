import type { ShotRecord } from './engine/playerProfile';
export type { ShotRecord };

export interface Vec2 {
  x: number;
  y: number;
}

export type SpinType = 'flat' | 'topspin' | 'slice';

export interface Shot {
  origin: Vec2;
  landing: Vec2;
  speed: number;      // m/s
  spinType: SpinType;
  travelTime: number; // seconds
  vz: number;         // vertical launch velocity (m/s) for arc visualization
}

export type ValidationReason = 'out_of_bounds' | 'unreachable' | 'impossible_angle' | 'net_fault';

export interface ShotValidation {
  valid: boolean;
  reason?: ValidationReason;
  returnShot?: Shot;
  deflectionAngle?: number; // radians
  returnSpeed?: number;
}

export type GamePhase =
  | 'ai_hitting'
  | 'awaiting_input'
  | 'player_hitting'
  | 'point_over';

export interface GameState {
  phase: GamePhase;
  ballPosition: Vec2;       // court meters (lerped during flight)
  playerPos: Vec2;
  aiPos: Vec2;
  playerStartPos: Vec2;     // where player was when current shot began (for run animation)
  aiStartPos: Vec2;         // where AI was when current shot began (for run animation)
  currentShot: Shot | null;
  lastValidation: ShotValidation | null;
  score: { player: number; ai: number };
  rallyCount: number;
  animationProgress: number; // 0..1
  pointResult: 'player_wins' | 'ai_wins' | null;
  shotHistory: ShotRecord[];
}
