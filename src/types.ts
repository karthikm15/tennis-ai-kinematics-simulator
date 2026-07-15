import type { ShotRecord } from './engine/playerProfile';
export type { ShotRecord };

export type Difficulty = 'easy' | 'medium' | 'hard' | 'expert';
export type PlayStyle  = 'aggressive' | 'balanced' | 'consistent';

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
  hitHeight?: number; // contact point height above court (m); defaults to HIT_HEIGHT = 1.0
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
  | 'awaiting_serve'
  | 'awaiting_input'
  | 'player_hitting'
  | 'point_over';

export interface TennisScore {
  pointsInGame: { player: number; ai: number };
  games:        { player: number; ai: number };
  isTiebreak:   boolean;
}

export interface MatchStats {
  totalPoints: { player: number; ai: number };
  longestRally: number;
  rallyTotal:   number;
  rallyCount:   number;
}

export interface GameState {
  phase: GamePhase;
  ballPosition: Vec2;
  playerPos: Vec2;
  aiPos: Vec2;
  playerStartPos: Vec2;
  aiStartPos: Vec2;
  currentShot: Shot | null;
  lastValidation: ShotValidation | null;
  tennisScore: TennisScore;
  rallyCount: number;
  animationProgress: number;
  pointResult: 'player_wins' | 'ai_wins' | null;
  shotHistory: ShotRecord[];
  servingPlayer: 'player' | 'ai';
  matchOver: boolean;
  matchStats: MatchStats;
  // ── Energy / fatigue ──
  playerEnergy: number;            // 0–100 stamina tank
  aiEnergy: number;
  playerWinded: boolean;           // true right after a hard hit — next swing is capped
  aiWinded: boolean;
  playerLastEffort: number | null; // energy spent on the player's most recent shot (run + swing)
  aiLastEffort: number | null;
  pendingPlayerRunCost: number;    // run cost charged on ball arrival, folded into effort on the swing
  exhaustedLoser: 'player' | 'ai' | null; // point lost because fatigue made the ball unreachable
}
