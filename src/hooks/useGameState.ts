import { useReducer, useEffect, useRef, useState } from 'react';
import {
  GameState, GamePhase, Shot, ShotValidation, Vec2, ShotRecord, ValidationReason,
  Difficulty, PlayStyle, TennisScore, MatchStats,
} from '../types';
import {
  validateReturn, validateServe,
  canAiReach, canPlayerReach,
  reachableAiPos, reachablePlayerPos,
  checkNetClearance,
  computeDeflectionAngle, computeVzFromHeight,
  PLAYER_MAX_SPEED_MS, AI_MAX_SPEED_MS, HIT_HEIGHT,
} from '../engine/kinematics';
import { generateAiShot, generateAiServe, defaultAiPos, defaultPlayerPos } from '../engine/ai';
import { canvasClickToCourt } from '../engine/court';
import { initRLAgent, rlAgentShot, resetRLFrameBuffer, buildRLConfig, RLShotConfig } from '../engine/rl_agent';
import {
  MAX_ENERGY, clampEnergy, energySpeedFactor,
  recoverWhileResting, recoverWhileTracking, recoverBetweenPoints,
  runCost, swingCost, serveCost, isHardShot,
  maxAffordableReturnSpeed, maxAffordableServeSpeed,
  MIN_AI_SHOT_SPEED,
} from '../engine/energy';

// ── Module-level AI config (updated by hook when difficulty/style change) ─────

let _rlReady  = false;
let _rlConfig: RLShotConfig = buildRLConfig('medium', 'balanced');

function _getAiShot(aiPos: Vec2, playerPos: Vec2, lastBall: Vec2): Shot {
  if (_rlReady) {
    try { return rlAgentShot(aiPos, playerPos, lastBall, _rlConfig); } catch { /* fall through */ }
  }
  return generateAiShot(aiPos, playerPos);
}

// ── Energy helpers ────────────────────────────────────────────────────────────

function distM(a: Vec2, b: Vec2): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function sub2(a: Vec2, b: Vec2): Vec2 {
  return { x: a.x - b.x, y: a.y - b.y };
}

// Slow a shot down to what the hitter's tank can pay for. A slower ball
// can't arrive earlier, so travel time only ever grows (loopier, more
// attackable — exactly what a tired shot looks like).
function clampShotPower(shot: Shot, maxSpeed: number): Shot {
  if (shot.speed <= maxSpeed) return shot;
  const speed      = Math.max(MIN_AI_SHOT_SPEED, maxSpeed);
  const dist       = distM(shot.origin, shot.landing);
  const travelTime = Math.max(dist / speed, shot.travelTime);
  const hitHeight  = shot.hitHeight ?? HIT_HEIGHT;
  return { ...shot, speed, travelTime, vz: computeVzFromHeight(hitHeight, travelTime) };
}

// ── Tennis scoring helpers ────────────────────────────────────────────────────

function makeInitialScore(): TennisScore {
  return { pointsInGame: { player: 0, ai: 0 }, games: { player: 0, ai: 0 }, isTiebreak: false };
}

function makeInitialStats(): MatchStats {
  return { totalPoints: { player: 0, ai: 0 }, longestRally: 0, rallyTotal: 0, rallyCount: 0 };
}

function advancePoint(
  score: TennisScore,
  winner: 'player' | 'ai',
): { score: TennisScore; setWon: boolean } {
  const newPts = { ...score.pointsInGame, [winner]: score.pointsInGame[winner] + 1 };
  const pw = newPts[winner];
  const pl = newPts[winner === 'player' ? 'ai' : 'player'];

  const gameWon = score.isTiebreak ? (pw >= 7 && pw - pl >= 2) : (pw >= 4 && pw - pl >= 2);

  if (!gameWon) {
    return { score: { ...score, pointsInGame: newPts }, setWon: false };
  }

  const newGames = { ...score.games, [winner]: score.games[winner] + 1 };
  const gw = newGames[winner];
  const gl = newGames[winner === 'player' ? 'ai' : 'player'];

  const setWon = score.isTiebreak || (gw >= 6 && gw - gl >= 2);
  const goTiebreak = !score.isTiebreak && newGames.player === 6 && newGames.ai === 6;

  return {
    score: {
      pointsInGame: { player: 0, ai: 0 },
      games: newGames,
      isTiebreak: !setWon && goTiebreak,
    },
    setWon,
  };
}

function updateStats(stats: MatchStats, winner: 'player' | 'ai', rally: number): MatchStats {
  return {
    totalPoints: { ...stats.totalPoints, [winner]: stats.totalPoints[winner] + 1 },
    longestRally: Math.max(stats.longestRally, rally),
    rallyTotal:   stats.rallyTotal + rally,
    rallyCount:   stats.rallyCount + 1,
  };
}

// ── Actions ───────────────────────────────────────────────────────────────────

type Action =
  | { type: 'INIT_AI_SHOT'; shot: Shot }
  | { type: 'RECORD_SHOT'; record: ShotRecord }
  | { type: 'SET_PROGRESS'; progress: number }
  | { type: 'AI_SHOT_LANDED' }
  | { type: 'PLAYER_CLICKED'; validation: ShotValidation }
  | { type: 'PLAYER_SERVED'; shot: Shot }
  | { type: 'PLAYER_SHOT_LANDED' }
  | { type: 'RESET_POINT' }
  | { type: 'RESET_MATCH' };

// ── Initial state ─────────────────────────────────────────────────────────────

function makeInitialState(): GameState {
  const aiPos     = defaultAiPos();
  const playerPos = defaultPlayerPos();
  const servingPlayer: 'player' | 'ai' = Math.random() < 0.5 ? 'player' : 'ai';

  const energyFields = {
    playerEnergy: MAX_ENERGY,
    aiEnergy: MAX_ENERGY,
    playerWinded: false,
    aiWinded: false,
    playerLastEffort: null,
    aiLastEffort: null,
    pendingPlayerRunCost: 0,
    exhaustedLoser: null,
  } as const;

  if (servingPlayer === 'ai') {
    const firstShot = generateAiServe(aiPos);
    const cost = serveCost(firstShot.speed);
    return {
      phase: 'ai_hitting',
      ballPosition: firstShot.origin,
      playerPos,
      aiPos,
      playerStartPos: playerPos,
      aiStartPos:     aiPos,
      currentShot: firstShot,
      lastValidation: null,
      tennisScore: makeInitialScore(),
      rallyCount: 0,
      animationProgress: 0,
      pointResult: null,
      shotHistory: [],
      servingPlayer,
      matchOver: false,
      matchStats: makeInitialStats(),
      ...energyFields,
      aiEnergy: clampEnergy(MAX_ENERGY - cost),
      aiWinded: isHardShot(cost),
      aiLastEffort: cost,
    };
  } else {
    return {
      phase: 'awaiting_serve',
      ballPosition: playerPos,
      playerPos,
      aiPos,
      playerStartPos: playerPos,
      aiStartPos:     aiPos,
      currentShot: null,
      lastValidation: null,
      tennisScore: makeInitialScore(),
      rallyCount: 0,
      animationProgress: 0,
      pointResult: null,
      shotHistory: [],
      servingPlayer,
      matchOver: false,
      matchStats: makeInitialStats(),
      ...energyFields,
    };
  }
}

// ── Reducer ───────────────────────────────────────────────────────────────────

function lerp(a: Vec2, b: Vec2, t: number): Vec2 {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

function reducer(state: GameState, action: Action): GameState {
  switch (action.type) {

    case 'RESET_MATCH': {
      resetRLFrameBuffer();
      return makeInitialState();
    }

    case 'INIT_AI_SHOT': {
      return {
        ...state,
        phase: 'ai_hitting',
        currentShot: action.shot,
        ballPosition: action.shot.origin,
        animationProgress: 0,
        lastValidation: null,
        pointResult: null,
        playerStartPos: state.playerPos,
        aiStartPos:     state.aiPos,
        exhaustedLoser: null,
      };
    }

    case 'SET_PROGRESS': {
      if (!state.currentShot) return state;
      return {
        ...state,
        animationProgress: action.progress,
        ballPosition: lerp(state.currentShot.origin, state.currentShot.landing, action.progress),
      };
    }

    case 'AI_SHOT_LANDED': {
      if (!state.currentShot) return state;
      const { currentShot, playerStartPos } = state;
      const isServe = state.rallyCount === 0;
      const tt = currentShot.travelTime;

      // Both players recover during the ball's flight: the AI rests after its
      // hit, the player recovers more slowly while chasing the incoming ball.
      let playerEnergy = recoverWhileTracking(state.playerEnergy, tt);
      const aiEnergy   = recoverWhileResting(state.aiEnergy, tt);
      const speedFactor = energySpeedFactor(playerEnergy);

      if (!isServe && !canPlayerReach(playerStartPos, currentShot.landing, tt, speedFactor)) {
        // If fresh legs would have gotten there, this point was lost to fatigue
        const exhausted = canPlayerReach(playerStartPos, currentShot.landing, tt);
        const stoppedAt = reachablePlayerPos(playerStartPos, currentShot.landing, tt, speedFactor);
        const { score, setWon } = advancePoint(state.tennisScore, 'ai');
        return {
          ...state,
          phase: 'point_over',
          playerPos: stoppedAt,
          ballPosition: currentShot.landing,
          animationProgress: 1,
          tennisScore: score,
          pointResult: 'ai_wins',
          matchOver: setWon,
          matchStats: updateStats(state.matchStats, 'ai', state.rallyCount),
          playerEnergy,
          aiEnergy,
          exhaustedLoser: exhausted ? 'player' : null,
        };
      }

      // Charge the sprint to the ball now; it folds into the shot's total
      // effort when the swing happens.
      const rc = runCost(
        distM(playerStartPos, currentShot.landing), tt, PLAYER_MAX_SPEED_MS * speedFactor,
      );
      playerEnergy = clampEnergy(playerEnergy - rc);

      return {
        ...state,
        phase: 'awaiting_input',
        playerPos: currentShot.landing,
        ballPosition: currentShot.landing,
        animationProgress: 1,
        playerEnergy,
        aiEnergy,
        pendingPlayerRunCost: rc,
      };
    }

    case 'PLAYER_CLICKED': {
      const { validation } = action;
      if (!validation.valid) {
        const { score, setWon } = advancePoint(state.tennisScore, 'ai');
        return {
          ...state,
          phase: 'point_over',
          lastValidation: validation,
          tennisScore: score,
          pointResult: 'ai_wins',
          matchOver: setWon,
          matchStats: updateStats(state.matchStats, 'ai', state.rallyCount),
        };
      }
      const sc = swingCost(
        validation.returnSpeed ?? validation.returnShot!.speed,
        validation.deflectionAngle ?? 0,
      );
      return {
        ...state,
        phase: 'player_hitting',
        currentShot: validation.returnShot!,
        ballPosition: validation.returnShot!.origin,
        animationProgress: 0,
        lastValidation: validation,
        rallyCount: state.rallyCount + 1,
        playerStartPos: state.playerPos,
        aiStartPos:     state.aiPos,
        playerEnergy: clampEnergy(state.playerEnergy - sc),
        playerWinded: isHardShot(sc),
        playerLastEffort: state.pendingPlayerRunCost + sc,
        pendingPlayerRunCost: 0,
      };
    }

    case 'PLAYER_SERVED': {
      const sc = serveCost(action.shot.speed);
      return {
        ...state,
        phase: 'player_hitting',
        currentShot: action.shot,
        ballPosition: action.shot.origin,
        animationProgress: 0,
        rallyCount: 0,
        playerStartPos: state.playerPos,
        aiStartPos:     state.aiPos,
        playerEnergy: clampEnergy(state.playerEnergy - sc),
        playerWinded: isHardShot(sc),
        playerLastEffort: sc,
        pendingPlayerRunCost: 0,
      };
    }

    case 'PLAYER_SHOT_LANDED': {
      if (!state.currentShot) return state;
      const { currentShot, aiStartPos } = state;
      const isServe = state.rallyCount === 0;
      const tt = currentShot.travelTime;

      // Mirror of AI_SHOT_LANDED: player rests after hitting, AI recovers
      // more slowly while chasing the incoming ball.
      const playerEnergy = recoverWhileResting(state.playerEnergy, tt);
      let aiEnergy       = recoverWhileTracking(state.aiEnergy, tt);
      const speedFactor  = energySpeedFactor(aiEnergy);

      if (!isServe && !canAiReach(aiStartPos, currentShot.landing, tt, speedFactor)) {
        const exhausted = canAiReach(aiStartPos, currentShot.landing, tt);
        const stoppedAt = reachableAiPos(aiStartPos, currentShot.landing, tt, speedFactor);
        const { score, setWon } = advancePoint(state.tennisScore, 'player');
        return {
          ...state,
          phase: 'point_over',
          aiPos: stoppedAt,
          ballPosition: currentShot.landing,
          animationProgress: 1,
          tennisScore: score,
          pointResult: 'player_wins',
          matchOver: setWon,
          matchStats: updateStats(state.matchStats, 'player', state.rallyCount),
          playerEnergy,
          aiEnergy,
          exhaustedLoser: exhausted ? 'ai' : null,
        };
      }

      const newAiPos = currentShot.landing;
      const rc = runCost(distM(aiStartPos, newAiPos), tt, AI_MAX_SPEED_MS * speedFactor);
      aiEnergy = clampEnergy(aiEnergy - rc);

      // AI's reply is capped by what its tank can pay for right now
      const rawShot  = _getAiShot(newAiPos, state.playerPos, currentShot.landing);
      const nextShot = clampShotPower(rawShot, maxAffordableReturnSpeed(aiEnergy, state.aiWinded));
      const deflAngle = computeDeflectionAngle(
        sub2(currentShot.landing, currentShot.origin),
        sub2(nextShot.landing, nextShot.origin),
      );
      const sc = swingCost(nextShot.speed, deflAngle);
      aiEnergy = clampEnergy(aiEnergy - sc);
      const aiWinded     = isHardShot(sc);
      const aiLastEffort = rc + sc;

      if (!checkNetClearance(nextShot.origin, nextShot.landing, nextShot.speed)) {
        const { score, setWon } = advancePoint(state.tennisScore, 'player');
        return {
          ...state,
          phase: 'point_over',
          aiPos: newAiPos,
          ballPosition: newAiPos,
          animationProgress: 1,
          tennisScore: score,
          pointResult: 'player_wins',
          lastValidation: { valid: false, reason: 'net_fault' },
          matchOver: setWon,
          matchStats: updateStats(state.matchStats, 'player', state.rallyCount),
          playerEnergy,
          aiEnergy,
          aiWinded,
          aiLastEffort,
        };
      }

      return {
        ...state,
        phase: 'ai_hitting',
        aiPos: newAiPos,
        currentShot: nextShot,
        ballPosition: nextShot.origin,
        animationProgress: 0,
        rallyCount: state.rallyCount + 1,
        playerStartPos: state.playerPos,
        aiStartPos:     newAiPos,
        playerEnergy,
        aiEnergy,
        aiWinded,
        aiLastEffort,
      };
    }

    case 'RESET_POINT': {
      const aiPos     = defaultAiPos();
      const playerPos = defaultPlayerPos();
      const servingPlayer: 'player' | 'ai' = state.servingPlayer === 'player' ? 'ai' : 'player';
      resetRLFrameBuffer();

      // The ~25s between points restores a big chunk of both tanks and
      // clears any winded state — fatigue only partially carries over.
      const playerEnergy = recoverBetweenPoints(state.playerEnergy);
      let aiEnergy       = recoverBetweenPoints(state.aiEnergy);

      const energyReset = {
        playerEnergy,
        playerWinded: false,
        aiWinded: false,
        playerLastEffort: null,
        aiLastEffort: null,
        pendingPlayerRunCost: 0,
        exhaustedLoser: null,
      };

      if (servingPlayer === 'ai') {
        const nextShot = clampShotPower(
          generateAiServe(aiPos), maxAffordableServeSpeed(aiEnergy, false),
        );
        const cost = serveCost(nextShot.speed);
        aiEnergy = clampEnergy(aiEnergy - cost);
        return {
          ...state,
          phase: 'ai_hitting',
          aiPos,
          playerPos,
          playerStartPos: playerPos,
          aiStartPos:     aiPos,
          currentShot: nextShot,
          ballPosition: nextShot.origin,
          animationProgress: 0,
          lastValidation: null,
          pointResult: null,
          rallyCount: 0,
          servingPlayer,
          ...energyReset,
          aiEnergy,
          aiWinded: isHardShot(cost),
          aiLastEffort: cost,
        };
      } else {
        return {
          ...state,
          phase: 'awaiting_serve',
          aiPos,
          playerPos,
          playerStartPos: playerPos,
          aiStartPos:     aiPos,
          currentShot: null,
          ballPosition: playerPos,
          animationProgress: 0,
          lastValidation: null,
          pointResult: null,
          rallyCount: 0,
          servingPlayer,
          ...energyReset,
          aiEnergy,
        };
      }
    }

    case 'RECORD_SHOT': {
      return { ...state, shotHistory: [...state.shotHistory, action.record] };
    }

    default:
      return state;
  }
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useGameState(difficulty: Difficulty = 'medium', style: PlayStyle = 'balanced') {
  const [state, dispatch] = useReducer(reducer, undefined, makeInitialState);
  const [serveError, setServeError] = useState<ValidationReason | null>(null);
  const serveErrorTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load RL weights once on mount; fall back to heuristic until ready
  useEffect(() => {
    initRLAgent().then(() => { _rlReady = true; }).catch(console.error);
  }, []);

  // Sync AI config when difficulty/style change
  useEffect(() => {
    _rlConfig = buildRLConfig(difficulty, style);
  }, [difficulty, style]);

  // Reset RL frame history at the start of every new point
  useEffect(() => {
    if (state.rallyCount === 0) resetRLFrameBuffer();
  }, [state.rallyCount]);

  // Track current phase and shot in a ref so the rAF closure sees fresh values
  const phaseRef = useRef<GamePhase>(state.phase);
  const shotRef  = useRef<Shot | null>(state.currentShot);
  useEffect(() => { phaseRef.current = state.phase; },         [state.phase]);
  useEffect(() => { shotRef.current  = state.currentShot; },   [state.currentShot]);

  // Animation loop — restarts whenever phase transitions into a flight phase
  useEffect(() => {
    const phase = state.phase;
    if (phase !== 'ai_hitting' && phase !== 'player_hitting') return;

    const shot = state.currentShot;
    if (!shot) return;

    const durationSec = shot.travelTime;
    let rafId: number;
    let startTime: number | null = null;

    function tick(timestamp: number) {
      if (!startTime) startTime = timestamp;
      const elapsed  = (timestamp - startTime) / 1000;
      const progress = Math.min(elapsed / durationSec, 1);

      dispatch({ type: 'SET_PROGRESS', progress });

      if (progress < 1) {
        rafId = requestAnimationFrame(tick);
      } else {
        if (phaseRef.current === 'ai_hitting') {
          dispatch({ type: 'AI_SHOT_LANDED' });
        } else if (phaseRef.current === 'player_hitting') {
          dispatch({ type: 'PLAYER_SHOT_LANDED' });
        }
      }
    }

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.phase, state.currentShot?.origin, state.currentShot?.landing]);

  // Auto-reset after point_over (skip if match is over — App handles that transition)
  useEffect(() => {
    if (state.phase !== 'point_over' || state.matchOver) return;
    const id = setTimeout(() => dispatch({ type: 'RESET_POINT' }), 2500);
    return () => clearTimeout(id);
  }, [state.phase, state.matchOver]);

  function showServeError(reason: ValidationReason) {
    if (serveErrorTimer.current) clearTimeout(serveErrorTimer.current);
    setServeError(reason);
    serveErrorTimer.current = setTimeout(() => setServeError(null), 1500);
  }

  function handleCanvasClick(canvasPx: Vec2) {
    if (state.phase === 'awaiting_serve') {
      const target = canvasClickToCourt(canvasPx.x, canvasPx.y);
      if (target.x < 0) return;
      const maxServe   = maxAffordableServeSpeed(state.playerEnergy, state.playerWinded);
      const validation = validateServe(state.playerPos, target, maxServe);
      if (validation.valid && validation.returnShot) {
        setServeError(null);
        dispatch({ type: 'PLAYER_SERVED', shot: validation.returnShot });
      } else if (validation.reason) {
        showServeError(validation.reason);
      }
      return;
    }

    if (state.phase !== 'awaiting_input' || !state.currentShot) return;
    const target     = canvasClickToCourt(canvasPx.x, canvasPx.y);
    const maxReturn  = maxAffordableReturnSpeed(state.playerEnergy, state.playerWinded);
    const validation = validateReturn(state.currentShot, state.playerPos, target, maxReturn);

    if (target.x >= 0) {
      const record: ShotRecord = {
        ballLandX:      state.currentShot.landing.x,
        ballLandY:      state.currentShot.landing.y,
        targetX:        target.x,
        targetY:        target.y,
        valid:          validation.valid,
        reason:         validation.reason,
        deflectionAngle: validation.deflectionAngle,
        returnSpeed:    validation.returnSpeed,
      };
      dispatch({ type: 'RECORD_SHOT', record });
    }

    dispatch({ type: 'PLAYER_CLICKED', validation });
  }

  function resetMatch() {
    dispatch({ type: 'RESET_MATCH' });
  }

  return { state, handleCanvasClick, serveError, resetMatch };
}
