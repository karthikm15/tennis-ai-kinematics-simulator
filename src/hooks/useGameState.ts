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
} from '../engine/kinematics';
import { generateAiShot, generateAiServe, defaultAiPos, defaultPlayerPos } from '../engine/ai';
import { canvasClickToCourt } from '../engine/court';
import { initRLAgent, rlAgentShot, resetRLFrameBuffer, buildRLConfig, RLShotConfig } from '../engine/rl_agent';

// ── Module-level AI config (updated by hook when difficulty/style change) ─────

let _rlReady  = false;
let _rlConfig: RLShotConfig = buildRLConfig('medium', 'balanced');

function _getAiShot(aiPos: Vec2, playerPos: Vec2, lastBall: Vec2): Shot {
  if (_rlReady) {
    try { return rlAgentShot(aiPos, playerPos, lastBall, _rlConfig); } catch { /* fall through */ }
  }
  return generateAiShot(aiPos, playerPos);
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

  if (servingPlayer === 'ai') {
    const firstShot = generateAiServe(aiPos);
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

      if (!isServe && !canPlayerReach(playerStartPos, currentShot.landing, currentShot.travelTime)) {
        const stoppedAt = reachablePlayerPos(playerStartPos, currentShot.landing, currentShot.travelTime);
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
        };
      }

      return {
        ...state,
        phase: 'awaiting_input',
        playerPos: currentShot.landing,
        ballPosition: currentShot.landing,
        animationProgress: 1,
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
      };
    }

    case 'PLAYER_SERVED': {
      return {
        ...state,
        phase: 'player_hitting',
        currentShot: action.shot,
        ballPosition: action.shot.origin,
        animationProgress: 0,
        rallyCount: 0,
        playerStartPos: state.playerPos,
        aiStartPos:     state.aiPos,
      };
    }

    case 'PLAYER_SHOT_LANDED': {
      if (!state.currentShot) return state;
      const { currentShot, aiStartPos } = state;
      const isServe = state.rallyCount === 0;

      if (!isServe && !canAiReach(aiStartPos, currentShot.landing, currentShot.travelTime)) {
        const stoppedAt = reachableAiPos(aiStartPos, currentShot.landing, currentShot.travelTime);
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
        };
      }

      const newAiPos = currentShot.landing;
      const nextShot = _getAiShot(newAiPos, state.playerPos, currentShot.landing);

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
      };
    }

    case 'RESET_POINT': {
      const aiPos     = defaultAiPos();
      const playerPos = defaultPlayerPos();
      const servingPlayer: 'player' | 'ai' = state.servingPlayer === 'player' ? 'ai' : 'player';
      resetRLFrameBuffer();

      if (servingPlayer === 'ai') {
        const nextShot = generateAiServe(aiPos);
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
      const validation = validateServe(state.playerPos, target);
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
    const validation = validateReturn(state.currentShot, state.playerPos, target);

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
