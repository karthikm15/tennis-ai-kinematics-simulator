import { useReducer, useEffect, useRef, useState } from 'react';
import { GameState, GamePhase, Shot, ShotValidation, Vec2, ShotRecord, ValidationReason } from '../types';
import {
  validateReturn, validateServe,
  canAiReach, canPlayerReach,
  reachableAiPos, reachablePlayerPos,
  checkNetClearance,
} from '../engine/kinematics';
import { generateAiShot, generateAiServe, defaultAiPos, defaultPlayerPos } from '../engine/ai';
import { canvasClickToCourt } from '../engine/court';
import { initRLAgent, rlAgentShot, resetRLFrameBuffer } from '../engine/rl_agent';

// Module-level flag: flips to true once weights are fetched
let _rlReady = false;

function _getAiShot(aiPos: Vec2, playerPos: Vec2, lastBall: Vec2): Shot {
  if (_rlReady) {
    try { return rlAgentShot(aiPos, playerPos, lastBall); } catch { /* fall through */ }
  }
  return generateAiShot(aiPos, playerPos);
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
  | { type: 'RESET_POINT' };

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
      score: { player: 0, ai: 0 },
      rallyCount: 0,
      animationProgress: 0,
      pointResult: null,
      shotHistory: [],
      servingPlayer,
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
      score: { player: 0, ai: 0 },
      rallyCount: 0,
      animationProgress: 0,
      pointResult: null,
      shotHistory: [],
      servingPlayer,
    };
  }
}

// ── Reducer ───────────────────────────────────────────────────────────────────

function lerp(a: Vec2, b: Vec2, t: number): Vec2 {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

function reducer(state: GameState, action: Action): GameState {
  switch (action.type) {

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

      // rallyCount === 0 means this is the serve — receiver always gets to it.
      // Movement checks only apply once the rally is under way (shot 2+).
      const isServe = state.rallyCount === 0;

      if (!isServe && !canPlayerReach(playerStartPos, currentShot.landing, currentShot.travelTime)) {
        const stoppedAt = reachablePlayerPos(playerStartPos, currentShot.landing, currentShot.travelTime);
        return {
          ...state,
          phase: 'point_over',
          playerPos: stoppedAt,
          ballPosition: currentShot.landing,
          animationProgress: 1,
          score: { ...state.score, ai: state.score.ai + 1 },
          pointResult: 'ai_wins',
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
        return {
          ...state,
          phase: 'point_over',
          lastValidation: validation,
          score: { ...state.score, ai: state.score.ai + 1 },
          pointResult: 'ai_wins',
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
        // Player stays at landing (they just hit); AI starts running from here
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

      // rallyCount === 0 means this is the player's serve — AI always returns it.
      const isServe = state.rallyCount === 0;

      if (!isServe && !canAiReach(aiStartPos, currentShot.landing, currentShot.travelTime)) {
        const stoppedAt = reachableAiPos(aiStartPos, currentShot.landing, currentShot.travelTime);
        return {
          ...state,
          phase: 'point_over',
          aiPos: stoppedAt,
          ballPosition: currentShot.landing,
          animationProgress: 1,
          score: { ...state.score, player: state.score.player + 1 },
          pointResult: 'player_wins',
        };
      }

      // AI reached the ball — generate return shot, check net clearance
      const newAiPos  = currentShot.landing;
      const nextShot  = _getAiShot(newAiPos, state.playerPos, currentShot.landing);

      if (!checkNetClearance(nextShot.origin, nextShot.landing, nextShot.speed)) {
        return {
          ...state,
          phase: 'point_over',
          aiPos: newAiPos,
          ballPosition: newAiPos,
          animationProgress: 1,
          score: { ...state.score, player: state.score.player + 1 },
          pointResult: 'player_wins',
          lastValidation: { valid: false, reason: 'net_fault' },
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
      // Alternate server each point
      const servingPlayer: 'player' | 'ai' = state.servingPlayer === 'player' ? 'ai' : 'player';

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

export function useGameState() {
  const [state, dispatch] = useReducer(reducer, undefined, makeInitialState);
  const [serveError, setServeError] = useState<ValidationReason | null>(null);
  const serveErrorTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load RL weights once on mount; fall back to heuristic until ready
  useEffect(() => {
    initRLAgent().then(() => { _rlReady = true; }).catch(console.error);
  }, []);

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

  // Auto-reset after point_over
  useEffect(() => {
    if (state.phase !== 'point_over') return;
    const id = setTimeout(() => dispatch({ type: 'RESET_POINT' }), 2500);
    return () => clearTimeout(id);
  }, [state.phase]);

  function showServeError(reason: ValidationReason) {
    if (serveErrorTimer.current) clearTimeout(serveErrorTimer.current);
    setServeError(reason);
    serveErrorTimer.current = setTimeout(() => setServeError(null), 1500);
  }

  function handleCanvasClick(canvasPx: Vec2) {
    if (state.phase === 'awaiting_serve') {
      const target = canvasClickToCourt(canvasPx.x, canvasPx.y);
      if (target.x < 0) return; // above horizon
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

    // Record this attempt (valid or not) — skip completely off-court clicks (above horizon)
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

  return { state, handleCanvasClick, serveError };
}
