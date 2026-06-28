import { useReducer, useEffect, useRef } from 'react';
import { GameState, GamePhase, Shot, ShotValidation, Vec2, ShotRecord } from '../types';
import { validateReturn, canReach, reachablePos, PLAYER_MAX_SPEED_MS, AI_MAX_SPEED_MS } from '../engine/kinematics';
import { generateAiShot, defaultAiPos, defaultPlayerPos } from '../engine/ai';
import { canvasClickToCourt } from '../engine/court';

// ── Actions ───────────────────────────────────────────────────────────────────

type Action =
  | { type: 'INIT_AI_SHOT'; shot: Shot }
  | { type: 'RECORD_SHOT'; record: ShotRecord }
  | { type: 'SET_PROGRESS'; progress: number }
  | { type: 'AI_SHOT_LANDED' }
  | { type: 'PLAYER_CLICKED'; validation: ShotValidation }
  | { type: 'PLAYER_SHOT_LANDED' }
  | { type: 'RESET_POINT' };

// ── Initial state ─────────────────────────────────────────────────────────────

function makeInitialState(): GameState {
  const aiPos     = defaultAiPos();
  const playerPos = defaultPlayerPos();
  const firstShot = generateAiShot(aiPos);
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
  };
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

      // Check if player could have run from their start position to the landing
      if (!canReach(playerStartPos, currentShot.landing, PLAYER_MAX_SPEED_MS, currentShot.travelTime)) {
        const stoppedAt = reachablePos(playerStartPos, currentShot.landing, PLAYER_MAX_SPEED_MS, currentShot.travelTime);
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

    case 'PLAYER_SHOT_LANDED': {
      if (!state.currentShot) return state;
      const { currentShot, aiStartPos } = state;

      // Check if AI could have run from their start position to the landing
      if (!canReach(aiStartPos, currentShot.landing, AI_MAX_SPEED_MS, currentShot.travelTime)) {
        const stoppedAt = reachablePos(aiStartPos, currentShot.landing, AI_MAX_SPEED_MS, currentShot.travelTime);
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

      // AI reached the ball — hit from the landing spot
      const newAiPos  = currentShot.landing;
      const nextShot  = generateAiShot(newAiPos);
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
      const nextShot  = generateAiShot(aiPos);
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
      };
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

  function handleCanvasClick(canvasPx: Vec2) {
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

  return { state, handleCanvasClick };
}
