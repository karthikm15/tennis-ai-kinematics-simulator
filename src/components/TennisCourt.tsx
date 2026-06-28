import { useRef, useEffect, useCallback } from 'react';
import { GameState, Shot, Vec2 } from '../types';
import {
  CANVAS_W, CANVAS_H, COURT,
  project3D, FOCAL,
} from '../engine/court';
import { ballHeightAt, PLAYER_MAX_SPEED_MS, AI_MAX_SPEED_MS } from '../engine/kinematics';

interface Props {
  state: GameState;
  handleCanvasClick: (px: Vec2) => void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getBallZ(progress: number, shot: Shot): number {
  return ballHeightAt(progress * shot.travelTime, shot.vz);
}

function lerpN(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

// Animate a player running from startPos toward targetPos, capped by their max speed.
// Returns where they are at animation fraction t (0..1) over the shot's travel time.
function computeRunningPos(startPos: Vec2, targetPos: Vec2, maxSpeed: number, travelTime: number, t: number): Vec2 {
  const dx   = targetPos.x - startPos.x;
  const dy   = targetPos.y - startPos.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  if (dist < 0.001) return startPos;
  const maxReach    = maxSpeed * travelTime;
  const reachFrac   = Math.min(1, maxReach / dist);
  const finalX      = startPos.x + dx * reachFrac;
  const finalY      = startPos.y + dy * reachFrac;
  return { x: startPos.x + (finalX - startPos.x) * t, y: startPos.y + (finalY - startPos.y) * t };
}

type Ctx = CanvasRenderingContext2D;

function line3D(ctx: Ctx, x1: number, y1: number, z1: number,
                            x2: number, y2: number, z2: number) {
  const a = project3D(x1, y1, z1);
  const b = project3D(x2, y2, z2);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function fillPoly3D(ctx: Ctx, pts: [number, number, number][], style: string) {
  const ps = pts.map(([x, y, z]) => project3D(x, y, z));
  ctx.beginPath();
  ctx.moveTo(ps[0].x, ps[0].y);
  for (let i = 1; i < ps.length; i++) ctx.lineTo(ps[i].x, ps[i].y);
  ctx.closePath();
  ctx.fillStyle = style;
  ctx.fill();
}

// ── Clean background ──────────────────────────────────────────────────────────

function drawBackground(ctx: Ctx) {
  const W = COURT.widthM;
  const L = COURT.lengthM;

  // Solid light sky
  ctx.fillStyle = '#dce8f0';
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  // Court surroundings — standard hard-court teal/grey perimeter
  fillPoly3D(ctx, [
    [-2.5, -3, 0.01], [W+2.5, -3, 0.01],
    [W+2.5, L+3, 0.01], [-2.5, L+3, 0.01],
  ], '#5a7a6a');
}

// ── Hard court surface (US Open blue) ─────────────────────────────────────────

function drawCourt(ctx: Ctx) {
  const W   = COURT.widthM;
  const netY = COURT.netYM;
  const L   = COURT.lengthM;
  const svc = COURT.serviceBoxDepthM;
  const mid = W / 2;

  // Main surface (dark navy-blue)
  fillPoly3D(ctx, [
    [0, 0, 0], [W, 0, 0], [W, L, 0], [0, L, 0],
  ], '#2558a0');

  // Service boxes (brighter blue — standard US Open two-tone)
  fillPoly3D(ctx, [
    [0, netY-svc, 0], [W, netY-svc, 0], [W, netY, 0], [0, netY, 0],
  ], '#3468b8');
  fillPoly3D(ctx, [
    [0, netY, 0], [W, netY, 0], [W, netY+svc, 0], [0, netY+svc, 0],
  ], '#3468b8');

  // Subtle asphalt grain (faint horizontal lines)
  for (let gy = 0.4; gy < L; gy += 0.7) {
    const lp = project3D(0, gy, 0.002);
    const rp = project3D(W, gy, 0.002);
    ctx.beginPath();
    ctx.moveTo(lp.x, lp.y); ctx.lineTo(rp.x, rp.y);
    ctx.strokeStyle = 'rgba(0,0,50,0.055)';
    ctx.lineWidth = 0.6;
    ctx.stroke();
  }

  // Court lines (white)
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2.5;
  line3D(ctx, 0, 0, 0,  W, 0, 0);            // player baseline
  line3D(ctx, 0, L, 0,  W, L, 0);            // AI baseline
  line3D(ctx, 0, 0, 0,  0, L, 0);            // far sideline
  line3D(ctx, W, 0, 0,  W, L, 0);            // near sideline
  line3D(ctx, 0, netY-svc, 0, W, netY-svc, 0); // player service line
  line3D(ctx, 0, netY+svc, 0, W, netY+svc, 0); // AI service line
  line3D(ctx, mid, netY-svc, 0, mid, netY,   0); // center service line (player half)
  line3D(ctx, mid, netY,    0, mid, netY+svc, 0); // center service line (AI half)

  ctx.lineWidth = 1.5;
  line3D(ctx, mid-0.2, 0, 0, mid+0.2, 0, 0); // player baseline centre mark
  line3D(ctx, mid-0.2, L, 0, mid+0.2, L, 0); // AI baseline centre mark
}

// ── Net ───────────────────────────────────────────────────────────────────────

function drawNet(ctx: Ctx) {
  const W   = COURT.widthM;
  const netY = COURT.netYM;
  const hc  = COURT.netHeightCenterM;
  const hp  = COURT.netHeightPostM;
  const mid = W / 2;

  // Net fill
  const pts: [number,number,number][] = [
    [0, netY, 0], [W, netY, 0], [W, netY, hp], [mid, netY, hc], [0, netY, hp],
  ];
  const ps = pts.map(([x,y,z]) => project3D(x,y,z));
  ctx.beginPath();
  ctx.moveTo(ps[0].x, ps[0].y);
  ps.forEach((p,i) => { if (i > 0) ctx.lineTo(p.x, p.y); });
  ctx.closePath();
  ctx.fillStyle = 'rgba(200,215,225,0.14)';
  ctx.fill();

  // Horizontal mesh bands
  for (let i = 1; i < 7; i++) {
    const frac = i / 7;
    const zl   = hp * frac;
    const zc   = hc * frac;
    ctx.strokeStyle = `rgba(255,255,255,${0.07 + frac * 0.08})`;
    ctx.lineWidth = 0.6;
    line3D(ctx, 0,   netY, zl, mid, netY, zc);
    line3D(ctx, mid, netY, zc, W,   netY, zl);
  }

  // Vertical mesh lines
  ctx.lineWidth = 0.5;
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  for (let i = 1; i < 12; i++) {
    const x    = (i / 12) * W;
    const t    = Math.abs(x / W - 0.5) * 2;
    const zTop = hc + (hp - hc) * t;
    line3D(ctx, x, netY, 0, x, netY, zTop);
  }

  // Net posts
  ctx.strokeStyle = 'rgba(210,210,210,0.95)';
  ctx.lineWidth = 3;
  line3D(ctx, 0, netY, 0, 0, netY, hp);
  line3D(ctx, W, netY, 0, W, netY, hp);

  // Top tape (white band)
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  line3D(ctx, 0,   netY, hp,  mid, netY, hc);
  line3D(ctx, mid, netY, hc,  W,   netY, hp);
}

// ── Trajectory trail ──────────────────────────────────────────────────────────

function drawTrail(ctx: Ctx, shot: Shot, progress: number) {
  const N = 30;
  for (let i = 0; i < N; i++) {
    const p    = (i / (N - 1)) * progress;
    const wx   = lerpN(shot.origin.x, shot.landing.x, p);
    const wy   = lerpN(shot.origin.y, shot.landing.y, p);
    const wz   = getBallZ(p, shot);
    const proj = project3D(wx, wy, wz);
    const alpha = (i / N) * 0.5;
    const r    = Math.max(1.5, (FOCAL * 0.055) / proj.depth);
    ctx.beginPath();
    ctx.arc(proj.x, proj.y, r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(220,255,80,${alpha})`;
    ctx.fill();
  }
}

// ── Ball shadow ───────────────────────────────────────────────────────────────

function drawBallShadow(ctx: Ctx, wx: number, wy: number, wz: number) {
  const ground = project3D(wx, wy, 0);
  const alpha  = Math.max(0, 0.55 * (1 - wz / 5));
  const sr     = Math.max(3, (FOCAL * 0.11) / ground.depth);
  ctx.beginPath();
  ctx.ellipse(ground.x, ground.y, sr, sr * 0.32, 0, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(0,0,0,${alpha})`;
  ctx.fill();
}

// ── Tennis ball ───────────────────────────────────────────────────────────────

function drawBall(ctx: Ctx, wx: number, wy: number, wz: number) {
  const proj = project3D(wx, wy, wz);
  const r    = Math.max(4.5, (FOCAL * 0.13) / proj.depth);

  // Main ball gradient (yellow-green)
  const grad = ctx.createRadialGradient(
    proj.x - r * 0.32, proj.y - r * 0.38, r * 0.04,
    proj.x,            proj.y,             r,
  );
  grad.addColorStop(0,   '#e8ff40');
  grad.addColorStop(0.55,'#b0cc00');
  grad.addColorStop(1,   '#6e8800');
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, r, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Seam (white curved line)
  ctx.save();
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, r, 0, Math.PI * 2);
  ctx.clip();
  ctx.strokeStyle = 'rgba(255,255,255,0.45)';
  ctx.lineWidth = Math.max(0.8, r * 0.18);
  ctx.beginPath();
  ctx.arc(proj.x + r * 0.15, proj.y, r * 0.72, -0.4, Math.PI + 0.4);
  ctx.stroke();
  ctx.restore();

  // Specular highlight
  ctx.beginPath();
  ctx.arc(proj.x - r * 0.30, proj.y - r * 0.34, r * 0.24, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,200,0.5)';
  ctx.fill();
}

// ── Human player figure ───────────────────────────────────────────────────────
//
// Umpire view: world-Y → screen-X. Players appear in PROFILE:
//   player faces RIGHT (+Y = toward AI),  AI faces LEFT (-Y = toward player).

function drawHumanPlayer(
  ctx: Ctx,
  wx: number, wy: number,
  shirtColor: string,
  isAI: boolean,
  swingT: number,
) {
  const facing = isAI ? -1 : 1;

  // Perspective scale
  const pGround = project3D(wx, wy, 0);
  const s       = FOCAL / Math.max(1, pGround.depth);

  // ── Shadow ──────────────────────────────────────────────────────────────────
  ctx.beginPath();
  ctx.ellipse(pGround.x, pGround.y, Math.max(8, s*0.44), Math.max(3, s*0.16), 0, 0, Math.PI*2);
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fill();

  // ── Legs ────────────────────────────────────────────────────────────────────
  const pHip = project3D(wx, wy, 0.92);
  const pKnF = project3D(wx, wy + facing*0.14, 0.46);
  const pKnB = project3D(wx, wy - facing*0.08, 0.46);
  const pFtF = project3D(wx, wy + facing*0.20, 0.03);
  const pFtB = project3D(wx, wy - facing*0.12, 0.03);

  const legW = Math.max(3.5, s * 0.125);

  // Back leg (darker)
  ctx.strokeStyle = '#111';
  ctx.lineWidth   = legW;
  ctx.lineCap     = 'round';
  ctx.beginPath(); ctx.moveTo(pFtB.x, pFtB.y); ctx.lineTo(pKnB.x, pKnB.y); ctx.lineTo(pHip.x, pHip.y); ctx.stroke();

  // Front leg
  ctx.strokeStyle = '#1e1e1e';
  ctx.beginPath(); ctx.moveTo(pFtF.x, pFtF.y); ctx.lineTo(pKnF.x, pKnF.y); ctx.lineTo(pHip.x, pHip.y); ctx.stroke();

  // Shoes
  const shr = Math.max(4.5, s * 0.135);
  [pFtF, pFtB].forEach(p => {
    ctx.beginPath();
    ctx.ellipse(p.x, p.y, shr * 1.5, shr * 0.6, 0, 0, Math.PI*2);
    ctx.fillStyle = '#e8e8e8';
    ctx.fill();
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  });

  // ── Shorts ──────────────────────────────────────────────────────────────────
  fillPoly3D(ctx, [
    [wx, wy-0.26, 0.82], [wx, wy+0.26, 0.82],
    [wx, wy+0.22, 0.96], [wx, wy-0.22, 0.96],
  ], isAI ? '#222230' : '#f0f0f0');

  // ── Shirt (wider at shoulders) ───────────────────────────────────────────────
  fillPoly3D(ctx, [
    [wx, wy-0.22, 0.96], [wx, wy+0.22, 0.96],
    [wx, wy+0.28, 1.42], [wx, wy-0.28, 1.42],
  ], shirtColor);

  // Collar
  fillPoly3D(ctx, [
    [wx, wy-0.07, 1.39], [wx, wy+0.07, 1.39],
    [wx, wy+0.07, 1.45], [wx, wy-0.07, 1.45],
  ], '#f8f8f8');

  // ── Head ────────────────────────────────────────────────────────────────────
  const pNeck = project3D(wx, wy, 1.46);
  const pHead = project3D(wx, wy, 1.80);
  const hr    = Math.max(7, s * 0.22);

  // Neck
  ctx.strokeStyle = '#d49060';
  ctx.lineWidth   = Math.max(2.5, s * 0.09);
  ctx.beginPath(); ctx.moveTo(pNeck.x, pNeck.y); ctx.lineTo(pHead.x, pHead.y + hr * 0.5); ctx.stroke();

  // Skin
  ctx.beginPath();
  ctx.arc(pHead.x, pHead.y, hr, 0, Math.PI*2);
  ctx.fillStyle = '#e8a86a';
  ctx.fill();

  // Hair
  ctx.save();
  ctx.beginPath();
  ctx.arc(pHead.x, pHead.y, hr * 1.02, Math.PI, 0);
  ctx.closePath();
  ctx.fillStyle = isAI ? '#2e1a08' : '#141414';
  ctx.fill();
  ctx.restore();

  // Eye
  ctx.beginPath();
  ctx.arc(pHead.x + facing * hr * 0.32, pHead.y + hr * 0.14, Math.max(1.2, hr * 0.15), 0, Math.PI*2);
  ctx.fillStyle = '#0a0a0a';
  ctx.fill();

  // ── Racket arm ───────────────────────────────────────────────────────────────
  let armWY: number, armWZ: number;
  if (swingT <= 0) {
    armWY = wy + facing * 0.46;
    armWZ = 1.26;
  } else {
    const t = Math.min(1, swingT);
    armWY = wy + facing * lerpN(-0.42, 0.80, t);
    armWZ = t < 0.5
      ? lerpN(1.06, 1.26, t * 2)
      : lerpN(1.26, 1.56, (t - 0.5) * 2);
  }

  const pShoulder = project3D(wx, wy + facing * 0.24, 1.40);
  const pElbow    = project3D(wx + 0.14, armWY * 0.55 + (wy + facing*0.24) * 0.45, (armWZ + 1.40) * 0.5);
  const pHand     = project3D(wx + 0.22, armWY, armWZ);
  const rWY       = armWY + facing * 0.48;
  const pRacket   = project3D(wx + 0.22, rWY, armWZ);

  // Upper arm (shirt-colored sleeve)
  ctx.strokeStyle = shirtColor;
  ctx.lineWidth   = Math.max(3, s * 0.12);
  ctx.lineCap     = 'round';
  ctx.beginPath(); ctx.moveTo(pShoulder.x, pShoulder.y); ctx.lineTo(pElbow.x, pElbow.y); ctx.stroke();

  // Forearm (skin)
  ctx.strokeStyle = '#c87848';
  ctx.lineWidth   = Math.max(2.5, s * 0.1);
  ctx.beginPath(); ctx.moveTo(pElbow.x, pElbow.y); ctx.lineTo(pHand.x, pHand.y); ctx.stroke();

  // Grip / handle
  ctx.strokeStyle = '#4a2c0a';
  ctx.lineWidth   = Math.max(3, s * 0.11);
  ctx.beginPath(); ctx.moveTo(pHand.x, pHand.y); ctx.lineTo(pRacket.x, pRacket.y); ctx.stroke();

  // Racket frame
  const rw = Math.max(7, s * 0.32);
  const rh = Math.max(9, s * 0.38);
  ctx.beginPath();
  ctx.ellipse(pRacket.x, pRacket.y, rw, rh, 0, 0, Math.PI*2);
  ctx.strokeStyle = '#7088a0';
  ctx.lineWidth   = Math.max(2.5, s * 0.1);
  ctx.stroke();

  // String grid (clipped inside frame)
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(pRacket.x, pRacket.y, rw, rh, 0, 0, Math.PI*2);
  ctx.clip();
  ctx.strokeStyle = 'rgba(195,215,235,0.65)';
  ctx.lineWidth = 0.9;
  for (let g = -4; g <= 4; g++) {
    const fx = g / 4;
    ctx.beginPath();
    ctx.moveTo(pRacket.x + fx * rw, pRacket.y - rh);
    ctx.lineTo(pRacket.x + fx * rw, pRacket.y + rh);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pRacket.x - rw, pRacket.y + fx * rh);
    ctx.lineTo(pRacket.x + rw, pRacket.y + fx * rh);
    ctx.stroke();
  }
  ctx.restore();
}

// ── Landing pulse ─────────────────────────────────────────────────────────────

function drawLandingPulse(ctx: Ctx, wx: number, wy: number, tick: number) {
  const proj  = project3D(wx, wy, 0);
  const base  = Math.max(6, (FOCAL * 0.17) / proj.depth);
  const pulse = base + base * 0.65 * Math.sin(tick * 0.09);
  const alpha = 0.3 + 0.35 * Math.sin(tick * 0.09);

  // Outer ring
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, pulse, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(255,240,40,${alpha})`;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Second inner ring
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, pulse * 0.6, 0, Math.PI * 2);
  ctx.strokeStyle = `rgba(255,230,60,${alpha * 0.6})`;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Centre dot
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, base * 0.32, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(255,245,40,0.7)`;
  ctx.fill();
}

// ── Net shadow ────────────────────────────────────────────────────────────────

function drawNetShadow(ctx: Ctx) {
  const W    = COURT.widthM;
  const netY = COURT.netYM;
  ctx.strokeStyle = 'rgba(0,0,0,0.18)';
  ctx.lineWidth   = 3.5;
  line3D(ctx, 0, netY + 0.05, 0, W, netY + 0.05, 0);
}

// ── Main component ────────────────────────────────────────────────────────────

export default function TennisCourt({ state, handleCanvasClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tickRef   = useRef(0);
  const rafRef    = useRef<number>(0);
  const swingRef  = useRef<{ who: 'player' | 'ai'; startProg: number } | null>(null);

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    tickRef.current++;
    const tick = tickRef.current;

    ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

    // ── Scene ──────────────────────────────────────────────────────────────
    drawBackground(ctx);
    drawCourt(ctx);
    drawNetShadow(ctx);
    drawNet(ctx);

    // ── Ball world position ────────────────────────────────────────────────
    const shot    = state.currentShot;
    const prog    = state.animationProgress;
    const inFlight = shot && (state.phase === 'ai_hitting' || state.phase === 'player_hitting');

    let ballWX = state.ballPosition.x;
    let ballWY = state.ballPosition.y;
    let ballWZ = 0;

    if (inFlight && shot) {
      ballWX = lerpN(shot.origin.x, shot.landing.x, prog);
      ballWY = lerpN(shot.origin.y, shot.landing.y, prog);
      ballWZ = getBallZ(prog, shot);
    }

    // ── Trajectory trail ───────────────────────────────────────────────────
    if (inFlight && shot) drawTrail(ctx, shot, prog);

    // ── Swing tracking ─────────────────────────────────────────────────────
    if ((state.phase === 'player_hitting' || state.phase === 'ai_hitting') &&
        prog < 0.05 && !swingRef.current) {
      swingRef.current = {
        who: state.phase === 'player_hitting' ? 'player' : 'ai',
        startProg: prog,
      };
    }
    if (state.phase === 'awaiting_input' || state.phase === 'point_over') {
      swingRef.current = null;
    }

    let playerSwing = -1;
    let aiSwing     = -1;
    if (swingRef.current) {
      const elapsed = prog - swingRef.current.startProg;
      const st = Math.min(1, elapsed / 0.30);
      if (swingRef.current.who === 'player') playerSwing = st;
      else                                   aiSwing     = st;
    }

    // ── Animated player positions ──────────────────────────────────────────
    let visualPlayerPos = state.playerPos;
    let visualAiPos     = state.aiPos;

    if (inFlight && shot) {
      if (state.phase === 'ai_hitting') {
        // Player runs toward ball's landing; AI stands still (just hit)
        visualPlayerPos = computeRunningPos(
          state.playerStartPos, shot.landing, PLAYER_MAX_SPEED_MS, shot.travelTime, prog,
        );
      } else if (state.phase === 'player_hitting') {
        // AI runs toward ball's landing; player stands still (just hit)
        visualAiPos = computeRunningPos(
          state.aiStartPos, shot.landing, AI_MAX_SPEED_MS, shot.travelTime, prog,
        );
      }
    }

    // ── Players ────────────────────────────────────────────────────────────
    drawHumanPlayer(ctx, visualAiPos.x,     visualAiPos.y,     '#cc3838', true,  aiSwing);
    drawHumanPlayer(ctx, visualPlayerPos.x, visualPlayerPos.y, '#2870d8', false, playerSwing);

    // ── Ball ───────────────────────────────────────────────────────────────
    drawBallShadow(ctx, ballWX, ballWY, ballWZ);
    drawBall(ctx, ballWX, ballWY, ballWZ);

    // ── Landing indicator ──────────────────────────────────────────────────
    if (state.phase === 'awaiting_input' && shot) {
      drawLandingPulse(ctx, shot.landing.x, shot.landing.y, tick);
    }

    // ── UI overlays ────────────────────────────────────────────────────────
    if (state.phase === 'awaiting_input') {
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.65)';
      ctx.font = 'bold 13px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText('Click the right half to aim your return', CANVAS_W / 2, 10);
      ctx.restore();
    }

    if (state.phase === 'point_over') {
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
      const isAiWin  = state.pointResult === 'ai_wins';
      const reason   = state.lastValidation?.reason;
      const line1    = isAiWin
        ? (reason === 'out_of_bounds' ? 'OUT' : reason === 'net_fault' ? 'NET FAULT' : reason === 'impossible_angle' ? 'BAD ANGLE' : 'UNREACHABLE — AI wins')
        : 'WINNER!';
      const line2    = isAiWin ? 'AI wins the point' : 'Player wins the point';
      ctx.textAlign  = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = 'bold 32px sans-serif';
      ctx.fillStyle = 'rgba(0,0,0,0.8)';
      ctx.fillText(line1, CANVAS_W/2 + 2, CANVAS_H/2 - 18 + 2);
      ctx.fillStyle = isAiWin ? '#ff5050' : '#40ff70';
      ctx.fillText(line1, CANVAS_W/2, CANVAS_H/2 - 18);
      ctx.font = '18px sans-serif';
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.fillText(line2, CANVAS_W/2, CANVAS_H/2 + 18);
      ctx.restore();
    }

    rafRef.current = requestAnimationFrame(render);
  }, [state]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(rafRef.current);
  }, [render]);

  function onClick(e: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect   = canvas.getBoundingClientRect();
    const scaleX = CANVAS_W / rect.width;
    const scaleY = CANVAS_H / rect.height;
    handleCanvasClick({
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top)  * scaleY,
    });
  }

  return (
    <canvas
      ref={canvasRef}
      width={CANVAS_W}
      height={CANVAS_H}
      onClick={onClick}
      style={{
        cursor: state.phase === 'awaiting_input' ? 'crosshair' : 'default',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '12px',
        display: 'block',
        width: '100%',
        maxWidth: '100%',
        boxShadow: '0 4px 24px rgba(0,0,0,0.5)',
      }}
    />
  );
}
