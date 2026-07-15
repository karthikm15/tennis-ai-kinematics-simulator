import { useState, useEffect } from 'react';
import TennisCourt from './components/TennisCourt';
import ShotInfo from './components/ShotInfo';
import ScoreBoard from './components/ScoreBoard';
import EnergyBar from './components/EnergyBar';
import PlayerProfile from './components/PlayerProfile';
import PreMatchScreen from './components/PreMatchScreen';
import MatchStatsScreen from './components/MatchStatsScreen';
import { useGameState } from './hooks/useGameState';
import { Difficulty, PlayStyle } from './types';

type AppPhase = 'pre_match' | 'playing' | 'set_over';

export default function App() {
  const [appPhase, setAppPhase]     = useState<AppPhase>('pre_match');
  const [difficulty, setDifficulty] = useState<Difficulty>('medium');
  const [style, setStyle]           = useState<PlayStyle>('balanced');

  const { state, handleCanvasClick, serveError, resetMatch } = useGameState(difficulty, style);

  // When match ends, wait for the point-over animation then show stats
  useEffect(() => {
    if (!state.matchOver || appPhase !== 'playing') return;
    const id = setTimeout(() => setAppPhase('set_over'), 3000);
    return () => clearTimeout(id);
  }, [state.matchOver, appPhase]);

  if (appPhase === 'pre_match') {
    return (
      <PreMatchScreen
        difficulty={difficulty}
        style={style}
        onDifficultyChange={setDifficulty}
        onStyleChange={setStyle}
        onStart={() => { resetMatch(); setAppPhase('playing'); }}
      />
    );
  }

  if (appPhase === 'set_over') {
    return (
      <MatchStatsScreen
        tennisScore={state.tennisScore}
        matchStats={state.matchStats}
        difficulty={difficulty}
        style={style}
        onPlayAgain={() => setAppPhase('pre_match')}
      />
    );
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f1923',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '28px 24px',
      gap: '20px',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        alignSelf: 'stretch',
        maxWidth: 1080,
        width: '100%',
        margin: '0 auto',
      }}>
        <div style={{
          width: 8, height: 8,
          borderRadius: '50%',
          background: '#4ade80',
          boxShadow: '0 0 8px #4ade80',
        }} />
        <span style={{
          color: '#94a3b8',
          fontSize: 13,
          fontWeight: 500,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          Tennis AI · Kinematics Simulator
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <MatchBadge difficulty={difficulty} style={style} />
        </div>
      </div>

      {/* Main layout */}
      <div style={{
        display: 'flex',
        gap: '20px',
        alignItems: 'flex-start',
        width: '100%',
        maxWidth: 1080,
        margin: '0 auto',
      }}>
        {/* Court */}
        <div style={{ flex: '1 1 auto', minWidth: 0 }}>
          <TennisCourt state={state} handleCanvasClick={handleCanvasClick} serveError={serveError} />
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: 8,
            paddingLeft: 4,
            paddingRight: 4,
          }}>
            <span style={{ color: '#2870d8', fontSize: 12, fontWeight: 600, letterSpacing: '0.06em' }}>
              ← YOUR SIDE
            </span>
            <span style={{ color: '#cc3838', fontSize: 12, fontWeight: 600, letterSpacing: '0.06em' }}>
              AI SIDE →
            </span>
          </div>
        </div>

        {/* Side panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: 240, flexShrink: 0 }}>
          <ScoreBoard
            tennisScore={state.tennisScore}
            rallyCount={state.rallyCount}
            servingPlayer={state.servingPlayer}
          />
          <EnergyBar
            playerEnergy={state.playerEnergy}
            aiEnergy={state.aiEnergy}
            playerWinded={state.playerWinded}
            aiWinded={state.aiWinded}
            playerLastEffort={state.playerLastEffort}
            aiLastEffort={state.aiLastEffort}
          />
          <ShotInfo
            phase={state.phase}
            currentShot={state.currentShot}
            lastValidation={state.lastValidation}
          />
          <PlayerProfile history={state.shotHistory} />
          <Legend />
        </div>
      </div>
    </div>
  );
}

function MatchBadge({ difficulty, style }: { difficulty: Difficulty; style: PlayStyle }) {
  const diffColor: Record<Difficulty, string> = {
    easy: '#4ade80', medium: '#f59e0b', hard: '#f97316', expert: '#f87171',
  };
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      <span style={{
        background: `${diffColor[difficulty]}18`,
        color: diffColor[difficulty],
        fontSize: 11, fontWeight: 600,
        padding: '3px 8px', borderRadius: 20,
        letterSpacing: '0.05em', textTransform: 'capitalize',
      }}>
        {difficulty}
      </span>
      <span style={{
        background: 'rgba(148,163,184,0.1)',
        color: '#94a3b8',
        fontSize: 11, fontWeight: 600,
        padding: '3px 8px', borderRadius: 20,
        letterSpacing: '0.05em', textTransform: 'capitalize',
      }}>
        {style}
      </span>
    </div>
  );
}

function Legend() {
  return (
    <div style={{
      background: '#1a2535',
      borderRadius: 12,
      padding: '14px 16px',
      border: '1px solid rgba(255,255,255,0.06)',
    }}>
      <p style={{ margin: '0 0 10px', color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        How to Play
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <LegendRow dot="#2870d8" label="You (blue)" detail="Left side" />
        <LegendRow dot="#cc3838" label="AI (red)" detail="Right side" />
        <LegendRow dot="#b8e020" label="Ball" detail="Yellow-green" />
      </div>
      <div style={{
        marginTop: 12,
        paddingTop: 10,
        borderTop: '1px solid rgba(255,255,255,0.06)',
        color: '#475569',
        fontSize: 12,
        lineHeight: 1.6,
      }}>
        When the ball lands, click anywhere on the <span style={{ color: '#94a3b8' }}>right half</span> to return it.
        Big hits and long sprints burn <span style={{ color: '#94a3b8' }}>energy</span> — run out and you won't reach the next ball. Rally calmly to recover.
      </div>
    </div>
  );
}

function LegendRow({ dot, label, detail }: { dot: string; label: string; detail: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ width: 8, height: 8, borderRadius: '50%', background: dot, flexShrink: 0 }} />
      <span style={{ color: '#cbd5e1', fontSize: 13, flex: 1 }}>{label}</span>
      <span style={{ color: '#475569', fontSize: 12 }}>{detail}</span>
    </div>
  );
}
