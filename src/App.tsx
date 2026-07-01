import TennisCourt from './components/TennisCourt';
import ShotInfo from './components/ShotInfo';
import ScoreBoard from './components/ScoreBoard';
import PlayerProfile from './components/PlayerProfile';
import { useGameState } from './hooks/useGameState';

export default function App() {
  const { state, handleCanvasClick, serveError } = useGameState();

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
          {/* Court labels */}
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
          <ScoreBoard score={state.score} rallyCount={state.rallyCount} />
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
