import { GamePhase, Shot, ShotValidation, ValidationReason } from '../types';

interface Props {
  phase: GamePhase;
  currentShot: Shot | null;
  lastValidation: ShotValidation | null;
}

const REASON_SHORT: Record<ValidationReason, string> = {
  out_of_bounds:    'Out of bounds',
  unreachable:      'Ball unreachable',
  impossible_angle: 'Impossible angle',
  net_fault:        'Net fault',
};

const PHASE_LABEL: Record<GamePhase, { text: string; color: string }> = {
  ai_hitting:     { text: 'AI hitting',    color: '#cc3838' },
  awaiting_input: { text: 'Your turn',     color: '#4ade80' },
  player_hitting: { text: 'Ball in flight', color: '#2870d8' },
  point_over:     { text: 'Point over',    color: '#f59e0b' },
};

function ms2kph(ms: number) { return (ms * 3.6).toFixed(0) + ' km/h'; }
function rad2deg(r: number) { return (r * 180 / Math.PI).toFixed(1) + '°'; }

export default function ShotInfo({ phase, currentShot, lastValidation }: Props) {
  const phaseInfo = PHASE_LABEL[phase];

  return (
    <div style={{
      background: '#1a2535',
      borderRadius: 12,
      padding: '16px',
      border: '1px solid rgba(255,255,255,0.06)',
      display: 'flex',
      flexDirection: 'column',
      gap: 12,
    }}>
      {/* Phase badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{ margin: 0, color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Shot Physics
        </p>
        <span style={{
          background: `${phaseInfo.color}22`,
          color: phaseInfo.color,
          fontSize: 11,
          fontWeight: 600,
          padding: '3px 8px',
          borderRadius: 20,
          letterSpacing: '0.05em',
        }}>
          {phaseInfo.text}
        </span>
      </div>

      {/* Incoming shot stats */}
      {currentShot && (
        <div style={{
          background: 'rgba(255,255,255,0.04)',
          borderRadius: 8,
          padding: '10px 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}>
          <p style={{ margin: '0 0 4px', color: '#475569', fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Incoming
          </p>
          <StatRow label="Speed"  value={ms2kph(currentShot.speed)} />
          <StatRow label="Flight" value={currentShot.travelTime.toFixed(2) + ' s'} />
          <StatRow label="Spin"   value={currentShot.spinType} />
        </div>
      )}

      {/* Return validation */}
      {lastValidation && (
        <div style={{
          background: 'rgba(255,255,255,0.04)',
          borderRadius: 8,
          padding: '10px 12px',
          display: 'flex',
          flexDirection: 'column',
          gap: 6,
        }}>
          <p style={{ margin: '0 0 4px', color: '#475569', fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Your Return
          </p>
          {lastValidation.deflectionAngle !== undefined && (
            <StatRow label="Deflection" value={rad2deg(lastValidation.deflectionAngle)} />
          )}
          {lastValidation.returnSpeed !== undefined && (
            <StatRow label="Speed" value={ms2kph(lastValidation.returnSpeed)} />
          )}
          <div style={{
            marginTop: 2,
            padding: '5px 8px',
            borderRadius: 6,
            background: lastValidation.valid ? 'rgba(74,222,128,0.1)' : 'rgba(248,113,113,0.1)',
            borderLeft: `3px solid ${lastValidation.valid ? '#4ade80' : '#f87171'}`,
            color: lastValidation.valid ? '#4ade80' : '#f87171',
            fontSize: 12,
            fontWeight: 600,
          }}>
            {lastValidation.valid
              ? '✓ Valid return'
              : `✗ ${lastValidation.reason ? REASON_SHORT[lastValidation.reason] : 'Invalid'}`}
          </div>
        </div>
      )}

      {/* Prompt when waiting */}
      {phase === 'awaiting_input' && !lastValidation && (
        <div style={{
          padding: '8px 10px',
          borderRadius: 8,
          background: 'rgba(74,222,128,0.08)',
          border: '1px solid rgba(74,222,128,0.2)',
          color: '#86efac',
          fontSize: 12,
          lineHeight: 1.5,
        }}>
          Click the <strong>right half</strong> of the court to return the ball.
        </div>
      )}
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ color: '#475569', fontSize: 12 }}>{label}</span>
      <span style={{ color: '#cbd5e1', fontSize: 13, fontWeight: 500 }}>{value}</span>
    </div>
  );
}
