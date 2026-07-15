import { effortLabel } from '../engine/energy';

interface Props {
  playerEnergy: number;
  aiEnergy: number;
  playerWinded: boolean;
  aiWinded: boolean;
  playerLastEffort: number | null;
  aiLastEffort: number | null;
}

function barColor(energy: number): string {
  if (energy > 60) return '#4ade80';
  if (energy > 30) return '#f59e0b';
  return '#f87171';
}

export default function EnergyBar({
  playerEnergy, aiEnergy, playerWinded, aiWinded, playerLastEffort, aiLastEffort,
}: Props) {
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
      <p style={{ margin: 0, color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        Energy
      </p>
      <EnergyRow label="You" color="#2870d8" energy={playerEnergy} winded={playerWinded} lastEffort={playerLastEffort} />
      <EnergyRow label="AI"  color="#cc3838" energy={aiEnergy}     winded={aiWinded}     lastEffort={aiLastEffort} />
    </div>
  );
}

function EnergyRow({
  label, color, energy, winded, lastEffort,
}: { label: string; color: string; energy: number; winded: boolean; lastEffort: number | null }) {
  const pct = Math.round(energy);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ color, fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {label}
        </span>
        {winded && (
          <span style={{
            background: 'rgba(248,113,113,0.12)',
            color: '#f87171',
            fontSize: 10, fontWeight: 700,
            padding: '1px 6px', borderRadius: 20,
            letterSpacing: '0.06em',
          }}>
            ⚡ WINDED
          </span>
        )}
        <span style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: 12, fontWeight: 600 }}>
          {pct}%
        </span>
      </div>
      <div style={{
        height: 8,
        borderRadius: 4,
        background: 'rgba(255,255,255,0.06)',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: `${pct}%`,
          borderRadius: 4,
          background: barColor(energy),
          transition: 'width 0.35s ease, background 0.35s ease',
        }} />
      </div>
      {lastEffort !== null && lastEffort > 0.5 && (
        <span style={{ color: '#475569', fontSize: 11 }}>
          Last shot: {effortLabel(lastEffort)} (−{lastEffort.toFixed(0)})
        </span>
      )}
    </div>
  );
}
