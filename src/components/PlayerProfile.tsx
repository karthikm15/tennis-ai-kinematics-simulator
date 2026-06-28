import { ShotRecord } from '../types';
import { computeProfile } from '../engine/playerProfile';
import { COURT } from '../engine/court';

interface Props {
  history: ShotRecord[];
}

const NET = COURT.netYM;
const L   = COURT.lengthM;

// ── Heatmap ───────────────────────────────────────────────────────────────────

function Heatmap({ heatmap }: { heatmap: [[number,number,number],[number,number,number],[number,number,number]] }) {
  const max = Math.max(1, ...heatmap.flat());

  // heatmap[col][row]: col=x-zone (0=L,1=C,2=R), row=depth (0=short,1=mid,2=deep)
  // Display: rows top=deep, bottom=short; cols left=L, right=R

  const colLabels = ['L', 'C', 'R'];
  const rowLabels = ['Deep', 'Mid', 'Short']; // top → bottom on screen = deep → short

  return (
    <div>
      <p style={sectionLabel}>Target Heatmap</p>
      <div style={{ display: 'flex', gap: 6, alignItems: 'flex-start' }}>
        {/* Row labels */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, justifyContent: 'space-around', paddingTop: 2 }}>
          {rowLabels.map(r => (
            <div key={r} style={{ color: '#475569', fontSize: 10, height: 28, display: 'flex', alignItems: 'center' }}>
              {r}
            </div>
          ))}
        </div>

        <div style={{ flex: 1 }}>
          {/* Grid: iterate rows top→bottom (deep=2, mid=1, short=0) */}
          {[2, 1, 0].map(rowIdx => (
            <div key={rowIdx} style={{ display: 'flex', gap: 3, marginBottom: 3 }}>
              {[0, 1, 2].map(colIdx => {
                const count = heatmap[colIdx][rowIdx];
                const intensity = count / max;
                return (
                  <HeatCell key={colIdx} count={count} intensity={intensity} />
                );
              })}
            </div>
          ))}
          {/* Column labels */}
          <div style={{ display: 'flex', gap: 3, marginTop: 2 }}>
            {colLabels.map(c => (
              <div key={c} style={{ flex: 1, textAlign: 'center', color: '#475569', fontSize: 10 }}>{c}</div>
            ))}
          </div>
        </div>

        {/* Net + Baseline labels */}
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: 99, paddingBottom: 14 }}>
          <div style={{ color: '#475569', fontSize: 9, textAlign: 'right' }}>BASE</div>
          <div style={{ color: '#475569', fontSize: 9, textAlign: 'right' }}>NET</div>
        </div>
      </div>
    </div>
  );
}

function HeatCell({ count, intensity }: { count: number; intensity: number }) {
  const bg = intensity === 0
    ? 'rgba(255,255,255,0.04)'
    : `rgba(56,189,248,${0.12 + intensity * 0.78})`;

  return (
    <div style={{
      flex: 1,
      height: 28,
      borderRadius: 4,
      background: bg,
      border: '1px solid rgba(255,255,255,0.06)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: intensity > 0.3 ? '#fff' : '#475569',
      fontSize: 11,
      fontWeight: 600,
      transition: 'background 0.3s',
    }}>
      {count > 0 ? count : ''}
    </div>
  );
}

// ── Directional bars ──────────────────────────────────────────────────────────

function DirectionalTendencies({ directional }: { directional: ReturnType<typeof computeProfile>['directional'] }) {
  return (
    <div>
      <p style={sectionLabel}>Direction (by ball position)</p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {directional.map(row => {
          if (row.total === 0) return null;
          const crossPct = Math.round((row.crossCount / row.total) * 100);
          const samePct  = 100 - crossPct;
          return (
            <div key={row.label}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ color: '#64748b', fontSize: 11 }}>{row.label}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>
                  {crossPct}% cross · {samePct}% line
                </span>
              </div>
              <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.06)', overflow: 'hidden', display: 'flex' }}>
                <div style={{ width: `${crossPct}%`, background: '#38bdf8', borderRadius: 3 }} />
                <div style={{ width: `${samePct}%`, background: '#f59e0b', borderRadius: 3, marginLeft: 'auto' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
                <span style={{ fontSize: 9, color: '#38bdf8' }}>cross-court</span>
                <span style={{ fontSize: 9, color: '#f59e0b' }}>down-the-line</span>
              </div>
            </div>
          );
        })}
        {directional.every(r => r.total === 0) && (
          <div style={{ color: '#475569', fontSize: 12 }}>No data yet</div>
        )}
      </div>
    </div>
  );
}

// ── Stats row ─────────────────────────────────────────────────────────────────

function StatRow({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      <span style={{ color: '#475569', fontSize: 12, flex: 1 }}>{label}</span>
      <span style={{ color: '#cbd5e1', fontSize: 13, fontWeight: 500 }}>{value}</span>
      {sub && <span style={{ color: '#475569', fontSize: 11, marginLeft: 4 }}>{sub}</span>}
    </div>
  );
}

// ── Speed sparkline ───────────────────────────────────────────────────────────

function SpeedSparkline({ speeds }: { speeds: number[] }) {
  if (speeds.length < 2) return null;
  const max = Math.max(...speeds);
  const min = Math.min(...speeds);
  const range = max - min || 1;
  const W_PX = 180, H_PX = 24;

  const points = speeds.map((s, i) => {
    const x = (i / (speeds.length - 1)) * W_PX;
    const y = H_PX - ((s - min) / range) * H_PX;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div>
      <p style={{ ...sectionLabel, marginBottom: 4 }}>Return speed (last {speeds.length})</p>
      <svg width={W_PX} height={H_PX} style={{ overflow: 'visible' }}>
        <polyline points={points} fill="none" stroke="#38bdf8" strokeWidth={1.5} strokeLinejoin="round" />
        {speeds.map((s, i) => {
          const x = (i / (speeds.length - 1)) * W_PX;
          const y = H_PX - ((s - min) / range) * H_PX;
          return <circle key={i} cx={x} cy={y} r={2} fill="#38bdf8" />;
        })}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
        <span style={{ color: '#475569', fontSize: 9 }}>{(min*3.6).toFixed(0)} km/h</span>
        <span style={{ color: '#475569', fontSize: 9 }}>{(max*3.6).toFixed(0)} km/h</span>
      </div>
    </div>
  );
}

// ── Fault breakdown ───────────────────────────────────────────────────────────

function FaultBreakdown({ faults, total }: { faults: ReturnType<typeof computeProfile>['faults']; total: number }) {
  if (total === 0) return null;
  const rows: { label: string; key: keyof typeof faults; color: string }[] = [
    { label: 'Out',        key: 'out_of_bounds',    color: '#f87171' },
    { label: 'Net',        key: 'net_fault',         color: '#fb923c' },
    { label: 'Angle',      key: 'impossible_angle',  color: '#facc15' },
    { label: 'Unreachable',key: 'unreachable',        color: '#a78bfa' },
  ];
  return (
    <div>
      <p style={sectionLabel}>Fault breakdown</p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {rows.filter(r => faults[r.key] > 0).map(r => (
          <div key={r.key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: r.color, flexShrink: 0 }} />
            <span style={{ color: '#64748b', fontSize: 11, flex: 1 }}>{r.label}</span>
            <span style={{ color: r.color, fontSize: 11, fontWeight: 600 }}>{faults[r.key]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function PlayerProfile({ history }: Props) {
  const MIN_SHOTS = 1;

  const profile = computeProfile(history);

  return (
    <div style={{
      background: '#1a2535',
      borderRadius: 12,
      padding: '16px',
      border: '1px solid rgba(255,255,255,0.06)',
      display: 'flex',
      flexDirection: 'column',
      gap: 16,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{ margin: 0, color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Player Tendencies
        </p>
        <span style={{
          background: 'rgba(56,189,248,0.12)',
          color: '#38bdf8',
          fontSize: 11,
          fontWeight: 600,
          padding: '2px 8px',
          borderRadius: 20,
        }}>
          {profile.totalAttempts} shots
        </span>
      </div>

      {profile.totalAttempts < MIN_SHOTS ? (
        <div style={{ color: '#475569', fontSize: 12, textAlign: 'center', padding: '12px 0' }}>
          Play a shot to start tracking
        </div>
      ) : (
        <>
          <Heatmap heatmap={profile.heatmap} />

          <Divider />

          <DirectionalTendencies directional={profile.directional} />

          <Divider />

          {/* Summary stats */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <StatRow
              label="Fault rate"
              value={`${Math.round(profile.faultRate * 100)}%`}
              sub={`${profile.totalAttempts - profile.validCount}/${profile.totalAttempts}`}
            />
            <StatRow
              label="Avg depth"
              value={profile.avgDepth > 0.66 ? 'Deep' : profile.avgDepth > 0.33 ? 'Mid' : 'Short'}
              sub={`${Math.round(NET + profile.avgDepth * (L - NET))}m from net`}
            />
            {profile.avgReturnSpeed > 0 && (
              <StatRow
                label="Avg return"
                value={`${(profile.avgReturnSpeed * 3.6).toFixed(0)} km/h`}
              />
            )}
          </div>

          {profile.recentSpeeds.length >= 2 && (
            <>
              <Divider />
              <SpeedSparkline speeds={profile.recentSpeeds} />
            </>
          )}

          {Object.values(profile.faults).some(v => v > 0) && (
            <>
              <Divider />
              <FaultBreakdown faults={profile.faults} total={profile.totalAttempts - profile.validCount} />
            </>
          )}
        </>
      )}
    </div>
  );
}

// ── Shared styles ─────────────────────────────────────────────────────────────

const sectionLabel: React.CSSProperties = {
  margin: '0 0 8px',
  color: '#64748b',
  fontSize: 11,
  fontWeight: 600,
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
};

function Divider() {
  return <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }} />;
}
