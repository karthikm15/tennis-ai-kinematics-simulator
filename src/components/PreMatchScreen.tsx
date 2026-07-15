import { Difficulty, PlayStyle } from '../types';

interface Props {
  difficulty: Difficulty;
  style: PlayStyle;
  onDifficultyChange: (d: Difficulty) => void;
  onStyleChange: (s: PlayStyle) => void;
  onStart: () => void;
}

const DIFFICULTIES: { value: Difficulty; label: string; desc: string }[] = [
  { value: 'easy',   label: 'Easy',   desc: 'Slow balls, aims toward you' },
  { value: 'medium', label: 'Medium', desc: 'Balanced — current default' },
  { value: 'hard',   label: 'Hard',   desc: 'Fast, accurate placement' },
  { value: 'expert', label: 'Expert', desc: 'Full RL output, no mercy' },
];

const STYLES: { value: PlayStyle; label: string; desc: string; icon: string }[] = [
  { value: 'aggressive', label: 'Aggressive', desc: 'Goes for corners, fast shots', icon: '⚡' },
  { value: 'balanced',   label: 'Balanced',   desc: 'Mix of power and consistency', icon: '⚖️' },
  { value: 'consistent', label: 'Consistent', desc: 'Deep baseline, keeps ball in play', icon: '🔁' },
];

export default function PreMatchScreen({ difficulty, style, onDifficultyChange, onStyleChange, onStart }: Props) {
  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f1923',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '40px 24px',
      gap: '40px',
    }}>
      {/* Header */}
      <div style={{ textAlign: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginBottom: 12 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#4ade80', boxShadow: '0 0 8px #4ade80' }} />
          <span style={{ color: '#94a3b8', fontSize: 12, fontWeight: 500, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            Tennis AI · Kinematics Simulator
          </span>
        </div>
        <h1 style={{ margin: 0, color: '#f1f5f9', fontSize: 36, fontWeight: 800, letterSpacing: '-0.02em' }}>
          Set Up Match
        </h1>
        <p style={{ margin: '10px 0 0', color: '#475569', fontSize: 15 }}>
          One set · First to 6 games wins
        </p>
      </div>

      {/* Config card */}
      <div style={{
        background: '#1a2535',
        borderRadius: 16,
        padding: '32px',
        border: '1px solid rgba(255,255,255,0.06)',
        width: '100%',
        maxWidth: 520,
        display: 'flex',
        flexDirection: 'column',
        gap: 28,
      }}>

        {/* Difficulty */}
        <div>
          <SectionLabel>Difficulty</SectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {DIFFICULTIES.map(d => (
              <PillButton
                key={d.value}
                label={d.label}
                desc={d.desc}
                active={difficulty === d.value}
                color="#2870d8"
                onClick={() => onDifficultyChange(d.value)}
              />
            ))}
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: 1, background: 'rgba(255,255,255,0.06)' }} />

        {/* Playing style */}
        <div>
          <SectionLabel>AI Playing Style</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {STYLES.map(s => (
              <StyleButton
                key={s.value}
                icon={s.icon}
                label={s.label}
                desc={s.desc}
                active={style === s.value}
                onClick={() => onStyleChange(s.value)}
              />
            ))}
          </div>
        </div>

        {/* Start button */}
        <button
          onClick={onStart}
          style={{
            marginTop: 4,
            padding: '16px',
            background: '#2870d8',
            color: '#fff',
            border: 'none',
            borderRadius: 10,
            fontSize: 16,
            fontWeight: 700,
            letterSpacing: '0.04em',
            cursor: 'pointer',
            transition: 'opacity 0.15s',
          }}
          onMouseEnter={e => (e.currentTarget.style.opacity = '0.85')}
          onMouseLeave={e => (e.currentTarget.style.opacity = '1')}
        >
          Start Match
        </button>
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p style={{ margin: '0 0 12px', color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
      {children}
    </p>
  );
}

function PillButton({ label, desc, active, color, onClick }: {
  label: string; desc: string; active: boolean; color: string; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '12px 14px',
        background: active ? `${color}22` : 'rgba(255,255,255,0.03)',
        border: active ? `1.5px solid ${color}` : '1.5px solid rgba(255,255,255,0.08)',
        borderRadius: 8,
        cursor: 'pointer',
        textAlign: 'left',
        transition: 'all 0.15s',
      }}
    >
      <div style={{ color: active ? color : '#94a3b8', fontSize: 14, fontWeight: 700, marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ color: '#475569', fontSize: 11, lineHeight: 1.4 }}>
        {desc}
      </div>
    </button>
  );
}

function StyleButton({ icon, label, desc, active, onClick }: {
  icon: string; label: string; desc: string; active: boolean; onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '12px 14px',
        background: active ? 'rgba(204,56,56,0.12)' : 'rgba(255,255,255,0.03)',
        border: active ? '1.5px solid #cc3838' : '1.5px solid rgba(255,255,255,0.08)',
        borderRadius: 8,
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        transition: 'all 0.15s',
      }}
    >
      <span style={{ fontSize: 20 }}>{icon}</span>
      <div style={{ textAlign: 'left' }}>
        <div style={{ color: active ? '#cc3838' : '#94a3b8', fontSize: 14, fontWeight: 700, marginBottom: 2 }}>
          {label}
        </div>
        <div style={{ color: '#475569', fontSize: 11 }}>
          {desc}
        </div>
      </div>
      {active && (
        <div style={{ marginLeft: 'auto', width: 8, height: 8, borderRadius: '50%', background: '#cc3838' }} />
      )}
    </button>
  );
}
