interface Props {
  score: { player: number; ai: number };
  rallyCount: number;
}

export default function ScoreBoard({ score, rallyCount }: Props) {
  return (
    <div style={{
      background: '#1a2535',
      borderRadius: 12,
      padding: '16px',
      border: '1px solid rgba(255,255,255,0.06)',
    }}>
      <p style={{ margin: '0 0 14px', color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        Score
      </p>

      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <ScoreCard label="You" score={score.player} color="#2870d8" />
        <ScoreCard label="AI" score={score.ai} color="#cc3838" />
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '7px 10px',
        background: 'rgba(255,255,255,0.04)',
        borderRadius: 8,
      }}>
        <span style={{ color: '#475569', fontSize: 12 }}>Rally length</span>
        <span style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: 13, fontWeight: 600 }}>
          {rallyCount}
        </span>
      </div>
    </div>
  );
}

function ScoreCard({ label, score, color }: { label: string; score: number; color: string }) {
  return (
    <div style={{
      flex: 1,
      background: 'rgba(255,255,255,0.04)',
      borderRadius: 8,
      padding: '10px 12px',
      borderBottom: `2px solid ${color}`,
    }}>
      <div style={{ color: '#475569', fontSize: 11, fontWeight: 600, marginBottom: 4, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ color, fontSize: 28, fontWeight: 700, lineHeight: 1 }}>
        {score}
      </div>
    </div>
  );
}
