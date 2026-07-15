import { TennisScore, MatchStats, Difficulty, PlayStyle } from '../types';

interface Props {
  tennisScore: TennisScore;
  matchStats:  MatchStats;
  difficulty:  Difficulty;
  style:       PlayStyle;
  onPlayAgain: () => void;
}

const DIFFICULTY_LABELS: Record<Difficulty, string> = {
  easy: 'Easy', medium: 'Medium', hard: 'Hard', expert: 'Expert',
};
const STYLE_LABELS: Record<PlayStyle, string> = {
  aggressive: 'Aggressive', balanced: 'Balanced', consistent: 'Consistent',
};

export default function MatchStatsScreen({ tennisScore, matchStats, difficulty, style, onPlayAgain }: Props) {
  const { games } = tennisScore;
  const playerWon = games.player > games.ai;

  const avgRally = matchStats.rallyCount > 0
    ? (matchStats.rallyTotal / matchStats.rallyCount).toFixed(1)
    : '0';

  const totalPts  = matchStats.totalPoints.player + matchStats.totalPoints.ai;
  const playerPct = totalPts > 0 ? Math.round((matchStats.totalPoints.player / totalPts) * 100) : 0;
  const aiPct     = 100 - playerPct;

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f1923',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '40px 24px',
      gap: '32px',
    }}>
      {/* Result banner */}
      <div style={{ textAlign: 'center' }}>
        <div style={{
          display: 'inline-block',
          padding: '6px 16px',
          background: playerWon ? 'rgba(74,222,128,0.1)' : 'rgba(204,56,56,0.1)',
          borderRadius: 20,
          color: playerWon ? '#4ade80' : '#f87171',
          fontSize: 12,
          fontWeight: 600,
          letterSpacing: '0.1em',
          textTransform: 'uppercase',
          marginBottom: 14,
        }}>
          {playerWon ? 'You Won' : 'AI Won'}
        </div>
        <h1 style={{ margin: 0, color: '#f1f5f9', fontSize: 40, fontWeight: 800, letterSpacing: '-0.02em' }}>
          <span style={{ color: '#2870d8' }}>{games.player}</span>
          <span style={{ color: '#475569', margin: '0 12px', fontWeight: 300 }}>–</span>
          <span style={{ color: '#cc3838' }}>{games.ai}</span>
        </h1>
        <p style={{ margin: '10px 0 0', color: '#475569', fontSize: 13 }}>
          {DIFFICULTY_LABELS[difficulty]} · {STYLE_LABELS[style]}
        </p>
      </div>

      {/* Stats card */}
      <div style={{
        background: '#1a2535',
        borderRadius: 16,
        padding: '28px',
        border: '1px solid rgba(255,255,255,0.06)',
        width: '100%',
        maxWidth: 480,
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
      }}>
        <p style={{ margin: 0, color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Match Stats
        </p>

        {/* Points bar */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ color: '#2870d8', fontSize: 13, fontWeight: 700 }}>
              {matchStats.totalPoints.player} pts
            </span>
            <span style={{ color: '#64748b', fontSize: 12 }}>Points Won</span>
            <span style={{ color: '#cc3838', fontSize: 13, fontWeight: 700 }}>
              {matchStats.totalPoints.ai} pts
            </span>
          </div>
          <div style={{ display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden', background: '#0f1923' }}>
            <div style={{ width: `${playerPct}%`, background: '#2870d8', transition: 'width 0.6s ease' }} />
            <div style={{ width: `${aiPct}%`, background: '#cc3838' }} />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <StatRow label="Total points played"  value={String(totalPts)} />
          <StatRow label="Longest rally"        value={`${matchStats.longestRally} shots`} />
          <StatRow label="Average rally length" value={`${avgRally} shots`} />
          <StatRow label="Total rallies"        value={String(matchStats.rallyCount)} />
        </div>
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, width: '100%', maxWidth: 480 }}>
        <button
          onClick={onPlayAgain}
          style={{
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
          Play Again
        </button>
      </div>
    </div>
  );
}

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '8px 12px',
      background: 'rgba(255,255,255,0.03)',
      borderRadius: 8,
    }}>
      <span style={{ color: '#64748b', fontSize: 13 }}>{label}</span>
      <span style={{ color: '#cbd5e1', fontSize: 14, fontWeight: 600 }}>{value}</span>
    </div>
  );
}
