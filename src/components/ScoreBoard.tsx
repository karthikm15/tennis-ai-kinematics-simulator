import { TennisScore } from '../types';

interface Props {
  tennisScore: TennisScore;
  rallyCount:  number;
  servingPlayer: 'player' | 'ai';
}

// ── Tennis score labels ───────────────────────────────────────────────────────

function getPointLabel(p: number, opp: number, isTiebreak: boolean): string {
  if (isTiebreak) return String(p);
  if (p >= 3 && opp >= 3) {
    if (p === opp) return 'Deuce';
    return p > opp ? 'Ad' : '–';
  }
  return ['0', '15', '30', '40'][Math.min(p, 3)];
}

export default function ScoreBoard({ tennisScore, rallyCount, servingPlayer }: Props) {
  const { pointsInGame: pts, games, isTiebreak } = tennisScore;
  const playerLabel = getPointLabel(pts.player, pts.ai, isTiebreak);
  const aiLabel     = getPointLabel(pts.ai, pts.player, isTiebreak);
  const isDeuce     = !isTiebreak && pts.player >= 3 && pts.ai >= 3 && pts.player === pts.ai;

  return (
    <div style={{
      background: '#1a2535',
      borderRadius: 12,
      padding: '16px',
      border: '1px solid rgba(255,255,255,0.06)',
    }}>
      <p style={{ margin: '0 0 12px', color: '#64748b', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        Score {isTiebreak && <span style={{ color: '#f59e0b', marginLeft: 6 }}>Tiebreak</span>}
      </p>

      {/* Games row */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <GamesCard label="You" games={games.player} color="#2870d8" isServing={servingPlayer === 'player'} />
        <GamesCard label="AI"  games={games.ai}     color="#cc3838" isServing={servingPlayer === 'ai'} />
      </div>

      {/* Points row */}
      {isDeuce ? (
        <div style={{
          textAlign: 'center',
          padding: '8px 0',
          color: '#f59e0b',
          fontSize: 15,
          fontWeight: 700,
          letterSpacing: '0.05em',
          background: 'rgba(245,158,11,0.08)',
          borderRadius: 8,
          marginBottom: 10,
        }}>
          DEUCE
        </div>
      ) : (
        <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
          <PointCard label="You" point={playerLabel} color="#2870d8" />
          <PointCard label="AI"  point={aiLabel}     color="#cc3838" />
        </div>
      )}

      {/* Rally length */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        padding: '7px 10px',
        background: 'rgba(255,255,255,0.04)',
        borderRadius: 8,
      }}>
        <span style={{ color: '#475569', fontSize: 12 }}>Rally</span>
        <span style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: 13, fontWeight: 600 }}>
          {rallyCount} shots
        </span>
      </div>
    </div>
  );
}

function GamesCard({
  label, games, color, isServing,
}: { label: string; games: number; color: string; isServing: boolean }) {
  return (
    <div style={{
      flex: 1,
      background: 'rgba(255,255,255,0.04)',
      borderRadius: 8,
      padding: '10px 12px',
      borderBottom: `2px solid ${color}`,
      position: 'relative',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4 }}>
        {isServing && (
          <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#b8e020', flexShrink: 0 }} />
        )}
        <div style={{ color: '#475569', fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {label}
        </div>
      </div>
      <div style={{ color, fontSize: 28, fontWeight: 700, lineHeight: 1 }}>
        {games}
      </div>
      <div style={{ color: '#334155', fontSize: 10, marginTop: 2, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        games
      </div>
    </div>
  );
}

function PointCard({ label, point, color }: { label: string; point: string; color: string }) {
  const isAd = point === 'Ad';
  return (
    <div style={{
      flex: 1,
      background: isAd ? `${color}18` : 'rgba(255,255,255,0.03)',
      borderRadius: 8,
      padding: '8px 12px',
      border: isAd ? `1px solid ${color}44` : '1px solid transparent',
    }}>
      <div style={{ color: '#334155', fontSize: 10, fontWeight: 600, marginBottom: 3, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </div>
      <div style={{ color: isAd ? color : '#94a3b8', fontSize: 20, fontWeight: 700, lineHeight: 1 }}>
        {point}
      </div>
    </div>
  );
}
