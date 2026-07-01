from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Surface = Literal["hard", "clay", "grass", "indoor", "unknown"]
Server = Literal["ai", "player"]
ShotPhase = Literal["serve", "return", "rally", "approach", "volley", "overhead"]
StrokeType = Literal[
    "serve",
    "forehand",
    "backhand",
    "slice_backhand",
    "volley_forehand",
    "volley_backhand",
    "overhead",
    "lob",
    "drop_shot",
    "unknown",
]
ShotDirection = Literal[
    "crosscourt",
    "down_the_line",
    "middle",
    "inside_out",
    "inside_in",
    "wide",
    "body",
    "tee",
    "unknown",
]
ShotDepth = Literal["short", "medium", "deep", "unknown"]
ShotPace = Literal["slow", "medium", "fast", "unknown"]
ShotSpin = Literal["flat", "topspin", "slice", "kick", "unknown"]
ShotIntent = Literal["neutral", "attack", "defend", "reset"]
ShotOutcome = Literal["in_play", "winner", "forced_error", "unforced_error", "net", "long", "wide"]


@dataclass(frozen=True)
class RallyShotToken:
    phase: ShotPhase
    stroke: StrokeType
    direction: ShotDirection
    depth: ShotDepth
    spin: ShotSpin
    pace: ShotPace
    intent: ShotIntent = "neutral"
    outcome: ShotOutcome = "in_play"

    def action_key(self) -> str:
        return "|".join(
            [
                self.phase,
                self.stroke,
                self.direction,
                self.depth,
                self.spin,
                self.pace,
                self.intent,
                self.outcome,
            ]
        )

    @classmethod
    def from_key(cls, key: str) -> "RallyShotToken":
        parts = key.split("|")
        if len(parts) != 8:
            raise ValueError(f"Expected 8 action fields, got {len(parts)}: {key}")
        return cls(*parts)  # type: ignore[arg-type]


@dataclass(frozen=True)
class MatchContext:
    surface: Surface = "unknown"
    point_score: str = "unknown"
    server: Server = "ai"


@dataclass(frozen=True)
class RallyState:
    match: MatchContext
    rally_count: int
    previous_actions: list[RallyShotToken] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "match": asdict(self.match),
            "rally_count": self.rally_count,
            "previous_actions": [asdict(action) for action in self.previous_actions],
        }


@dataclass(frozen=True)
class TrainingExample:
    match_id: str
    state: RallyState
    target: RallyShotToken


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 0.9


@dataclass(frozen=True)
class ActionMask:
    pace: list[ShotPace] = field(default_factory=list)
    direction: list[ShotDirection] = field(default_factory=list)
    depth: list[ShotDepth] = field(default_factory=list)
    stroke: list[StrokeType] = field(default_factory=list)


@dataclass(frozen=True)
class OpponentAction:
    action: RallyShotToken
    confidence: float
    top_k: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self.action)
        payload["confidence"] = self.confidence
        payload["topK"] = [{"action": key, "probability": prob} for key, prob in self.top_k]
        return payload

