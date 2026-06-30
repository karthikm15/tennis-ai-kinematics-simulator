from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from typing import Iterable

from tennis_policy.schemas import ActionMask, OpponentAction, RallyShotToken, RallyState, TrainingExample


class FrequencyPolicy:
    def __init__(self, counts: Counter[str] | None = None) -> None:
        self.counts: Counter[str] = counts or Counter()

    def fit(self, examples: list[TrainingExample]) -> "FrequencyPolicy":
        return self.fit_iter(examples)

    def fit_iter(self, examples: Iterable[TrainingExample]) -> "FrequencyPolicy":
        self.counts = Counter(example.target.action_key() for example in examples)
        return self

    def predict_distribution(self, state: RallyState | None = None) -> dict[str, float]:
        return _normalize(self.counts)

    def sample_action(self, state: RallyState, mask: ActionMask | None = None, seed: int | None = None) -> OpponentAction:
        return _sample_from_distribution(self.predict_distribution(state), mask=mask, seed=seed)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"type": "frequency", "counts": dict(self.counts)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FrequencyPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Counter(payload["counts"]))


class MarkovPolicy(FrequencyPolicy):
    def __init__(self, order: int = 2, table: dict[tuple[str, ...], Counter[str]] | None = None) -> None:
        super().__init__()
        self.order = order
        self.table: dict[tuple[str, ...], Counter[str]] = table or defaultdict(Counter)

    def fit(self, examples: list[TrainingExample]) -> "MarkovPolicy":
        return self.fit_iter(examples)

    def fit_iter(self, examples: Iterable[TrainingExample]) -> "MarkovPolicy":
        self.counts = Counter()
        self.table = defaultdict(Counter)
        for example in examples:
            self.counts[example.target.action_key()] += 1
            history = tuple(action.action_key() for action in example.state.previous_actions[-self.order :])
            self.table[history][example.target.action_key()] += 1
        return self

    def predict_distribution(self, state: RallyState | None = None) -> dict[str, float]:
        if state:
            for order in range(self.order, 0, -1):
                key = tuple(action.action_key() for action in state.previous_actions[-order:])
                if key in self.table:
                    return _normalize(self.table[key])
        return super().predict_distribution(state)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        serial_table = {"\t".join(key): dict(counter) for key, counter in self.table.items()}
        payload = {"type": "markov", "order": self.order, "counts": dict(self.counts), "table": serial_table}
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MarkovPolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        table = defaultdict(Counter)
        for key, counter in payload["table"].items():
            table[tuple(part for part in key.split("\t") if part)] = Counter(counter)
        policy = cls(order=int(payload["order"]), table=table)
        policy.counts = Counter(payload["counts"])
        return policy


def load_baseline_policy(path: str | Path) -> FrequencyPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("type") == "markov":
        return MarkovPolicy.load(path)
    return FrequencyPolicy.load(path)


def negative_log_likelihood(policy: FrequencyPolicy, examples: list[TrainingExample]) -> float:
    total = 0.0
    count = 0
    for example in examples:
        distribution = policy.predict_distribution(example.state)
        probability = distribution.get(example.target.action_key(), 1e-12)
        total -= math.log(probability)
        count += 1
    return total / max(count, 1)


def top_k_accuracy(policy: FrequencyPolicy, examples: list[TrainingExample], k: int = 3) -> float:
    hits = 0
    count = 0
    for example in examples:
        ranked = sorted(policy.predict_distribution(example.state).items(), key=lambda item: item[1], reverse=True)
        if example.target.action_key() in {key for key, _ in ranked[:k]}:
            hits += 1
        count += 1
    return hits / max(count, 1)


def _normalize(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total <= 0:
        fallback = "rally|forehand|crosscourt|deep|topspin|medium|neutral|in_play"
        return {fallback: 1.0}
    return {key: value / total for key, value in counts.items()}


def _sample_from_distribution(
    distribution: dict[str, float], mask: ActionMask | None = None, seed: int | None = None
) -> OpponentAction:
    rng = random.Random(seed)
    filtered = _apply_mask(distribution, mask)
    keys = list(filtered)
    weights = [filtered[key] for key in keys]
    selected = rng.choices(keys, weights=weights, k=1)[0]
    ranked = sorted(filtered.items(), key=lambda item: item[1], reverse=True)[:5]
    return OpponentAction(action=RallyShotToken.from_key(selected), confidence=filtered[selected], top_k=ranked)


def _apply_mask(distribution: dict[str, float], mask: ActionMask | None) -> dict[str, float]:
    if not mask:
        return distribution

    kept: dict[str, float] = {}
    for key, probability in distribution.items():
        try:
            action = RallyShotToken.from_key(key)
        except ValueError:
            continue
        if action.pace in mask.pace:
            continue
        if action.direction in mask.direction:
            continue
        if action.depth in mask.depth:
            continue
        if action.stroke in mask.stroke:
            continue
        kept[key] = probability

    return kept or distribution
