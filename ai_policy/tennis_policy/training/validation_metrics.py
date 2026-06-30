from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from tennis_policy.data.sequence_builder import iter_examples
from tennis_policy.models.baseline_markov import FrequencyPolicy
from tennis_policy.schemas import RallyShotToken, TrainingExample

FACTORS = ["phase", "stroke", "direction", "depth", "spin", "pace", "intent", "outcome"]


@dataclass
class EvalMetrics:
    examples: int = 0
    nll_total: float = 0.0
    top1_hits: int = 0
    top3_hits: int = 0
    top5_hits: int = 0
    factor_hits: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.factor_hits is None:
            self.factor_hits = Counter()

    def as_rates(self) -> dict[str, float]:
        denominator = max(self.examples, 1)
        rates = {
            "examples": float(self.examples),
            "nll": self.nll_total / denominator,
            "top1": self.top1_hits / denominator,
            "top3": self.top3_hits / denominator,
            "top5": self.top5_hits / denominator,
        }
        for factor in FACTORS:
            hits = self.factor_hits[factor] if self.factor_hits else 0
            rates[f"{factor}_top1"] = hits / denominator
        return rates


def evaluate_policy(policy: FrequencyPolicy, examples: Iterable[TrainingExample]) -> EvalMetrics:
    metrics = EvalMetrics()
    for example in examples:
        distribution = policy.predict_distribution(example.state)
        ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
        target_key = example.target.action_key()
        probability = distribution.get(target_key, 1e-12)

        metrics.examples += 1
        metrics.nll_total -= math.log(probability)
        metrics.top1_hits += int(bool(ranked) and ranked[0][0] == target_key)
        metrics.top3_hits += int(target_key in {key for key, _ in ranked[:3]})
        metrics.top5_hits += int(target_key in {key for key, _ in ranked[:5]})

        if ranked:
            score_factors(RallyShotToken.from_key(ranked[0][0]), example.target, metrics)

    return metrics


def split_examples(path: str, eval_bucket: int, buckets: int, want_eval: bool) -> Iterable[TrainingExample]:
    for example in iter_examples(path):
        bucket = bucket_match_id(example.match_id, buckets)
        if (bucket == eval_bucket) == want_eval:
            yield example


def bucket_match_id(match_id: str, buckets: int) -> int:
    digest = hashlib.sha1(match_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def score_factors(predicted: RallyShotToken, target: RallyShotToken, metrics: EvalMetrics) -> None:
    assert metrics.factor_hits is not None
    for factor in FACTORS:
        metrics.factor_hits[factor] += int(getattr(predicted, factor) == getattr(target, factor))

