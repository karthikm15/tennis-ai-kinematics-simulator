from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from tennis_policy.data.sequence_builder import iter_examples
from tennis_policy.models.baseline_markov import FrequencyPolicy, MarkovPolicy
from tennis_policy.schemas import RallyShotToken, TrainingExample


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline next-action policies on held-out MCP matches.")
    parser.add_argument("--input", required=True, help="Rally JSONL file.")
    parser.add_argument("--kind", choices=["frequency", "markov"], default="markov")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--eval-bucket", type=int, default=9, help="Held-out match bucket from 0..9.")
    parser.add_argument("--buckets", type=int, default=10, help="Number of hash buckets for match-level splitting.")
    args = parser.parse_args()

    if args.kind == "frequency":
        policy: FrequencyPolicy = FrequencyPolicy()
    else:
        policy = MarkovPolicy(order=args.order)

    train_examples = _matching_examples(args.input, args.eval_bucket, args.buckets, want_eval=False)
    policy.fit_iter(train_examples)

    metrics = evaluate(policy, _matching_examples(args.input, args.eval_bucket, args.buckets, want_eval=True))
    print(f"eval_bucket={args.eval_bucket}/{args.buckets}")
    print(f"eval_examples={metrics.examples}")
    print(f"nll={metrics.nll_total / max(metrics.examples, 1):.4f}")
    print(f"top1={metrics.top1_hits / max(metrics.examples, 1):.4f}")
    print(f"top3={metrics.top3_hits / max(metrics.examples, 1):.4f}")
    print(f"top5={metrics.top5_hits / max(metrics.examples, 1):.4f}")
    for factor in ["phase", "stroke", "direction", "depth", "spin", "pace", "intent", "outcome"]:
        hits = metrics.factor_hits[factor] if metrics.factor_hits else 0
        print(f"{factor}_top1={hits / max(metrics.examples, 1):.4f}")


def evaluate(policy: FrequencyPolicy, examples: Iterable[TrainingExample]) -> EvalMetrics:
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
            _score_factors(RallyShotToken.from_key(ranked[0][0]), example.target, metrics)

    return metrics


def _matching_examples(path: str, eval_bucket: int, buckets: int, want_eval: bool) -> Iterable[TrainingExample]:
    for example in iter_examples(path):
        bucket = _bucket(example.match_id, buckets)
        if (bucket == eval_bucket) == want_eval:
            yield example


def _bucket(match_id: str, buckets: int) -> int:
    digest = hashlib.sha1(match_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def _score_factors(predicted: RallyShotToken, target: RallyShotToken, metrics: EvalMetrics) -> None:
    assert metrics.factor_hits is not None
    for factor in ["phase", "stroke", "direction", "depth", "spin", "pace", "intent", "outcome"]:
        metrics.factor_hits[factor] += int(getattr(predicted, factor) == getattr(target, factor))


if __name__ == "__main__":
    main()

