from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from tennis_policy.data.sequence_builder import build_examples
from tennis_policy.models.baseline_markov import FrequencyPolicy
from tennis_policy.training.validation_metrics import bucket_match_id, evaluate_policy, split_examples


ACTION_A = "rally|forehand|crosscourt|deep|topspin|medium|neutral|in_play"
ACTION_B = "rally|backhand|crosscourt|deep|topspin|medium|neutral|in_play"
ACTION_C = "rally|forehand|down_the_line|deep|topspin|medium|attack|winner"


class FixedPolicy(FrequencyPolicy):
    def __init__(self, distribution: dict[str, float]) -> None:
        super().__init__(Counter(distribution))
        self.distribution = distribution

    def predict_distribution(self, state=None) -> dict[str, float]:
        return self.distribution


class ValidationMetricsTests(unittest.TestCase):
    def test_evaluate_policy_computes_topk_nll_and_factor_accuracy(self) -> None:
        examples = _examples_for_targets([ACTION_A, ACTION_C])
        policy = FixedPolicy({ACTION_A: 0.6, ACTION_B: 0.3, ACTION_C: 0.1})

        metrics = evaluate_policy(policy, examples)
        rates = metrics.as_rates()

        self.assertEqual(metrics.examples, 2)
        self.assertAlmostEqual(rates["top1"], 0.5)
        self.assertAlmostEqual(rates["top3"], 1.0)
        self.assertAlmostEqual(rates["top5"], 1.0)
        self.assertAlmostEqual(rates["stroke_top1"], 1.0)
        self.assertAlmostEqual(rates["direction_top1"], 0.5)
        self.assertGreater(rates["nll"], 1.0)

    def test_match_bucket_split_has_no_match_id_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rallies.jsonl"
            rows = [
                {"match_id": "match-a", "surface": "hard", "server": "ai", "point_score": "0-0", "shots": [ACTION_A, ACTION_B]},
                {"match_id": "match-b", "surface": "clay", "server": "player", "point_score": "15-0", "shots": [ACTION_A, ACTION_C]},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            eval_bucket = bucket_match_id("match-a", 10)
            train = list(split_examples(str(path), eval_bucket, 10, want_eval=False))
            eval_ = list(split_examples(str(path), eval_bucket, 10, want_eval=True))

        train_matches = {example.match_id for example in train}
        eval_matches = {example.match_id for example in eval_}
        self.assertTrue(eval_matches)
        self.assertFalse(train_matches & eval_matches)


def _examples_for_targets(targets: list[str]):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rallies.jsonl"
        rows = [
            {
                "match_id": f"match-{idx}",
                "surface": "hard",
                "server": "ai",
                "point_score": "0-0",
                "shots": [ACTION_A, target],
            }
            for idx, target in enumerate(targets)
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return build_examples(path)


if __name__ == "__main__":
    unittest.main()

