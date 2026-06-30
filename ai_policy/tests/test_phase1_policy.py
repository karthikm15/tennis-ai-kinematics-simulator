from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tennis_policy.data.mcp_parser import parse_shot
from tennis_policy.data.mcp_code_parser import parse_mcp_point_sequence
from tennis_policy.data.sequence_builder import build_examples
from tennis_policy.models.baseline_markov import MarkovPolicy, load_baseline_policy
from tennis_policy.schemas import ActionMask


ROOT = Path(__file__).resolve().parents[1]


class Phase1PolicyTests(unittest.TestCase):
    def test_parse_common_shot(self) -> None:
        shot = parse_shot("forehand crosscourt deep topspin medium", index=3)
        self.assertEqual(shot.phase, "rally")
        self.assertEqual(shot.stroke, "forehand")
        self.assertEqual(shot.direction, "crosscourt")
        self.assertEqual(shot.depth, "deep")
        self.assertEqual(shot.spin, "topspin")

    def test_build_examples_from_sample(self) -> None:
        examples = build_examples(ROOT / "examples" / "sample_rallies.jsonl")
        self.assertGreaterEqual(len(examples), 8)
        self.assertGreaterEqual(examples[0].state.rally_count, 1)

    def test_parse_real_mcp_code_sequence(self) -> None:
        shots = parse_mcp_point_sequence("4b37y1r3n#")
        self.assertGreaterEqual(len(shots), 3)
        self.assertEqual(shots[0].phase, "serve")
        self.assertEqual(shots[1].stroke, "backhand")

    def test_markov_policy_save_load_and_mask(self) -> None:
        examples = build_examples(ROOT / "examples" / "sample_rallies.jsonl")
        policy = MarkovPolicy(order=2).fit(examples)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            policy.save(path)
            loaded = load_baseline_policy(path)
            action = loaded.sample_action(examples[-1].state, mask=ActionMask(pace=["fast"]), seed=11)
        self.assertNotEqual(action.action.pace, "fast")
        self.assertGreater(len(action.top_k), 0)


if __name__ == "__main__":
    unittest.main()
