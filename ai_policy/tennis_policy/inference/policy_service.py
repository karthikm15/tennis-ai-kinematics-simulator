from __future__ import annotations

import argparse
import json

from tennis_policy.data.mcp_parser import parse_shot
from tennis_policy.models.baseline_markov import load_baseline_policy
from tennis_policy.schemas import MatchContext, RallyState


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample one action from a saved Phase 1 baseline policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--history", nargs="*", default=["serve wide flat fast", "return backhand crosscourt deep topspin medium"])
    args = parser.parse_args()

    policy = load_baseline_policy(args.model)
    state = RallyState(
        match=MatchContext(surface="hard", point_score="30-15", server="ai"),
        rally_count=len(args.history),
        previous_actions=[parse_shot(raw, index=i) for i, raw in enumerate(args.history)],
    )
    action = policy.sample_action(state, seed=7)
    print(json.dumps(action.to_dict(), indent=2))


if __name__ == "__main__":
    main()

