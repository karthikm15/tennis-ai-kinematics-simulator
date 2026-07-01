from __future__ import annotations

import argparse

from tennis_policy.data.sequence_builder import build_examples, iter_examples
from tennis_policy.models.baseline_markov import FrequencyPolicy, MarkovPolicy, negative_log_likelihood, top_k_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 1 frequency/Markov opponent-policy baselines.")
    parser.add_argument("--input", required=True, help="JSONL rally file.")
    parser.add_argument("--model-out", required=True, help="Where to write the baseline model JSON.")
    parser.add_argument("--kind", choices=["frequency", "markov"], default="markov")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--stream", action="store_true", help="Stream examples from disk for large MCP JSONL files.")
    args = parser.parse_args()

    if args.kind == "frequency":
        policy = FrequencyPolicy()
    else:
        policy = MarkovPolicy(order=args.order)

    if args.stream:
        policy.fit_iter(iter_examples(args.input))
        examples_for_eval = iter_examples(args.input)
        example_count = sum(policy.counts.values())
    else:
        examples = build_examples(args.input)
        policy.fit(examples)
        examples_for_eval = examples
        example_count = len(examples)

    policy.save(args.model_out)
    print(f"examples={example_count}")
    print(f"nll={negative_log_likelihood(policy, examples_for_eval):.4f}")
    print(f"top3={top_k_accuracy(policy, iter_examples(args.input) if args.stream else examples, k=3):.4f}")
    print(f"saved={args.model_out}")


if __name__ == "__main__":
    main()
