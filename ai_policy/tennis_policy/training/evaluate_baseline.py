from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tennis_policy.models.baseline_markov import FrequencyPolicy, MarkovPolicy
from tennis_policy.training.validation_metrics import FACTORS, evaluate_policy, split_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline next-action policies on held-out MCP matches.")
    parser.add_argument("--input", required=True, help="Rally JSONL file.")
    parser.add_argument("--kind", choices=["frequency", "markov"], default="markov")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--eval-bucket", type=int, default=9, help="Held-out match bucket from 0..9.")
    parser.add_argument("--buckets", type=int, default=10, help="Number of hash buckets for match-level splitting.")
    parser.add_argument("--csv-out", help="Optional path to write one-row CSV metrics.")
    args = parser.parse_args()

    if args.kind == "frequency":
        policy: FrequencyPolicy = FrequencyPolicy()
    else:
        policy = MarkovPolicy(order=args.order)

    train_examples = split_examples(args.input, args.eval_bucket, args.buckets, want_eval=False)
    policy.fit_iter(train_examples)

    metrics = evaluate_policy(policy, split_examples(args.input, args.eval_bucket, args.buckets, want_eval=True))
    if metrics.examples == 0:
        raise SystemExit(
            "No evaluation examples found. Choose a different --eval-bucket or use a larger dataset."
        )

    rates = metrics.as_rates()
    print(f"eval_bucket={args.eval_bucket}/{args.buckets}")
    print(f"eval_examples={metrics.examples}")
    print(f"nll={rates['nll']:.4f}")
    print(f"top1={rates['top1']:.4f}")
    print(f"top3={rates['top3']:.4f}")
    print(f"top5={rates['top5']:.4f}")
    for factor in FACTORS:
        print(f"{factor}_top1={rates[f'{factor}_top1']:.4f}")

    if args.csv_out:
        _write_csv(args.csv_out, args, rates)


def _write_csv(path: str, args: argparse.Namespace, rates: dict[str, float]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "input": args.input,
        "kind": args.kind,
        "order": args.order,
        "eval_bucket": args.eval_bucket,
        "buckets": args.buckets,
        **rates,
    }
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
