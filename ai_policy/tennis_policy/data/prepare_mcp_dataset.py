from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tennis_policy.data.mcp_code_parser import parse_mcp_point_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Match Charting Project CSVs into Phase 1 rally JSONL.")
    parser.add_argument("--mcp-dir", required=True, help="Path to tennis_MatchChartingProject checkout.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--tour", choices=["m", "w", "both"], default="both")
    parser.add_argument("--limit-points", type=int, default=0, help="Optional cap for quick local runs.")
    parser.add_argument("--min-actions", type=int, default=2, help="Minimum decoded actions required per point.")
    args = parser.parse_args()

    mcp_dir = Path(args.mcp_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tours = ["m", "w"] if args.tour == "both" else [args.tour]
    matches = _load_matches(mcp_dir, tours)
    point_files = [path for tour in tours for path in sorted(mcp_dir.glob(f"charting-{tour}-points-*.csv"))]

    rows_written = 0
    rows_seen = 0
    rows_skipped = 0
    with output.open("w", encoding="utf-8") as handle:
        for path in point_files:
            with path.open("r", encoding="utf-8", newline="") as csv_handle:
                reader = csv.DictReader(csv_handle)
                for row in reader:
                    rows_seen += 1
                    sequence = row.get("2nd") or row.get("1st") or ""
                    actions = parse_mcp_point_sequence(sequence)
                    if len(actions) < args.min_actions:
                        rows_skipped += 1
                        continue

                    match_id = row["match_id"]
                    metadata = matches.get(match_id, {})
                    payload = {
                        "match_id": match_id,
                        "surface": _normalize_surface(metadata.get("Surface", "unknown")),
                        "server": "ai" if row.get("Svr") == "1" else "player",
                        "point_score": row.get("Pts", "unknown") or "unknown",
                        "shots": [action.action_key() for action in actions],
                        "source": {
                            "tour": "men" if "-m-" in path.name else "women",
                            "file": path.name,
                            "point": row.get("Pt"),
                            "raw_sequence": sequence,
                        },
                    }
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
                    rows_written += 1

                    if args.limit_points and rows_seen >= args.limit_points:
                        print_summary(rows_seen, rows_written, rows_skipped, output)
                        return

    print_summary(rows_seen, rows_written, rows_skipped, output)


def _load_matches(mcp_dir: Path, tours: list[str]) -> dict[str, dict[str, str]]:
    matches: dict[str, dict[str, str]] = {}
    for tour in tours:
        path = mcp_dir / f"charting-{tour}-matches.csv"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                matches[row["match_id"]] = row
    return matches


def _normalize_surface(value: str) -> str:
    surface = value.strip().lower()
    if surface in {"hard", "clay", "grass"}:
        return surface
    if "indoor" in surface or "carpet" in surface:
        return "indoor"
    return "unknown"


def print_summary(rows_seen: int, rows_written: int, rows_skipped: int, output: Path) -> None:
    print(f"points_seen={rows_seen}")
    print(f"rallies_written={rows_written}")
    print(f"points_skipped={rows_skipped}")
    print(f"output={output}")


if __name__ == "__main__":
    main()

