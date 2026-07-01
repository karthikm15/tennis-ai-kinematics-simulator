from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from tennis_policy.data.mcp_parser import parse_shot
from tennis_policy.schemas import RallyShotToken
from tennis_policy.schemas import MatchContext, RallyState, TrainingExample


def load_rally_rows(path: str | Path) -> Iterable[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc


def examples_from_rally(row: dict, max_history: int = 16) -> list[TrainingExample]:
    shots = [_parse_action(raw, index=i) for i, raw in enumerate(row.get("shots", []))]
    match = MatchContext(
        surface=row.get("surface", "unknown"),
        point_score=row.get("point_score", "unknown"),
        server=row.get("server", "ai"),
    )
    match_id = str(row.get("match_id", "unknown"))
    examples: list[TrainingExample] = []

    for target_idx in range(1, len(shots)):
        history = shots[max(0, target_idx - max_history) : target_idx]
        state = RallyState(match=match, rally_count=target_idx, previous_actions=history)
        examples.append(TrainingExample(match_id=match_id, state=state, target=shots[target_idx]))

    return examples


def _parse_action(raw: str, index: int) -> RallyShotToken:
    if raw.count("|") == 7:
        return RallyShotToken.from_key(raw)
    return parse_shot(raw, index=index)


def build_examples(path: str | Path, max_history: int = 16) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    examples.extend(iter_examples(path, max_history=max_history))
    return examples


def iter_examples(path: str | Path, max_history: int = 16) -> Iterator[TrainingExample]:
    for row in load_rally_rows(path):
        yield from examples_from_rally(row, max_history=max_history)


def write_examples_jsonl(examples: list[TrainingExample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "match_id": example.match_id,
                "state": example.state.to_dict(),
                "target": example.target.action_key(),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
