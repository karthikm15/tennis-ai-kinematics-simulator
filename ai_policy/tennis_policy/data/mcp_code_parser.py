from __future__ import annotations

import re

from tennis_policy.data.mcp_parser import parse_shot
from tennis_policy.schemas import RallyShotToken

SERVE_DIRECTION = {
    "4": "wide",
    "5": "body",
    "6": "tee",
}

SHOT_STROKES = {
    "f": "forehand",
    "b": "backhand",
    "r": "forehand",
    "s": "slice backhand",
    "v": "volley forehand",
    "z": "drop shot",
    "l": "lob",
    "o": "overhead",
}

SHOT_DIRECTION = {
    "1": "crosscourt",
    "2": "middle",
    "3": "down the line",
}

RETURN_DEPTH = {
    "7": "deep",
    "8": "medium",
    "9": "short",
}

OUTCOME_MARKERS = {
    "*": "winner",
    "#": "forced error",
    "@": "unforced error",
    "n": "net error",
    "w": "wide error",
    "d": "double fault error",
}

SHOT_START_RE = re.compile(r"[fbrsvzlo]")


def parse_mcp_point_sequence(sequence: str) -> list[RallyShotToken]:
    """Convert one MCP compact point sequence into canonical action tokens.

    MCP's full notation is rich. This first-pass decoder extracts the pieces
    needed for Phase 1 behavioral cloning and lets unknown details fall into
    explicit unknown buckets downstream.
    """

    sequence = (sequence or "").strip()
    if not sequence:
        return []

    shots: list[str] = []
    first = sequence[0]
    if first in SERVE_DIRECTION:
        shots.append(f"serve {SERVE_DIRECTION[first]} fast")
        cursor = 1
    else:
        cursor = 0

    while cursor < len(sequence):
        char = sequence[cursor]
        if char not in SHOT_STROKES:
            cursor += 1
            continue

        next_cursor = cursor + 1
        while next_cursor < len(sequence) and not SHOT_START_RE.match(sequence[next_cursor]):
            next_cursor += 1

        raw_token = sequence[cursor:next_cursor]
        shots.append(_decode_mcp_shot_token(raw_token, shot_index=len(shots)))
        cursor = next_cursor

    return [parse_shot(shot, index=index) for index, shot in enumerate(shots)]


def _decode_mcp_shot_token(raw_token: str, shot_index: int) -> str:
    stroke_code = raw_token[0]
    parts = [SHOT_STROKES.get(stroke_code, "unknown")]

    direction = _first_matching(raw_token, SHOT_DIRECTION)
    depth = _first_matching(raw_token, RETURN_DEPTH)
    outcome = _first_matching(raw_token, OUTCOME_MARKERS)

    if direction:
        parts.append(direction)
    else:
        parts.append("middle")

    if depth:
        parts.append(depth)
    elif "!" in raw_token or "+" in raw_token:
        parts.append("short")
    else:
        parts.append("deep" if shot_index <= 2 else "medium")

    if stroke_code == "s":
        parts.append("slice")
        parts.append("slow")
    elif stroke_code in {"z", "l"}:
        parts.append("slow")
    else:
        parts.append("topspin")
        parts.append("medium")

    if outcome:
        parts.append(outcome)

    return " ".join(parts)


def _first_matching(raw_token: str, mapping: dict[str, str]) -> str | None:
    for char in raw_token:
        if char in mapping:
            return mapping[char]
    return None

