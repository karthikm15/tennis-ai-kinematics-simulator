from __future__ import annotations

import re

from tennis_policy.schemas import RallyShotToken

TOKEN_RE = re.compile(r"[a-zA-Z_]+")


def _words(raw: str) -> set[str]:
    normalized = raw.lower().replace("-", " ").replace("/", " ")
    return set(TOKEN_RE.findall(normalized))


def parse_shot(raw: str, index: int = 0) -> RallyShotToken:
    """Parse a normalized/MCP-like shot description into canonical factors.

    This parser is deliberately conservative. It handles plain English-like
    tokens now and gives us one place to harden raw MCP notation over time.
    """

    words = _words(raw)
    joined = " ".join(sorted(words))

    if "serve" in words:
        phase = "serve"
        stroke = "serve"
    elif "return" in words:
        phase = "return"
        stroke = _infer_stroke(words)
    elif "volley" in words:
        phase = "volley"
        stroke = "volley_backhand" if "backhand" in words else "volley_forehand"
    elif "overhead" in words or "smash" in words:
        phase = "overhead"
        stroke = "overhead"
    elif "approach" in words:
        phase = "approach"
        stroke = _infer_stroke(words)
    else:
        phase = "rally" if index > 1 else "return"
        stroke = _infer_stroke(words)

    outcome = "in_play"
    if "winner" in words:
        outcome = "winner"
    elif "net" in words:
        outcome = "net"
    elif "long" in words:
        outcome = "long"
    elif "wide" in words and "serve" not in words:
        outcome = "wide"
    elif "error" in words or "fault" in words:
        outcome = "unforced_error" if "unforced" in words else "forced_error"

    direction = _infer_direction(words, joined)
    depth = _infer_depth(words)
    spin = _infer_spin(words)
    pace = _infer_pace(words)
    intent = _infer_intent(words, outcome, depth, pace)

    return RallyShotToken(
        phase=phase,  # type: ignore[arg-type]
        stroke=stroke,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        depth=depth,  # type: ignore[arg-type]
        spin=spin,  # type: ignore[arg-type]
        pace=pace,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        outcome=outcome,  # type: ignore[arg-type]
    )


def _infer_stroke(words: set[str]) -> str:
    if "slice" in words and "backhand" in words:
        return "slice_backhand"
    if "forehand" in words or "fh" in words:
        return "forehand"
    if "backhand" in words or "bh" in words:
        return "backhand"
    if "lob" in words:
        return "lob"
    if "drop" in words:
        return "drop_shot"
    if "slice" in words:
        return "slice_backhand"
    return "unknown"


def _infer_direction(words: set[str], joined: str) -> str:
    if "crosscourt" in words or "cc" in words:
        return "crosscourt"
    if "line" in words or "dtl" in words or "down the line" in joined:
        return "down_the_line"
    if "inside" in words and "out" in words:
        return "inside_out"
    if "inside" in words and "in" in words:
        return "inside_in"
    if "wide" in words:
        return "wide"
    if "body" in words:
        return "body"
    if "tee" in words or "t" in words:
        return "tee"
    if "middle" in words or "center" in words or "centre" in words:
        return "middle"
    return "unknown"


def _infer_depth(words: set[str]) -> str:
    if "deep" in words:
        return "deep"
    if "short" in words or "drop" in words:
        return "short"
    if "medium" in words or "mid" in words:
        return "medium"
    return "unknown"


def _infer_spin(words: set[str]) -> str:
    if "topspin" in words or "heavy" in words:
        return "topspin"
    if "slice" in words or "chip" in words:
        return "slice"
    if "kick" in words:
        return "kick"
    if "flat" in words:
        return "flat"
    return "unknown"


def _infer_pace(words: set[str]) -> str:
    if "fast" in words or "hard" in words:
        return "fast"
    if "slow" in words or "soft" in words:
        return "slow"
    if "medium" in words:
        return "medium"
    return "unknown"


def _infer_intent(words: set[str], outcome: str, depth: str, pace: str) -> str:
    if outcome == "winner" or "attack" in words or pace == "fast":
        return "attack"
    if "defend" in words or "defensive" in words or "lob" in words:
        return "defend"
    if "reset" in words or "chip" in words or depth == "medium":
        return "reset"
    return "neutral"

