from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tennis_policy.schemas import RallyShotToken

SPECIAL_TOKENS = ["<pad>", "<bos>", "<unk>"]


@dataclass
class ActionVocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def build(cls, action_keys: list[str], min_count: int = 1) -> "ActionVocab":
        counts = Counter(action_keys)
        tokens = SPECIAL_TOKENS + sorted(key for key, count in counts.items() if count >= min_count)
        token_to_id = {token: idx for idx, token in enumerate(tokens)}
        return cls(token_to_id=token_to_id, id_to_token={idx: token for token, idx in token_to_id.items()})

    def encode(self, action: RallyShotToken | str) -> int:
        key = action.action_key() if isinstance(action, RallyShotToken) else action
        return self.token_to_id.get(key, self.token_to_id["<unk>"])

    def decode(self, idx: int) -> str:
        return self.id_to_token.get(idx, "<unk>")

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    def to_dict(self) -> dict[str, dict[str, int] | dict[int, str]]:
        return {"token_to_id": self.token_to_id, "id_to_token": self.id_to_token}

    @classmethod
    def from_dict(cls, payload: dict) -> "ActionVocab":
        return cls(
            token_to_id={str(key): int(value) for key, value in payload["token_to_id"].items()},
            id_to_token={int(key): str(value) for key, value in payload["id_to_token"].items()},
        )

