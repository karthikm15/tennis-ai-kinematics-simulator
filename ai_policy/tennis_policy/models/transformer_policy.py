from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised only in no-torch environments
    torch = None
    nn = None


if nn is not None:

    class TransformerOpponentPolicy(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            pad_id: int,
            max_seq_len: int = 32,
            hidden_size: int = 256,
            layers: int = 4,
            heads: int = 4,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.pad_id = pad_id
            self.max_seq_len = max_seq_len
            self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
            self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
            batch_size, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
            padding_mask = input_ids.eq(self.pad_id)
            encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
            lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
            pooled = encoded[torch.arange(batch_size, device=input_ids.device), lengths]
            return self.output(pooled)

else:

    class TransformerOpponentPolicy:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required for TransformerOpponentPolicy. Install ai_policy/requirements.txt.")

