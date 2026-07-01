from __future__ import annotations

import argparse
from pathlib import Path

from tennis_policy.data.action_vocab import ActionVocab
from tennis_policy.data.sequence_builder import build_examples
from tennis_policy.models.transformer_policy import TransformerOpponentPolicy, torch


def main() -> None:
    if torch is None:
        raise SystemExit("PyTorch is required. Install with: pip install -r ai_policy/requirements.txt")

    parser = argparse.ArgumentParser(description="Train the Phase 1 Transformer behavioral cloning policy.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-history", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    examples = build_examples(args.input, max_history=args.max_history)
    keys = [example.target.action_key() for example in examples]
    for example in examples:
        keys.extend(action.action_key() for action in example.state.previous_actions)
    vocab = ActionVocab.build(keys)

    x, y = _tensorize(examples, vocab, args.max_history)
    model = TransformerOpponentPolicy(vocab_size=len(vocab.token_to_id), pad_id=vocab.pad_id, max_seq_len=args.max_history + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(x.size(0))
        total_loss = 0.0
        for start in range(0, x.size(0), args.batch_size):
            idx = permutation[start : start + args.batch_size]
            logits = model(x[idx])
            loss = loss_fn(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.numel()
        print(f"epoch={epoch} loss={total_loss / max(x.size(0), 1):.4f}")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.to_dict(),
            "max_history": args.max_history,
            "metadata": {
                "model_name": "general-transformer-v1",
                "model_version": "0.1.0",
                "supports_lora": True,
            },
        },
        checkpoint,
    )
    print(f"saved={checkpoint}")


def _tensorize(examples, vocab: ActionVocab, max_history: int):
    rows = []
    targets = []
    for example in examples:
        ids = [vocab.bos_id] + [vocab.encode(action) for action in example.state.previous_actions[-max_history:]]
        ids = ids[-(max_history + 1) :]
        padded = [vocab.pad_id] * (max_history + 1 - len(ids)) + ids
        rows.append(padded)
        targets.append(vocab.encode(example.target))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


if __name__ == "__main__":
    main()

