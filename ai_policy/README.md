# AI Opponent Policy

Python Phase 1 module for the tennis simulator's general opponent policy.

Phase 1 trains a behavioral cloning model: given match context and rally history,
predict the next categorical tennis action. The TypeScript simulator/physics layer
can then translate that categorical action into a continuous shot.

## What Is Included

- Canonical action/state schemas.
- A small action vocabulary shared by parser, baselines, and model code.
- A dataset builder that converts rally rows into next-shot examples.
- Frequency and Markov baselines.
- A compact PyTorch Transformer policy for behavioral cloning.
- A local inference wrapper with masked sampling.

## Quick Start

Run the dependency-free demo:

```bash
PYTHONPATH=ai_policy python3 -m tennis_policy.training.train_baseline --input ai_policy/examples/sample_rallies.jsonl --model-out ai_policy/artifacts/frequency_policy.json
PYTHONPATH=ai_policy python3 -m tennis_policy.inference.policy_service --model ai_policy/artifacts/frequency_policy.json
```

Run tests:

```bash
PYTHONPATH=ai_policy python3 -m unittest discover -s ai_policy/tests
```

Optional editable install:

```bash
python3 -m pip install -e ai_policy
```

Train the Transformer after installing PyTorch:

```bash
PYTHONPATH=ai_policy python3 -m tennis_policy.training.train_transformer \
  --input ai_policy/examples/sample_rallies.jsonl \
  --checkpoint ai_policy/artifacts/transformer_policy.pt \
  --epochs 5
```

## Expected Real Data Format

The dataset builder accepts JSONL rows. You can generate that JSONL from a local
Match Charting Project checkout:

```bash
git clone --depth 1 https://github.com/JeffSackmann/tennis_MatchChartingProject.git ai_policy/data/raw/tennis_MatchChartingProject

PYTHONPATH=ai_policy python3 -m tennis_policy.data.prepare_mcp_dataset \
  --mcp-dir ai_policy/data/raw/tennis_MatchChartingProject \
  --output ai_policy/data/processed/mcp_rallies.jsonl \
  --limit-points 5000

PYTHONPATH=ai_policy python3 -m tennis_policy.training.train_baseline \
  --input ai_policy/data/processed/mcp_rallies.jsonl \
  --model-out ai_policy/artifacts/mcp_markov_policy.json \
  --kind markov
```

Evaluate a baseline on held-out matches:

```bash
PYTHONPATH=ai_policy python3 -m tennis_policy.training.evaluate_baseline \
  --input ai_policy/data/processed/mcp_rallies.jsonl \
  --kind markov \
  --order 2 \
  --eval-bucket 9
```

The raw MCP data and generated processed files are ignored by git.

Manual/demo JSONL rows look like this:

```json
{"match_id":"demo-1","surface":"hard","server":"ai","point_score":"30-15","shots":["serve wide flat fast","return backhand crosscourt deep topspin medium","forehand crosscourt deep topspin medium"]}
```

Each rally row becomes multiple supervised examples:

```text
previous shots -> next shot
```

## License Note

The Tennis Abstract Match Charting Project is licensed CC BY-NC-SA 4.0. Treat MCP
training as non-commercial unless the project obtains a separate commercial data
license or uses an alternative dataset.
