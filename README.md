# Hybrid Token-Efficient Routing Agent

A 4-tier routing agent built for the AMD Developer Hackathon ACT II (Track 1). Most tasks resolve at zero token cost through deterministic solvers. Only tasks that miss every local path reach the paid API.

## How It Works

A regex classifier (< 1ms) routes each task through escalating tiers. Tier 1 uses hardcoded lookups, pattern matching, and code templates. Tier 2 executes Python code via subprocess. Tier 3 runs Gemma 3 1B locally (GGUF, 0 Fireworks tokens). Tier 4 falls back to minimax-m3 on Fireworks, the cheapest available model at $0.30/1M tokens.

```
Query → Classifier → Deterministic Solver → Code Exec → Local LLM → Cloud
```

## What Gets Solved Locally

- **549-entry knowledge base** covers physics, chemistry, biology, geography, history, economics
- **Math solver** handles word numbers, percentages, unit conversions, sympy algebra
- **Sentiment analyzer** detects negation ("not bad" → positive) and contrast ("good but expensive" → neutral)
- **50 code templates** cover sorting, search, BFS/DFS, linked lists, dynamic programming
- **Code debugger** parses Python syntax errors via ast module
- **Logic engine** extracts if-then rules and applies them to stated facts
- **Regex NER** pulls dates, numbers, emails, phone numbers, and organization names

## Token Budget

| Component | Cost |
|-----------|------|
| Deterministic solvers | 0 tokens |
| Code execution | 0 tokens |
| Gemma 3 1B (local) | 0 tokens |
| minimax-m3 (cloud) | $0.30/1M tokens |

Per-task output caps keep cloud responses short: 16 tokens for sentiment, 32 for math, 128 for summarization, 256 for code generation.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Classifier, 7 local solvers, 549 FACTS, 50 code templates |
| `cloud.py` | Fireworks API wrapper, cheapest model picker |
| `local_llm.py` | Gemma 3 1B GGUF inference via llama.cpp |
| `run.py` | Thread pool runner with MD5 answer cache |
| `Dockerfile` | CPU-only container, linux/amd64, 871 MB compressed |

## Run It

```bash
pip install -r requirements.txt
export FIREWORKS_API_KEY=fw_your_key_here
export FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
export ALLOWED_MODELS=accounts/fireworks/models/minimax-m3
python run.py
```

## Docker

```bash
docker pull morningstarxcdcode/amd-track1-agent:latest

docker run --rm \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  -e FIREWORKS_API_KEY=fw_your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/minimax-m3 \
  morningstarxcdcode/amd-track1-agent:latest
```

## Competition

- **Track 1:** Hybrid Token-Efficient Routing Agent
- **Deadline:** July 11, 2026
- **Scoring:** 80% accuracy gate, then fewest tokens wins
