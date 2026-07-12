# Hybrid Token-Efficient Routing Agent

AMD Developer Hackathon ACT II, Track 1.

## How it works

8-worker thread pool. Each task hits local deterministic solvers first (0 tokens).
Only unmatched tasks fall back to Fireworks cloud via a 3-retry chain.

- **659-entry knowledge base** — capitals, science, history, tech
- **sympy math solver** — fractions, percentages, word numbers, unit conversions
- **Word-set sentiment** — negation-aware, contrast detection, specific word reasons
- **Regex NER** — 235+ entity patterns (PERSON, ORG, LOCATION, DATE, EMAIL, URL)
- **50 code templates** — fibonacci, factorial, palindrome, binary search, etc.
- **AST code debugger** — syntax errors, mutable defaults, off-by-one
- **Cloud routing** — per-category model selection, fallback chain through all allowed models

## Stack

- Python 3.11, sympy, requests
- Fireworks AI (gpt-oss-120b, minimax-m3, deepseek-v4-pro, kimi-k2p7-code)
- CPU-only, 345MB, linux/amd64

## Run

```bash
docker pull ghcr.io/lustersourav/amd-track1-agent:latest
docker run --rm \
  -v /path/to/tasks.json:/input/tasks.json:ro \
  -v /path/to/output:/output \
  -e FIREWORKS_API_KEY=fw_your_key \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS=accounts/fireworks/models/gpt-oss-120b,accounts/fireworks/models/minimax-m3,accounts/fireworks/models/deepseek-v4-pro,accounts/fireworks/models/kimi-k2p7-code \
  ghcr.io/lustersourav/amd-track1-agent:latest
```
