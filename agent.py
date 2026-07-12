"""
Container entrypoint for AMD Developer Hackathon Act II, Track 1.
Combines reference code structure + our local solvers + DistilBERT routing.
Reads ALLOWED_MODELS at runtime. All API calls through FIREWORKS_BASE_URL.
"""
import json
import os
import sys
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from fireworks_client import chat
from cloud import cloud
from main import classify, LOCAL_SOLVERS

INPUT_PATH = Path(os.environ.get("TASK_INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("TASK_OUTPUT_PATH", "/output/results.json"))

# ─── ALLOWED MODELS ──────────────────────────────────────────────────────────

def get_allowed_models():
    allowed = os.environ.get("ALLOWED_MODELS", "")
    if allowed:
        return [m.strip() for m in allowed.split(",") if m.strip()]
    # Fallback for local testing
    return [
        "accounts/fireworks/models/gpt-oss-120b",
        "accounts/fireworks/models/minimax-m3",
        "accounts/fireworks/models/deepseek-v4-pro",
    ]

# ─── ROUTING ENGINE ──────────────────────────────────────────────────────────

def route_one(prompt: str) -> str:
    """Route a single prompt through the hybrid pipeline. Returns answer text."""
    # ponytail: local solvers first (zero tokens)
    category = classify(prompt)
    if category in LOCAL_SOLVERS:
        result = LOCAL_SOLVERS[category](prompt)
        if result:
            return result

    # ponytail: cloud fallback (token cost)
    try:
        answer = cloud(prompt, category)
        answer = re.sub(r'^\s*Answer\s*[:]\s*', '', answer).strip()
        return answer
    except Exception as e:
        return f"error: {e}"

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    tasks = json.loads(INPUT_PATH.read_text())
    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for i, task in enumerate(tasks):
            futures[pool.submit(
                lambda t=task: {"task_id": t["task_id"], "answer": route_one(t["prompt"])}
            )] = i

        for future in futures:
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=30)
            except FuturesTimeout:
                results[idx] = {"task_id": tasks[idx]["task_id"], "answer": "timeout"}
            except Exception as e:
                results[idx] = {"task_id": tasks[idx]["task_id"], "answer": f"error: {e}"}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {OUTPUT_PATH}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"agent failed: {e}", file=sys.stderr)
        sys.exit(1)
