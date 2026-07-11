"""Competition entry point: reads /input/tasks.json, writes /output/results.json"""

import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from main import route

# ponytail: cache key = query hash, saves duplicate API calls
_answer_cache = {}


def _task_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def _run_one(task: dict) -> dict:
    task_id = task["task_id"]
    prompt = task["prompt"]

    h = _task_hash(prompt)
    if h in _answer_cache:
        return {"task_id": task_id, "answer": _answer_cache[h]}

    try:
        result = route(prompt)
        answer = result["answer"]
        _answer_cache[h] = answer
        return {"task_id": task_id, "answer": answer}
    except Exception as e:
        return {"task_id": task_id, "answer": f"error: {e}"}


def main():
    with open("/input/tasks.json") as f:
        tasks = json.load(f)

    results = [None] * len(tasks)

    # ponytail: thread pool with 30s per-task timeout, 4 workers
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for i, task in enumerate(tasks):
            futures[pool.submit(_run_one, task)] = i

        for future in futures:
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=30)
            except FuturesTimeout:
                results[idx] = {"task_id": tasks[idx]["task_id"], "answer": "timeout"}
            except Exception as e:
                results[idx] = {"task_id": tasks[idx]["task_id"], "answer": f"error: {e}"}

    os.makedirs("/output", exist_ok=True)
    with open("/output/results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
