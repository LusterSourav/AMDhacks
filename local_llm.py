"""Local LLM inference using Gemma 3 1B GGUF — zero Fireworks tokens."""

import os
import json
from llama_cpp import Llama

_model = None
_model_path = os.environ.get("LOCAL_MODEL_PATH", "/models/gemma-3-1b-it-Q4_K_M.gguf")


def _get_model():
    global _model
    if _model is None:
        if os.path.exists(_model_path):
            _model = Llama(model_path=_model_path, n_ctx=2048, n_threads=4, verbose=False)
        else:
            raise FileNotFoundError(f"Model not found: {_model_path}")
    return _model


def local_infer(query: str, task_type: str = "default") -> str:
    """Run inference locally. Returns JSON string or None on failure."""
    try:
        model = _get_model()
    except (FileNotFoundError, Exception):
        return None

    # ponytail: task-specific prompts to minimize output tokens
    prompts = {
        "factual": f"Answer briefly in one sentence: {query}",
        "sentiment": f"Classify sentiment as positive/negative/neutral. Reply with JSON: {{\"sentiment\": \"...\", \"score\": 0.0}}\nText: {query}",
        "ner": f"Extract named entities. Reply with JSON: {{\"entities\": [{{\"text\": \"...\", \"type\": \"...\"}}]}}\nText: {query}",
        "summarization": f"Summarize in one sentence: {query}",
        "code_debug": f"Find bugs in this code. Reply with JSON: {{\"errors\": [{{\"type\": \"...\", \"message\": \"...\"}}]}}\nCode: {query}",
        "math": f"Solve this math problem. Reply with JSON: {{\"answer\": \"...\"}}\nProblem: {query}",
        "logic": f"Solve this logic puzzle. Reply with JSON: {{\"answer\": \"...\"}}\nPuzzle: {query}",
        "code_gen": f"Write Python code. Reply with JSON: {{\"code\": \"...\", \"language\": \"python\"}}\nRequest: {query}",
    }

    prompt = prompts.get(task_type, f"Answer briefly: {query}")

    try:
        resp = model(
            [{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0,
            stop=["\n\n", "```"],
        )
        text = resp["choices"][0]["text"].strip()

        # ponytail: try to extract JSON from response
        json_match = None
        if "{" in text:
            start = text.index("{")
            # find matching closing brace
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        json_match = text[start:i+1]
                        break

        if json_match:
            # validate it's valid JSON
            json.loads(json_match)
            return json_match

        # if no JSON found, wrap as answer
        return json.dumps({"answer": text})

    except Exception:
        return None
