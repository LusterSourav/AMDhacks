import os
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["FIREWORKS_BASE_URL"],
            api_key=os.environ["FIREWORKS_API_KEY"],
        )
    return _client


# ponytail: aggressive token limits — every token counts
MAX_TOKENS = {
    "math": 32,
    "sentiment": 16,
    "ner": 64,
    "factual": 64,
    "summarization": 128,
    "code_debug": 128,
    "logic": 64,
    "code_gen": 256,
    "default": 64,
}

# ponytail: model size ranking — cheaper models first (per-token pricing)
# minimax-m3: $0.30/$1.20 per 1M tokens (cheapest)
# kimi-k2p7-code: $0.95/$4.00 per 1M tokens (3x more expensive)
# Gemma 4 series: on-demand GPU only ($7/hr)
_MODEL_PRIORITY = [
    "minimax-m3", "minimax", "kimi-k2p7-code", "kimi",
    "gemma-4", "gemma",
]


def _pick_cheapest_model(allowed: list) -> str:
    """Sort allowed models by size, pick cheapest."""
    if not allowed:
        return "accounts/fireworks/models/minimax-m3"
    # ponytail: try to find smallest model in allowed list
    for priority in _MODEL_PRIORITY:
        for m in allowed:
            if priority in m.lower():
                return m
    return allowed[0]


def cloud(query: str, task_type: str = "default") -> str:
    client = _get_client()
    models = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    model = _pick_cheapest_model(models)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"Answer briefly:\n{query}"}],
        response_format={"type": "json_object"},
        max_tokens=MAX_TOKENS.get(task_type, 64),
        temperature=0,
    )
    return resp.choices[0].message.content
