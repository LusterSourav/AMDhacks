"""
Cloud inference with model routing by category.
Reads ALLOWED_MODELS at runtime. Uses fireworks_client for retries + token tracking.
Falls back through: primary → fallback chain → Claude.
"""
import os
import re
import time
import requests

from fireworks_client import chat

# ─── ALLOWED MODELS (read at runtime) ────────────────────────────────────────

def get_allowed_models():
    allowed = os.environ.get("ALLOWED_MODELS", "")
    if allowed:
        return [m.strip() for m in allowed.split(",") if m.strip()]
    return []

# ─── MODEL SELECTION ──────────────────────────────────────────────────────────

_CATEGORY_PREFS = {
    "math": ["deepseek", "kimi", "minimax", "gpt-oss"],
    "code_debug": ["kimi", "deepseek", "gpt-oss", "minimax"],
    "code_gen": ["kimi", "deepseek", "gpt-oss", "minimax"],
    "logic": ["deepseek", "kimi", "gpt-oss", "minimax"],
    "factual": ["gpt-oss", "minimax", "deepseek", "kimi"],
    "sentiment": ["minimax", "gpt-oss", "deepseek", "kimi"],
    "ner": ["gpt-oss", "minimax", "deepseek", "kimi"],
    "summarization": ["gpt-oss", "minimax", "deepseek", "kimi"],
    "default": ["gpt-oss", "minimax", "deepseek", "kimi"],
}

def _select_model(category: str, allowed_models: list) -> str:
    """Pick best model for category from allowed list."""
    if not allowed_models:
        return "accounts/fireworks/models/gpt-oss-120b"
    prefs = _CATEGORY_PREFS.get(category, _CATEGORY_PREFS["default"])
    for pref in prefs:
        for m in allowed_models:
            if pref.lower() in m.lower():
                return m
    return allowed_models[0]

# ─── PROMPTS (plain text, no JSON, category-specific) ───────────────────────

_PROMPTS = {
    "math": "Solve this math problem. Give ONLY the final answer, nothing else.\n\nQuestion: {query}\nAnswer:",
    "sentiment": "Classify the sentiment: Positive, Negative, Neutral, or Mixed.\n\nReview: {query}\nAnswer:",
    "ner": "Extract named entities as PERSON, ORGANIZATION, LOCATION, DATE.\n\nText: {query}\nAnswer:",
    "factual": "Answer this question concisely.\n\nQuestion: {query}\nAnswer:",
    "summarization": "Follow the exact formatting instructions in the query. Do not add or skip any format requirements.\n\n{query}\n\nAnswer:",
    "code_debug": "Find the bug in this code and how to fix it.\n\nCode: {query}\nBug:",
    "code_gen": "Write a Python function for this task. Code only.\n\nTask: {query}\nCode:",
    "logic": "Solve this step by step. Give the final answer.\n\nPuzzle: {query}\nAnswer:",
    "default": "Answer concisely.\n\nQuestion: {query}\nAnswer:",
}

_MAX_TOKENS = {
    "math": 512, "sentiment": 256, "ner": 256, "factual": 1024,
    "summarization": 512, "code_debug": 1024, "code_gen": 1024, "logic": 256,
    "default": 512,
}

# Claude via Fireworks Anthropic compatibility
_CLAUDE_MODEL = "accounts/fireworks/models/claude-opus-4-5"

# ─── MAIN ENTRY POINT ───────────────────────────────────────────────────────

def cloud(query: str, task_type: str = "default", difficulty: str = "easy") -> str:
    """Route query to best model from ALLOWED_MODELS. Returns plain text."""
    allowed = get_allowed_models()
    model = _select_model(task_type, allowed)
    max_tokens = _MAX_TOKENS.get(task_type, 256)
    prompt = _PROMPTS.get(task_type, _PROMPTS["default"]).format(query=query)

    # Try primary model via fireworks_client (3 retries, 20s timeout)
    try:
        result = chat(model, prompt, max_tokens=max_tokens)
        return result["text"]
    except Exception:
        pass

    # Try ALL other allowed models as fallback
    for m in allowed:
        if m == model:
            continue
        try:
            result = chat(m, prompt, max_tokens=max_tokens)
            return result["text"]
        except Exception:
            continue

    # Try Claude as last resort
    try:
        api_key = os.environ.get("FIREWORKS_API_KEY", "")
        base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        url = f"{base_url.replace('/v1', '')}/v1/messages"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        payload = {"model": _CLAUDE_MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"].strip()
    except Exception:
        pass

    return "Answer not available."
