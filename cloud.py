import os
from openai import OpenAI

_client = None

_PROMPTS = {
    "math": "Work through it in brief steps, then end with 'Answer: <value>' on its own line.",
    "sentiment": "State the sentiment as positive, negative, or neutral, then one short reason.",
    "ner": "List each entity as 'label: value', one per line, using labels person, organization, location, date.",
    "factual": "Give a correct, clear answer in under 120 words.",
    "summarization": "Output only the summary and obey any length or format constraint stated in the task.",
    "code_debug": "State the bug in one sentence, then give the corrected code in a single fenced block.",
    "code_gen": "Output only the code in a single fenced block.",
    "logic": "Reason in brief numbered steps, checking each constraint, then end with 'Answer: <value>' on its own line.",
    "default": "Answer concisely and directly.",
}

_MAX_TOKENS = {
    "math": 256, "sentiment": 64, "ner": 128, "factual": 128,
    "summarization": 256, "code_debug": 512, "code_gen": 512,
    "logic": 256, "default": 128,
}


def cloud(query: str, task_type: str = "default") -> str:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ.get("FIREWORKS_API_KEY", "missing"),
            base_url=os.environ.get("FIREWORKS_BASE_URL",
                                    "https://api.fireworks.ai/inference/v1"),
            timeout=25,
        )
    models = [m.strip() for m in os.environ.get("ALLOWED_MODELS", "").split(",") if m.strip()]
    model = min(models, key=len) if models else "accounts/fireworks/models/minimax-m3"

    system = _PROMPTS.get(task_type, _PROMPTS["default"])
    messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]
    resp = _client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=_MAX_TOKENS.get(task_type, 128),
        temperature=0,
    )
    return resp.choices[0].message.content
