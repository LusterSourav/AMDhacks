FROM python:3.11-slim

WORKDIR /app

# ponytail: only wget needed — use pre-built llama-cpp wheel for linux/amd64
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -r requirements.txt

RUN mkdir -p /models && \
    wget -q "https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf" \
    -O /models/gemma-3-1b-it-Q4_K_M.gguf

COPY . .

ENV LOCAL_MODEL_PATH=/models/gemma-3-1b-it-Q4_K_M.gguf

CMD ["python", "run.py"]
