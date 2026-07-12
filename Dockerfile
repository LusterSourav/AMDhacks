FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py fireworks_client.py cloud.py main.py baseline_router.py code_exec.py run.py ./

ENV PYTHONUNBUFFERED=1

CMD ["python3", "agent.py"]
