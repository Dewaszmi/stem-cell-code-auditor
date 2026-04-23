FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONPATH="/app/src"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/repos

COPY . .

ENTRYPOINT ["python", "-m", "stem_cell_code_auditor.main"]