FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .

# The sandbox uses its own 100-candidate artifact set. This is an offline image
# build step; rank.py itself only loads artifacts and never computes embeddings.
RUN python scripts/precompute.py \
    --candidates sandbox/sample_candidates.jsonl \
    --jd sandbox/job_description.docx \
    --output-dir data/recall

CMD ["python", "rank.py", "--candidates", "sandbox/sample_candidates.jsonl", "--out", "/app/PrajniX_sandbox.csv"]
