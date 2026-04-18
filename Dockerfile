# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Working directory inside container ────────────────────────────────────────
WORKDIR /app

# ── Copy dependency list first (maximises layer-cache reuse) ──────────────────
COPY requirements.txt .

# ── Install dependencies (no pip cache → smaller image) ───────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code and saved models ────────────────────────────────────
COPY app.py .
COPY models/ ./models/

# ── Document which port the server listens on ─────────────────────────────────
EXPOSE 8000

# ── Start the server (0.0.0.0 makes it reachable from outside the container) ──
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
