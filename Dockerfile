# Built by the scorer on linux/amd64 (network on, <10 min). Runs with no network, read-only root
# (writable /tmp), 1 CPU, 512 MB, uid 65534, map mounted at /arena.
# Pin the base image so the final rebuild is identical.
FROM python:3.13-slim

# Make Python fast to start and never try to write .pyc files to the read-only disk.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /bot
# If you add dependencies, install them system-wide (not --user) with pinned versions, e.g.:
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
COPY mechfight.py bot.py ./
COPY brain ./brain
# Precompile as root at build time so the read-only runtime never needs to.
RUN python3 -m compileall -q /bot

ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
