#!/usr/bin/env sh
# Full pre-submit check. Usage: sh scripts/check.sh [--docker]
set -e
cd "$(dirname "$0")/.."
PY=${PY:-python3}
echo "== scenario tests";      $PY tools/test_brain.py
echo "== latency bench";       $PY tools/bench.py
echo "== conformance (local)"; $PY mechfight-kit/check/check.py -- $PY -u bot.py
if [ "$1" = "--docker" ]; then
  echo "== docker build";       docker build --platform linux/amd64 -t my-bot .
  echo "== conformance (match sandbox, exactly as the scorer runs it)"
  $PY mechfight-kit/check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 \
    --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 \
    -v "$PWD/mechfight-kit/arena:/arena:ro" my-bot
fi
