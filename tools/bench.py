"""Offline timing + crash test: replays the kit's recorded stream straight into Brain.act() in-process.

    python tools/bench.py            # per-tick latency stats, exceptions, mode histogram
    python tools/bench.py -v         # also print each tick's mode and action

The recorded observations don't react to your actions (same as check.py), so this measures speed and
robustness, not skill. Keep p99 well under 50 ms: the scorer machine is not your laptop.
"""
import argparse
import collections
import gzip
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KIT = os.path.join(ROOT, "mechfight-kit")

import mechfight as mf  # noqa: E402
from brain import Brain  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default=os.path.join(KIT, "check", "stream.jsonl.gz"))
    ap.add_argument("--map-dir", default=os.path.join(KIT, "arena"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    with gzip.open(a.stream, "rt") as f:
        msgs = [json.loads(l) for l in f if l.strip()]
    hello = msgs[0]
    hello["map"]["dir"] = a.map_dir
    mf.draw.caps = hello["config"]["debug"]

    t0 = time.perf_counter()
    bot = Brain(hello)
    print(f"init: {(time.perf_counter() - t0) * 1000:.0f} ms (budget {hello['config']['bot']['hello_budget_ms']} ms)")

    times, errors, modes = [], 0, collections.Counter()
    for m in msgs[1:]:
        if m.get("type") != "obs":
            continue
        mf.draw.clear()
        s = time.perf_counter()
        try:
            act = bot.act(m)
        except Exception:
            errors += 1
            traceback.print_exc()
            continue
        times.append((time.perf_counter() - s) * 1000)
        json.dumps(act)  # must be serializable
        modes[bot.mode] += 1
        if a.verbose:
            print(f"t={m['t']:6.1f} {bot.mode:12s} {json.dumps(act)[:160]}")

    times.sort()
    pct = lambda q: times[min(len(times) - 1, int(q * len(times)))]
    budget = hello["config"]["bot"]["tick_budget_ms"]
    print(f"ticks: {len(times)}  errors: {errors}")
    print(f"act() ms  p50 {pct(.5):.2f}  p90 {pct(.9):.2f}  p99 {pct(.99):.2f}  max {times[-1]:.2f}  (budget {budget})")
    print("modes:", dict(modes.most_common()))
    sys.exit(1 if errors or times[-1] > budget * 0.6 else 0)


if __name__ == "__main__":
    main()
