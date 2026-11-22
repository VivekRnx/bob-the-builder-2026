"""Greedy forward selection of bot changes in the simulator, with paired statistics.

Every candidate is a set of TUNING overrides (MECH_TUNING). Each round runs the current best config and
every remaining candidate on the SAME seeds, compares them seed by seed (paired difference, so map/spawn
luck cancels out), accepts the best candidate if it beats the current config by z >= --z, then repeats
with the rest. Squad (practice-like) and FFA scores can be mixed with --ffa-weight.

    python tools/greedy.py --cands tools/candidates.json -n 96 -j 22
    python tools/greedy.py --cands tools/candidates.json --only lethal_off,no_seek   # just measure

Results are appended to runs/greedy/log.jsonl; the accepted config is written to runs/greedy/best.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "greedy")
SEED_RE = re.compile(r"^seed (\d+): hero pts \[([^\]]*)\]")


OPP = "tools.baseline.brain_v1"


def run(tuning: dict, mode: str, n: int, seed: int, j: int, hero: str = "brain") -> dict:
    """-> {seed: mean hero points in that match}"""
    env = dict(os.environ, MECH_TUNING=json.dumps(tuning), PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, "-B", os.path.join(ROOT, "tools", "sim.py"), "-n", str(n), "-j", str(j),
           "--seed", str(seed), "--mode", mode, "--hero", hero, "--opp", OPP]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        m = SEED_RE.match(line)
        if m:
            pts = [float(x) for x in m.group(2).split(",") if x.strip()]
            if pts:
                res[int(m.group(1))] = statistics.mean(pts)
    if len(res) < n * 0.9:
        raise RuntimeError(f"sim returned {len(res)}/{n} matches for {tuning}:\n{out[-2000:]}")
    return res


def paired(a: dict, b: dict):
    """mean(b - a), standard error, n over common seeds"""
    ks = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in ks]
    if len(d) < 2:
        return 0.0, 1e9, len(d)
    return statistics.mean(d), statistics.stdev(d) / math.sqrt(len(d)), len(d)


def evaluate(tuning, args, cache, key):
    if key in cache:
        return cache[key]
    t0 = time.time()
    r = {"squad": run(tuning, "squad", args.n, args.seed, args.j)}
    if args.ffa_weight > 0:
        r["ffa"] = run(tuning, "ffa", args.n_ffa, args.seed + 5000, args.j)
    r["wall"] = time.time() - t0
    cache[key] = r
    return r


def score_diff(base, cand, w):
    ds, ses, _ = paired(base["squad"], cand["squad"])
    if w > 0:
        df, sef, _ = paired(base["ffa"], cand["ffa"])
        # FFA points are on the same 0-15 scale; weight them in
        return (1 - w) * ds + w * df, math.sqrt(((1 - w) * ses) ** 2 + (w * sef) ** 2), ds, df
    return ds, ses, ds, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True, help="JSON file {name: {knob: value}}")
    ap.add_argument("--base", default=None, help="JSON file with the starting overrides (default: none)")
    ap.add_argument("-n", type=int, default=96)
    ap.add_argument("--n-ffa", type=int, default=24)
    ap.add_argument("--ffa-weight", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-j", type=int, default=22)
    ap.add_argument("--z", type=float, default=1.5, help="accept a candidate when diff/se >= z")
    ap.add_argument("--screen", action="store_true",
                    help="screen all candidates once, then add those with z >= --screen-z one by one (confirmed)")
    ap.add_argument("--screen-z", type=float, default=0.5)
    ap.add_argument("--opp", default="tools.housebot", help="opponent module for the simulator")
    ap.add_argument("--only", default=None, help="comma list: measure these candidates once, no greedy loop")
    a = ap.parse_args()
    global OPP
    OPP = a.opp
    os.makedirs(OUT, exist_ok=True)
    with open(a.cands) as f:
        cands = json.load(f)
    base = {}
    if a.base:
        with open(a.base) as f:
            base = json.load(f)
    if a.only:
        cands = {k: v for k, v in cands.items() if k in a.only.split(",")}
    log = open(os.path.join(OUT, "log.jsonl"), "a")
    accepted = []
    rnd = 0
    while cands:
        rnd += 1
        cache = {}
        b = evaluate(base, a, cache, "__base__")
        print(f"\n== round {rnd}: base squad {statistics.mean(b['squad'].values()):.3f}"
              + (f" ffa {statistics.mean(b['ffa'].values()):.3f}" if 'ffa' in b else "")
              + f"  accepted so far: {accepted or 'none'}", flush=True)
        rows = []
        for name, over in cands.items():
            cfg = dict(base)
            cfg.update(over)
            r = evaluate(cfg, a, cache, name)
            d, se, ds, df = score_diff(b, r, a.ffa_weight)
            z = d / se if se > 0 else 0.0
            rows.append((z, d, se, name, ds, df))
            print(f"  {name:28s} diff {d:+.3f} ± {se:.3f}  z {z:+.2f}  squad {ds:+.3f}"
                  + (f" ffa {df:+.3f}" if df is not None else "") + f"  ({r['wall']:.0f}s)", flush=True)
            log.write(json.dumps({"round": rnd, "name": name, "over": over, "base": base, "diff": d, "se": se,
                                  "squad": ds, "ffa": df, "n": a.n, "seed": a.seed}) + "\n")
            log.flush()
        if a.only:
            break
        rows.sort(reverse=True)
        if a.screen:
            # one screening round, then add the promising ones in order, each confirmed on fresh seeds
            for z0, d0, se0, name, *_ in rows:
                if z0 < a.screen_z:
                    break
                a.seed += a.n
                cache = {}
                b = evaluate(base, a, cache, "__base__")
                cfg = dict(base)
                cfg.update(cands[name])
                r = evaluate(cfg, a, cache, name)
                d, se, ds, df = score_diff(b, r, a.ffa_weight)
                z = d / se if se > 0 else 0.0
                verdict = "ACCEPT" if z >= a.z else "reject"
                print(f"  confirm {name:20s} on seeds {a.seed}+: diff {d:+.3f} ± {se:.3f} z {z:+.2f}  -> {verdict}"
                      f"  (base now {statistics.mean(b['squad'].values()):.3f})", flush=True)
                log.write(json.dumps({"phase": "confirm", "name": name, "over": cands[name], "base": base,
                                      "diff": d, "se": se, "n": a.n, "seed": a.seed}) + "\n")
                log.flush()
                if z >= a.z:
                    base.update(cands[name])
                    accepted.append(name)
                    with open(os.path.join(OUT, "best.json"), "w") as f:
                        json.dump(base, f, indent=1)
            break
        z, d, se, name, *_ = rows[0]
        if z < a.z:
            print(f"no candidate beats the current config by z >= {a.z} (best: {name} z {z:+.2f}); stopping")
            break
        print(f"ACCEPT {name} (diff {d:+.3f}, z {z:+.2f})", flush=True)
        base.update(cands.pop(name))
        accepted.append(name)
        # drop candidates that no longer change anything
        cands = {k: v for k, v in cands.items() if any(base.get(kk) != vv for kk, vv in v.items())}
        with open(os.path.join(OUT, "best.json"), "w") as f:
            json.dump(base, f, indent=1)
        a.seed += a.n  # fresh seeds each round: a lucky accept can't keep winning on the same matches
    print(f"\nfinal accepted: {accepted}\nconfig: {json.dumps(base)}")


if __name__ == "__main__":
    main()
