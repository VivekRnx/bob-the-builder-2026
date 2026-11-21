"""Train hiding/escape/dodging on randomized short episodes in the existing simulator.

Examples (from project root):
  python tools/train_defense.py --run runs/defense1 --episodes 1000 --workers 4
  python tools/train_defense.py --run runs/defense1 --episodes 1000 --workers 4  # resume
  python tools/train_defense.py --run runs/defense1 --test --eval-seeds 60

Policy: linear softmax actor + linear value baseline, episodic REINFORCE.
No external dependencies. Tiny model trains on CPU; GPU frameworks would add overhead.
Validation is reused for model selection. --test uses disjoint seeds, never promotes.
The official bot and champion.json are never modified. Simulator results are approximate.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from brain.defense import ACTIONS, FEATURES, VERSION, Brain as Defense, initial_weights, probabilities
from brain.brain import Brain as Current
from tools import sim

SCENARIOS = ["gun", "low_hp", "laser", "mortar", "crossfire", "zone"]
SPLITS = {"train": 100000, "validation": 10000000, "test": 20000000}
_NAV_CACHE = {}


def worker_init():
    # Share immutable navigation raster inside each process; search state remains per brain.
    import brain.nav as nav
    import mechfight as mf
    original = nav.NavGrid
    def cached_grid(world, **kwargs):
        key = (id(world), tuple(sorted(kwargs.items())))
        if key not in _NAV_CACHE:
            _NAV_CACHE[key] = original(world, **kwargs)
        result = copy.copy(_NAV_CACHE[key])
        result._search = None
        return result
    nav.NavGrid = cached_grid
    mf.log = lambda *a, **kw: None


class QuietCurrent(Current):
    params = {"debug": False}


class Learner(Defense):
    params = {"debug": False}


def scenario(seed, cls, horizon):
    rng = random.Random(seed)
    kind = SCENARIOS[seed % len(SCENARIOS)]
    h = sim.load_hello()
    # Focus on defense; remove resource and NPC distractions in this curriculum.
    h["config"]["npc"]["count"] = 0
    h["config"]["crates"]["count"] = 0
    h["config"]["loot"]["floor_count"] = 0
    n = 4 if kind == "crossfire" else 2
    s = sim.Sim(h, [cls]+[QuietCurrent]*(n-1), [None]*n, seed)
    hero = s.mechs[0]
    cover_corner = None
    # Put gun/low-HP starts beside real building corners, so hiding is learnable
    # rather than relying on chance encounters with cover in a large open map.
    if kind in ("gun", "low_hp"):
        buildings = [p for p in s.world.prisms if p.top-p.base > 10 and p.maxx-p.minx > 15 and p.maxy-p.miny > 15]
        rng.shuffle(buildings)
        for prism in buildings[:80]:
            x, y = prism.minx-7, prism.miny-7
            p = [x, y, s.world.ground(x, y)]
            if s.world.in_bounds(x, y, 20) and not s.blocked(*p, hero.height, hero.radius):
                hero.pos = p
                cover_corner = prism
                break
    hero.hp = hero.max_hp*rng.uniform(0.2, 0.5) if kind == "low_hp" else hero.max_hp*rng.uniform(0.55, 1.0)
    hero.energy *= rng.uniform(0.3, 1.0)
    start_hp = hero.hp
    center = hero.pos[:2]
    # All ordinary episodes start safely in zone; zone episodes require an early rotation.
    s.zphases = [dict(c=center, nc=[center[0]+(65 if kind == "zone" else 0), center[1]],
                     r=180.0, nr=38.0 if kind == "zone" else 180.0,
                     s0=2.0, s1=max(4.0, horizon*0.7), dps=35.0, t0=0.0)]
    for i, enemy in enumerate(s.mechs[1:]):
        for _ in range(100):
            angle = rng.uniform(-math.pi, math.pi)
            distance = rng.uniform(35, 110)
            p = [hero.pos[0]+math.cos(angle)*distance, hero.pos[1]+math.sin(angle)*distance]
            p.append(s.world.ground(*p))
            if s.world.in_bounds(p[0], p[1], 15) and not s.blocked(*p, enemy.height, enemy.radius):
                enemy.pos = p
                break
        if cover_corner is not None and i == 0:
            p = [cover_corner.minx+22, cover_corner.miny-22]
            p.append(s.world.ground(*p))
            if not s.blocked(*p, enemy.height, enemy.radius):
                enemy.pos = p
        enemy.aim = [math.atan2(hero.pos[1]-enemy.pos[1], hero.pos[0]-enemy.pos[0]), 0.0]
        if kind in ("laser", "mortar") or kind == "crossfire" and i > 0:
            weapon = kind if kind in ("laser", "mortar") else ("laser" if i == 1 else "mortar")
            enemy.weapons[1] = sim.Weapon(weapon, enemy.wstats[weapon])
            if weapon == "laser":
                enemy.weapons[1].charge = enemy.wstats[weapon]["charge"]*0.6
    hero.aim = [math.atan2(s.mechs[1].pos[1]-hero.pos[1], s.mechs[1].pos[0]-hero.pos[0]), 0.0]
    # A visible near-impact shell guarantees actual dodging exposure in mortar episodes.
    if kind == "mortar":
        ws = s.cfg["weapons"]["mortar"]
        offset = rng.uniform(-6, 6)
        s.projectiles.append(dict(id=s._nid("proj"), kind="mortar", owner=1,
            pos=[hero.pos[0]+offset, hero.pos[1], hero.pos[2]+28],
            vel=[0.0, 0.0, -10.0], left=4.0, dmg=ws["damage"]))
        hero.aim = [0.0 if offset >= 0 else math.pi, 0.5]
    return s, kind, start_hp


def episode(job):
    weights, seed, horizon, training = job
    Learner.weights, Learner.training = weights, training
    s, kind, start_hp = scenario(seed, Learner if weights is not None else QuietCurrent, horizon)
    s.run(max_t=horizon, stop_when=lambda world: not world.mechs[0].alive)
    m = s.mechs[0]
    errors = getattr(m.bot, "errors", 0)
    if errors:
        raise RuntimeError(f"Controller failed {errors} times, seed {seed}")
    in_zone = hdist2(m.pos, s.zone()["center"]) <= s.zone()["radius"]
    if isinstance(m.bot, Defense):
        m.bot.finish(s.t, m.hp, m.max_hp, m.alive, in_zone)
    score = (3*(max(0, m.hp)-start_hp)/m.max_hp + 0.04*s.t
             - 0.15*m.diag["out_s"] + (2+max(0, m.hp)/m.max_hp if m.alive else -3))
    times = sorted(m.act_ms)
    return dict(seed=seed, scenario=kind, score=score, alive=m.alive,
                hp=max(0, m.hp)/m.max_hp, survival_s=s.t, damage=m.diag["taken"],
                zone_s=m.diag["out_s"], max_ms=max(times, default=0),
                late=sum(t > 50 for t in times), decisions=len(times),
                p99_ms=times[min(len(times)-1, int(len(times)*0.99))] if times else 0,
                trajectory=getattr(m.bot, "trajectory", []),
                actions=getattr(m.bot, "action_counts", {}))


def hdist2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def learn(weights, value, episodes, lr=0.08, gamma=0.99):
    samples = []
    for ep in episodes:
        ret = 0.0
        for step in reversed(ep["trajectory"]):
            ret = step["reward"]+gamma*ret
            predicted = sum(a*b for a, b in zip(value, step["x"]))
            samples.append((step, ret, ret-predicted))
    if not samples:
        raise RuntimeError("No trainable decisions collected")
    scale = max(1.0, statistics.pstdev(v[2] for v in samples))
    grad = [[0.0]*len(FEATURES) for _ in ACTIONS]
    vg = [0.0]*len(FEATURES)
    for step, ret, advantage in samples:
        adv = max(-5, min(5, advantage/scale))
        for i in range(len(ACTIONS)):
            coefficient = ((1.0 if i == step["action"] else 0.0)-step["p"][i])*adv
            for j, x in enumerate(step["x"]):
                grad[i][j] += coefficient*x
        for j, x in enumerate(step["x"]):
            vg[j] += max(-5, min(5, advantage))*x
    n = len(samples)
    norm = math.sqrt(sum((v/n)**2 for row in grad for v in row))
    step_size = lr/max(1.0, norm)
    result = [[max(-12, min(12, w+step_size*g/n)) for w, g in zip(row, gr)]
              for row, gr in zip(weights, grad)]
    value = [max(-12, min(12, v+0.01*g/n)) for v, g in zip(value, vg)]
    return result, value, len(samples)


def summary(rows):
    return dict(episodes=len(rows), mean_score=statistics.mean(r["score"] for r in rows),
                survival_rate=statistics.mean(r["alive"] for r in rows),
                mean_hp=statistics.mean(r["hp"] for r in rows),
                mean_damage=statistics.mean(r["damage"] for r in rows),
                mean_survival_s=statistics.mean(r["survival_s"] for r in rows),
                mean_zone_s=statistics.mean(r["zone_s"] for r in rows),
                late_fraction=sum(r["late"] for r in rows)/max(1, sum(r["decisions"] for r in rows)),
                max_ms=max(r["max_ms"] for r in rows))


def paired_report(candidate, reference):
    diffs = [a["score"]-b["score"] for a, b in zip(candidate, reference)]
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs)/math.sqrt(len(diffs)) if len(diffs) > 1 else 0
    groups = {}
    for kind in SCENARIOS:
        a = [r for r in candidate if r["scenario"] == kind]
        b = [r for r in reference if r["scenario"] == kind]
        if a:
            groups[kind] = dict(candidate=summary(a), reference=summary(b))
    # Approximate paired 95% interval, not a guarantee of real-server gains.
    return dict(mean_difference=mean, standard_error=se, ci95=[mean-1.96*se, mean+1.96*se],
                candidate=summary(candidate), reference=summary(reference), scenarios=groups)


def policy(weights):
    return dict(version=VERSION, features=FEATURES, actions=ACTIONS, weights=weights)


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def map_jobs(pool, jobs):
    return list(pool.map(episode, jobs, chunksize=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000, help="Total training episodes, including resumed ones")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--horizon", type=float, default=20)
    ap.add_argument("--eval-seeds", type=int, default=36)
    ap.add_argument("--validate-every", type=int, default=5, help="Batches between validation comparisons")
    ap.add_argument("--test", action="store_true", help="Evaluate saved champion on disjoint test seeds; does not train")
    args = ap.parse_args()
    if min(args.batch, args.workers, args.eval_seeds, args.validate_every) < 1 or args.horizon <= 0:
        ap.error("Counts and horizon must be positive")
    if args.episodes < 1 or args.episodes >= SPLITS["validation"]-SPLITS["train"]:
        ap.error("Episode count is outside supported training seed range")
    args.run.mkdir(parents=True, exist_ok=True)
    state_path = args.run/"state.json"
    settings = dict(horizon=args.horizon, batch=args.batch, eval_seeds=args.eval_seeds,
                    validate_every=args.validate_every, algorithm="linear-reinforce-v1")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["settings"] != settings:
            raise ValueError("Resume settings differ; use matching arguments or a new --run directory")
    else:
        if args.test:
            raise ValueError("No saved run to test")
        state = dict(settings=settings, completed=0, batches=0, weights=initial_weights(),
                     value=[0.0]*len(FEATURES), champion=None, champion_report=None)
    baseline_path = args.run/"baseline_fingerprint.json"
    fingerprint = hashlib.sha256()
    for p in sorted(ROOT/"brain"/n for n in ("__init__.py", "brain.py", "defense.py", "geom.py", "nav.py", "world.py"))+[(ROOT/"brain"/"champion.json"), Path(__file__), ROOT/"tools"/"sim.py"]:
        fingerprint.update(p.read_bytes())
    digest = fingerprint.hexdigest()
    if baseline_path.exists() and json.loads(baseline_path.read_text())["sha256"] != digest:
        raise ValueError("Source changed since this run began; start a new run to keep comparisons valid")
    atomic_json(baseline_path, dict(sha256=digest))
    atomic_json(args.run/"candidate.json", policy(state["weights"]))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as pool:
        split = "test" if args.test else "validation"
        seeds = [SPLITS[split]+i for i in range(args.eval_seeds)]
        print(json.dumps(dict(event="evaluation_baseline", split=split, episodes=len(seeds))), flush=True)
        reference = map_jobs(pool, [(None, seed, args.horizon, False) for seed in seeds])
        if args.test:
            if state["champion"] is None:
                print(json.dumps(dict(event="no_promoted_policy",
                    message="No policy passed validation; testing current candidate without activating it.")), flush=True)
            weights = state["champion"] if state["champion"] is not None else state["weights"]
            rows = map_jobs(pool, [(weights, seed, args.horizon, False) for seed in seeds])
            report = paired_report(rows, reference)
            report["split"] = "test"
            report["promoted_policy"] = state["champion"] is not None
            atomic_json(args.run/"test_report.json", report)
            print(json.dumps(report), flush=True)
            return
        while state["completed"] < args.episodes:
            count = min(args.batch, args.episodes-state["completed"])
            seeds_train = [SPLITS["train"]+state["completed"]+i for i in range(count)]
            rows = map_jobs(pool, [(state["weights"], seed, args.horizon, True) for seed in seeds_train])
            state["weights"], state["value"], samples = learn(state["weights"], state["value"], rows)
            state["completed"] += count
            state["batches"] += 1
            record = dict(event="train", completed=state["completed"], decisions=samples,
                          metrics=summary(rows), elapsed_s=round(time.monotonic()-started, 1),
                          actions={a: sum(r["actions"].get(a, 0) for r in rows) for a in ACTIONS})
            if state["batches"] % args.validate_every == 0 or state["completed"] == args.episodes:
                candidates = map_jobs(pool, [(state["weights"], seed, args.horizon, False) for seed in seeds])
                report = paired_report(candidates, reference)
                previous = state["champion_report"]
                # Candidate must beat baseline with a positive lower confidence bound,
                # avoid >2% late decisions, and improve the previous selection score.
                promote = (report["ci95"][0] > 0 and report["candidate"]["late_fraction"] <= 0.02
                           and (previous is None or report["mean_difference"] > previous["mean_difference"]))
                if promote:
                    state["champion"] = copy.deepcopy(state["weights"])
                    state["champion_report"] = report
                    atomic_json(args.run/"champion.json", dict(**policy(state["champion"]), validation=report))
                atomic_json(args.run/"validation_report.json", report)
                record["validation"] = report
                record["promoted"] = promote
            atomic_json(state_path, state)
            atomic_json(args.run/"candidate.json", policy(state["weights"]))
            with (args.run/"history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
    print(json.dumps(dict(event="complete", completed=state["completed"],
                          champion_promoted=state["champion"] is not None,
                          elapsed_s=round(time.monotonic()-started, 1))), flush=True)


if __name__ == "__main__":
    main()
