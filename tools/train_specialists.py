"""Independently train and evaluate each of the bot's eight decision modules.

  python tools/train_specialists.py --run runs/specialists-v1 --episodes 1008 --workers 4
  python tools/train_specialists.py --run runs/specialists-v1 --modules combat --episodes 2016
  python tools/train_specialists.py --run runs/specialists-v1 --test --eval-seeds 48
  python tools/train_specialists.py --run runs/specialists-v1 --status

One module varies at a time; all other modules and opponents use the frozen current bot.
Policies use episodic REINFORCE with a learned linear value baseline.
Validation selects candidates; disjoint test scenarios only report results.
Training rewards are task-specific proxies, not official placement scores.
CPU-only, standard library. No production policy is automatically activated.
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
from brain.specialists import MODULES, FEATURES, initial_weights, policy_document, specialist_class
from brain.brain import Brain as Current
from tools import sim
from tools.train_defense import scenario as defensive_scenario, worker_init, atomic_json

HORIZONS = dict(perception=30, targeting=20, strategy=20, navigation=25, dodging=12, combat=20, economy=30, zone=25)
ALGORITHM = "specialists-reinforce-v1"
MODULE_ORDER = list(MODULES)


class Reference(Current):
    params = {"debug": False}
    def __init__(self, hello):
        super().__init__(hello)
        self.errors = 0
    def act(self, obs):
        try:
            return super().act(obs)
        except Exception:
            self.errors += 1
            raise


def scenario(module, seed, hero_cls, horizon):
    rng = random.Random(seed)
    # Transform seeds without overlap: each external seed determines one world.
    if module == "targeting":
        world_seed = seed*6+4
    elif module == "dodging":
        world_seed = seed*6+(2 if seed % 2 else 3)
    else:
        world_seed = seed
    s, kind, _ = defensive_scenario(world_seed, hero_cls, horizon)
    hero = s.mechs[0]
    if module == "perception":
        s.cfg["vision"]["memory_s"] = [0.3, 1, 5, 8][seed % 4]
        s.cfg["vision"]["near_range"] = 25
        kind = "memory_"+str(s.cfg["vision"]["memory_s"])
    if module in ("navigation", "zone"):
        goal = s._open_point(hero.pos[:2], 140)
        radius = 20 if module == "navigation" else [18, 35, 55][seed % 3]
        s.zphases = [dict(c=hero.pos[:2], nc=goal[:2], r=190.0, nr=radius,
                         s0=1.0, s1=max(3, horizon*0.65), dps=45.0, t0=0.0)]
        kind = module+"_"+str(seed % 3)
        # Some routes are uncontested; other routes must be completed under fire.
        if seed % 3 == 0:
            for enemy in s.mechs[1:]:
                enemy.pos = s._open_point(hero.pos[:2], 230)
    if module in ("combat", "targeting"):
        for i, m in enumerate(s.mechs):
            kind_w = [None, "laser", "mortar", "lightning"][(seed+i) % 4]
            if kind_w:
                m.weapons[1] = sim.Weapon(kind_w, m.wstats[kind_w])
            m.hp = m.max_hp*rng.uniform(0.35, 1)
        kind = module+"_"+str(seed % 4)
    if module == "economy":
        # Vary both immediate and delayed choices. Some resource routes are contested.
        for _ in range(8):
            s.cubes[s._nid("cube")] = dict(pos=s._open_point(hero.pos[:2], 75))
        for k in ("laser", "mortar", "lightning"):
            s.pickups[s._nid("pickup")] = dict(kind=k, pos=s._open_point(hero.pos[:2], 90))
        for _ in range(4):
            s.crates[s._nid("crate")] = dict(pos=s._open_point(hero.pos[:2], 70), hp=s.cfg["crates"]["hp"])
        s.drops[s._nid("drop")] = dict(pos=s._open_point(hero.pos[:2], 90), lands=2.0, progress=0.0, holder=None)
        for _ in range(seed % 6):
            s.add_cube(hero)
        s.make_offer(hero)
        if seed % 2 == 0:
            for enemy in s.mechs[1:]:
                enemy.pos = s._open_point(hero.pos[:2], 220)
        kind = "resources_"+str(seed % 3)
    return s, kind


def snapshot(s):
    m = s.mechs[0]
    goal = s.zone()["next_center"]
    return dict(t=s.t, hp=max(0, m.hp), max_hp=m.max_hp, energy=m.energy,
                dealt=m.diag["dealt"], taken=m.diag["taken"], out=m.diag["out_s"],
                stuck=m.diag["stuck_s"], cubes=m.cubes, upgrades=len(m.upgrades),
                second=int(m.weapons[1] is not None), dist=math.dist(m.pos[:2], goal[:2]),
                kills=m.kills)


def memory_quality(s):
    m = s.mechs[0]
    tracks = getattr(m.bot, "tracks", {})
    # Ground truth is used ONLY as a training/evaluation label, never as policy input.
    near = [e for e in s.mechs[1:] if e.alive and math.dist(e.pos[:2], m.pos[:2]) < 180]
    if not near:
        return 0.0
    quality = sum(max(0.0, 1-math.dist(tracks[e.id].pos[:2], e.pos[:2])/100)
                  if e.id in tracks else 0.0 for e in near)/len(near)
    stale = sum(not s.mechs[k].alive for k in tracks if 0 <= k < len(s.mechs))
    return max(-1.0, quality-0.25*stale)


def reward_delta(module, before, now, quality):
    dt = max(0.0, now["t"]-before["t"])
    # Normalize health before/after separately, so upgrades cannot inflate health reward.
    reward = 3*(now["hp"]/now["max_hp"]-before["hp"]/before["max_hp"])+0.04*dt
    reward -= 0.15*(now["out"]-before["out"])
    if module in ("combat", "targeting"):
        reward += 2*(now["dealt"]-before["dealt"])/1400+0.3*(now["kills"]-before["kills"])
    if module == "economy":
        reward += 0.35*(now["cubes"]-before["cubes"])+0.2*(now["upgrades"]-before["upgrades"])+0.2*(now["second"]-before["second"])
    if module in ("navigation", "zone"):
        reward += 0.02*(before["dist"]-now["dist"])-0.04*(now["stuck"]-before["stuck"])
    if module == "perception":
        reward += 0.12*quality*dt
    return reward


def episode(job):
    module, weights, seed, horizon, training = job
    cls = specialist_class(module, weights, training) if weights is not None else Reference
    s, kind = scenario(module, seed, cls, horizon)
    previous = snapshot(s)
    total, accuracy, accuracy_count, reached = 0.0, 0.0, 0, False
    def recorder(world):
        nonlocal previous, total, accuracy, accuracy_count, reached
        current = snapshot(world)
        q = memory_quality(world) if module == "perception" else 0.0
        accuracy += q
        accuracy_count += 1
        reached = reached or current["dist"] < 20
        reward = reward_delta(module, previous, current, q)
        total += reward
        bot = world.mechs[0].bot
        trajectory = getattr(bot, "trajectory", [])
        if trajectory:
            trajectory[-1]["reward"] += reward
        previous = current
    s.recorder = recorder
    s.run(max_t=horizon, stop_when=lambda world: not world.mechs[0].alive)
    m = s.mechs[0]
    errors = sum(getattr(x.bot, "errors", 0) for x in s.mechs)
    if errors:
        raise RuntimeError(f"{module}: {errors} controller errors on seed {seed}")
    terminal = 2+m.hp/m.max_hp if m.alive else -3.0
    if module in ("navigation", "zone") and reached:
        terminal += 1.0
    total += terminal
    trajectory = getattr(m.bot, "trajectory", [])
    if trajectory:
        trajectory[-1]["reward"] += terminal
    return dict(seed=seed, scenario=kind, score=total, alive=m.alive, hp=max(0,m.hp)/m.max_hp,
                damage=m.diag["taken"], dealt=m.diag["dealt"], survival_s=s.t,
                zone_s=m.diag["out_s"], stuck_s=m.diag["stuck_s"], cubes=m.cubes,
                upgrades=len(m.upgrades), reached=reached, memory_quality=accuracy/max(1,accuracy_count),
                trajectory=trajectory, actions=getattr(m.bot, "action_counts", {}),
                max_ms=max(m.act_ms, default=0), late=sum(t > 50 for t in m.act_ms),
                decisions=len(m.act_ms))


def learn(weights, value, episodes, lr=0.25, gamma=0.99):
    samples = []
    for ep in episodes:
        ret = 0.0
        for step in reversed(ep["trajectory"]):
            ret = step["reward"]+gamma*ret
            estimate = sum(a*b for a,b in zip(value, step["x"]))
            samples.append((step, ret-estimate))
    if not samples:
        raise RuntimeError("No learning transitions collected")
    scale = max(1.0, statistics.pstdev(a for _,a in samples))
    grad = [[0.0]*len(FEATURES) for _ in range(5)]
    vg = [0.0]*len(FEATURES)
    choices = 0
    for step, advantage in samples:
        advantage = max(-5.0,min(5.0,advantage))
        # Forced baseline actions must not dilute gradients for real decisions.
        if sum(p > 0 for p in step["p"]) > 1:
            choices += 1
            for i in range(5):
                coefficient = (float(i == step["action"])-step["p"][i])*advantage/scale
                for j,x in enumerate(step["x"]):
                    grad[i][j] += coefficient*x
        for j,x in enumerate(step["x"]):
            vg[j] += advantage*x
    denominator = max(1, choices)
    norm = math.sqrt(sum((v/denominator)**2 for row in grad for v in row))
    updated = [[max(-12,min(12,w+lr*g/denominator/max(1,norm))) for w,g in zip(row,gr)]
               for row,gr in zip(weights,grad)]
    value = [max(-12,min(12,v+0.01*g/len(samples))) for v,g in zip(value,vg)]
    return updated, value, len(samples), choices


def summary(rows):
    keys = ["score","alive","hp","damage","dealt","survival_s","zone_s","stuck_s","cubes","upgrades","reached","memory_quality"]
    result = {k:statistics.mean(r[k] for r in rows) for k in keys}
    result.update(episodes=len(rows), late_fraction=sum(r["late"] for r in rows)/max(1,sum(r["decisions"] for r in rows)),
                  max_ms=max(r["max_ms"] for r in rows))
    return result


def paired(candidate, reference):
    if [r["seed"] for r in candidate] != [r["seed"] for r in reference]:
        raise ValueError("Paired evaluation seeds do not match")
    diffs = [a["score"]-b["score"] for a,b in zip(candidate,reference)]
    mean = statistics.mean(diffs)
    se = statistics.stdev(diffs)/math.sqrt(len(diffs)) if len(diffs)>1 else 0
    groups = {}
    for name in sorted({r["scenario"] for r in candidate}):
        groups[name] = dict(candidate=summary([r for r in candidate if r["scenario"]==name]),
                            baseline=summary([r for r in reference if r["scenario"]==name]))
    return dict(mean_difference=mean, ci95=[mean-1.96*se,mean+1.96*se],
                candidate=summary(candidate), baseline=summary(reference), scenarios=groups)


def source_fingerprint():
    paths = [ROOT/"brain"/n for n in ("__init__.py","brain.py","champion.json","geom.py","nav.py","world.py","defense.py","specialists.py")]
    paths += [ROOT/"tools"/n for n in ("train_specialists.py","train_defense.py","sim.py")]
    paths += [ROOT/"mechfight-kit"/"check"/"stream.jsonl.gz"]
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def jobs(pool, module, weights, seeds, horizon, training=False):
    return list(pool.map(episode, [(module,weights,s,horizon,training) for s in seeds], chunksize=1))


def seed_set(module, split, n, start=0):
    offset = MODULE_ORDER.index(module)*100000
    base = {"train":100000,"validation":10000000,"test":30000000}[split]
    if n+start > 90000:
        raise ValueError("Too many episodes for disjoint seed ranges")
    return [base+offset+start+i for i in range(n)]


def status(run):
    rows = {}
    for module in MODULES:
        path = run/module/"state.json"
        if path.exists():
            s = json.loads(path.read_text())
            rows[module] = dict(completed=s["completed"], promoted=s["champion"] is not None,
                                validation_difference=(s.get("champion_report") or {}).get("mean_difference"))
        else:
            rows[module] = dict(completed=0,promoted=False)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--modules",nargs="+",choices=list(MODULES),default=list(MODULES))
    ap.add_argument("--episodes",type=int,default=1008,help="Total per module, including resumed episodes")
    ap.add_argument("--batch",type=int,default=24)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--eval-seeds",type=int,default=24)
    ap.add_argument("--validate-every",type=int,default=7)
    ap.add_argument("--test",action="store_true")
    ap.add_argument("--status",action="store_true")
    args=ap.parse_args()
    if args.status:
        print(json.dumps(status(args.run),indent=2))
        return
    if min(args.episodes,args.batch,args.workers,args.eval_seeds,args.validate_every)<1:
        ap.error("All counts must be positive")
    if args.eval_seeds<2:
        ap.error("At least two evaluation seeds are required")
    fingerprint=source_fingerprint()
    started=time.monotonic()
    report_all={}
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        for module in args.modules:
            directory=args.run/module
            directory.mkdir(parents=True,exist_ok=True)
            state_path=directory/"state.json"
            settings=dict(batch=args.batch,horizon=HORIZONS[module],validation_seeds=args.eval_seeds,
                          validate_every=args.validate_every,algorithm=ALGORITHM)
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state["source_fingerprint"]!=fingerprint:
                    raise ValueError("Source changed; start a new --run directory")
                if not args.test and state["settings"]!=settings:
                    raise ValueError("Resume arguments changed; use original settings or a new run")
            else:
                if args.test:
                    raise ValueError(f"No training checkpoint for {module}")
                state=dict(module=module,settings=settings,source_fingerprint=fingerprint,completed=0,batches=0,
                           weights=initial_weights(),value=[0.0]*len(FEATURES),champion=None,champion_report=None)
            if not args.test and state["completed"]>=args.episodes:
                print(json.dumps(dict(event="already_complete",module=module,completed=state["completed"])),flush=True)
                continue
            split="test" if args.test else "validation"
            seeds=seed_set(module,split,args.eval_seeds)
            print(json.dumps(dict(event="baseline",module=module,split=split,episodes=len(seeds))),flush=True)
            reference=jobs(pool,module,None,seeds,HORIZONS[module])
            if args.test:
                weights=state["champion"] if state["champion"] is not None else state["weights"]
                candidate=jobs(pool,module,weights,seeds,HORIZONS[module])
                report=paired(candidate,reference)
                report.update(module=module,split="test",promoted=state["champion"] is not None,
                              model="champion" if state["champion"] is not None else "unpromoted_candidate")
                atomic_json(directory/"test_report.json",report)
                report_all[module]=report
                print(json.dumps(dict(event="test",module=module,difference=report["mean_difference"],
                           ci95=report["ci95"],survival=report["candidate"]["alive"],
                           baseline_survival=report["baseline"]["alive"])),flush=True)
                continue
            while state["completed"]<args.episodes:
                count=min(args.batch,args.episodes-state["completed"])
                train_seeds=seed_set(module,"train",count,state["completed"])
                rows=jobs(pool,module,state["weights"],train_seeds,HORIZONS[module],True)
                state["weights"],state["value"],samples,choices=learn(state["weights"],state["value"],rows)
                state["completed"]+=count
                state["batches"]+=1
                record=dict(event="train",module=module,completed=state["completed"],samples=samples,choices=choices,
                            metrics=summary(rows),actions={a:sum(r["actions"].get(a,0) for r in rows) for a in MODULES[module]},
                            elapsed_s=round(time.monotonic()-started,1))
                if state["batches"]%args.validate_every==0 or state["completed"]==args.episodes:
                    candidate=jobs(pool,module,state["weights"],seeds,HORIZONS[module])
                    report=paired(candidate,reference)
                    previous=state["champion_report"]
                    promote=(report["ci95"][0]>0 and report["candidate"]["late_fraction"]<=0.02
                             and report["candidate"]["alive"]>=report["baseline"]["alive"]-1/len(seeds)
                             and (previous is None or report["mean_difference"]>previous["mean_difference"]))
                    if promote:
                        state["champion"]=copy.deepcopy(state["weights"])
                        state["champion_report"]=report
                        atomic_json(directory/"champion.json",dict(**policy_document(module,state["champion"]),validation=report))
                    atomic_json(directory/"validation_report.json",report)
                    record.update(validation=report,promoted=promote)
                atomic_json(state_path,state)
                atomic_json(directory/"candidate.json",policy_document(module,state["weights"]))
                with (directory/"history.jsonl").open("a",encoding="utf-8") as f:
                    f.write(json.dumps(record)+"\n")
                compact={k:v for k,v in record.items() if k not in ("validation","metrics","actions")}
                if "validation" in record:
                    compact["validation_difference"]=record["validation"]["mean_difference"]
                    compact["ci95"]=record["validation"]["ci95"]
                print(json.dumps(compact),flush=True)
            print(json.dumps(dict(event="module_complete",module=module,completed=state["completed"],
                            promoted=state["champion"] is not None)),flush=True)
    if args.test:
        atomic_json(args.run/"test_summary.json",report_all)
    print(json.dumps(dict(event="suite_complete",elapsed_s=round(time.monotonic()-started,1),status=status(args.run))),flush=True)


if __name__=="__main__":
    main()
