"""Evolutionary tournament over the bot's TUNING knobs (CPU, local simulator). Resumable and monitored.

    python tools/evolve.py --run C:/Users/nvive/mechrl/evo/run1            # start or resume
    python tools/evolve.py --run ... --status                              # print progress
    python tools/evolve.py --run ... --final-test                          # champion vs start on the untouched test set

Each generation:
  1. screen    every candidate on the same squad seeds (practice-like tiered hunters) + mixed free-for-all
               matches (4 candidates x2 copies, reference x4, old bots x4 in one match)
  2. race      the best quarter get more squad seeds and more free-for-all rounds
  3. select    elites by shrunk mean fitness, kept diverse (minimum genome distance)
  4. challenge the best non-champion elite vs the champion on the fixed VALIDATION seeds, paired by seed;
               promote only if the gain is reliable (z > --promote-z) and the latency check passes
  5. breed     offspring by uniform crossover + mutation of elites

Seed sets never overlap: training (per-generation, fresh), validation (fixed, used only for challenges),
test (untouched until --final-test). Fitness = 0.5 * mean squad points + 0.5 * mean free-for-all points.
The simulator approximates the real game: treat promotions as candidates for real practice, not truth.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ----------------------------------------------------------------------------- search space
F, B, P = "float", "bool", "perm"
SPACE: List[Tuple[str, str, Any, Any]] = [
    ("disengage_ratio", F, 0.4, 1.2), ("cover_hp_frac", F, 0.15, 0.7), ("reengage_hp_frac", F, 0.5, 1.0),
    ("heal_below_frac", F, 0.0, 0.9), ("chase", B, None, None), ("finish_hp_frac", F, 0.1, 0.5),
    ("two_front", B, None, None), ("farm_quiet_s", F, 0.0, 15.0),
    ("fight_range.gun", F, 30.0, 130.0), ("fight_range.laser", F, 60.0, 200.0),
    ("fight_range.mortar", F, 60.0, 180.0), ("fight_range.lightning", F, 15.0, 40.0),
    ("strafe_period.0", F, 0.2, 1.0), ("strafe_period.1", F, 0.6, 2.5),
    ("fight_boost_energy", F, 30.0, 130.0), ("boost_min_energy", F, 10.0, 60.0),
    ("mortar_min_range", F, 18.0, 40.0), ("loot_radius", F, 30.0, 150.0), ("cube_radius", F, 30.0, 150.0),
    ("crate_radius", F, 40.0, 200.0), ("early_farm_mult", F, 1.0, 3.0), ("early_farm_s", F, 0.0, 150.0),
    ("npc_farm_min_hp_frac", F, 0.4, 1.01), ("drop_max_dist", F, 0.0, 300.0),
    ("drop_enemy_clear", F, 30.0, 150.0), ("drop_min_hp_frac", F, 0.0, 0.9),
    ("zone_inner_frac", F, 0.2, 0.8), ("zone_slack_s", F, 5.0, 40.0), ("safe_after_s", F, 1.0, 8.0),
    ("reload_quiet_s", F, 0.5, 5.0), ("lightning_min_energy", F, 5.0, 40.0),
    ("tgt_hp_w", F, 0.0, 200.0), ("tgt_attacker_bonus", F, 0.0, 150.0), ("tgt_sticky", F, 0.0, 100.0),
    ("aim_tol_mult", F, 0.5, 1.6),
    ("weapon_order", P, ["laser", "mortar", "lightning"], None),
    ("upgrade_priority", P, None, None),
]


def default_genome() -> Dict[str, Any]:
    from brain.brain import TUNING
    g: Dict[str, Any] = {}
    for name, kind, lo, hi in SPACE:
        if name == "weapon_order":
            g[name] = sorted(["laser", "mortar", "lightning"], key=lambda k: -TUNING["weapon_rank"][k])
        elif name == "upgrade_priority":
            g[name] = list(TUNING["upgrade_priority"])
        elif "." in name:
            k, sub = name.split(".")
            v = TUNING[k][int(sub)] if k == "strafe_period" else TUNING[k][sub]
            g[name] = float(v)
        else:
            v = TUNING[name]
            g[name] = bool(v) if kind == B else float(min(max(v, lo), hi)) if kind == F else v
    return g


def to_params(g: Dict[str, Any]) -> Dict[str, Any]:
    """Genome -> TUNING overrides understood by brain.Brain."""
    p: Dict[str, Any] = {"fight_range": {}, "strafe_period": [0.0, 0.0]}
    for name, kind, lo, hi in SPACE:
        v = g[name]
        if name == "weapon_order":
            p["weapon_rank"] = {k: 3 - i for i, k in enumerate(v)}
            p["weapon_rank"]["gun"] = 0
        elif name == "upgrade_priority":
            p["upgrade_priority"] = list(v)
        elif name.startswith("fight_range."):
            p["fight_range"][name.split(".")[1]] = v
        elif name.startswith("strafe_period."):
            p["strafe_period"][int(name.split(".")[1])] = v
        else:
            p[name] = v
    lo, hi = p["strafe_period"]
    p["strafe_period"] = (min(lo, hi - 0.1), max(hi, lo + 0.1))
    if p["fight_boost_energy"] > 100:
        p["fight_boost_energy"] = 999.0  # off
    return p


def gid(g: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(g, sort_keys=True).encode()).hexdigest()[:10]


def mutate(g: Dict[str, Any], rng: random.Random, rate: float = 0.25, scale: float = 0.15) -> Dict[str, Any]:
    c = copy.deepcopy(g)
    changed = False
    while not changed:
        for name, kind, lo, hi in SPACE:
            if kind == F and rng.random() < rate:
                c[name] = float(min(hi, max(lo, c[name] + rng.gauss(0, scale * (hi - lo)))))
                changed = True
            elif kind == B and rng.random() < rate * 0.4:
                c[name] = not c[name]
                changed = True
            elif kind == P and rng.random() < rate * 0.6:
                v = c[name]
                i, j = rng.randrange(len(v)), rng.randrange(len(v))
                v.insert(j, v.pop(i))  # move one item
                changed = True
    return c


def crossover(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    return {k: copy.deepcopy(a[k] if rng.random() < 0.5 else b[k]) for k in a}


def distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    d = 0.0
    for name, kind, lo, hi in SPACE:
        if kind == F:
            d += abs(a[name] - b[name]) / (hi - lo)
        elif kind == B:
            d += 0.5 * (a[name] != b[name])
        else:
            d += 0.5 * (a[name] != b[name])
    return d


# ----------------------------------------------------------------------------- match running (workers)
_HELLO = None


def _bot_class(spec: str, params: Optional[dict], label: str):
    from tools import sim
    base = sim.import_bot(spec)
    if params is None:
        return base
    return type(f"Evo_{label}", (base,), {"params": params, "name": f"evo-{label}"[:32]})


def play(job: dict) -> dict:
    """Run one match. job: {kind, seed, slots:[{spec, params, label}], tiers?, hero?}. Returns per-label points."""
    global _HELLO
    from tools import sim
    if _HELLO is None:
        _HELLO = sim.load_hello()
    rng = random.Random(job["seed"])
    slots = job["slots"]
    t0 = time.time()
    if job["kind"] == "squad":
        hero = slots[0]
        tiers = job["tiers"]  # 4 specs: easy, medium, hard, expert (each {spec, params, label})
        tier_of = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 3
        hero_slot = rng.randrange(16)
        order = [i for i in range(16) if i != hero_slot]
        rng.shuffle(order)
        bots: List[Any] = [None] * 16
        labels: List[str] = [""] * 16
        mt = {}
        for k, i in enumerate(order):
            tdef = tiers[tier_of[k]]
            bots[i] = _bot_class(tdef["spec"], tdef.get("params"), tdef["label"])
            labels[i] = tdef["label"]
            mt[i] = tier_of[k]
        bots[hero_slot] = _bot_class(hero["spec"], hero.get("params"), hero["label"])
        labels[hero_slot] = hero["label"]
        teams = [None if i == hero_slot else 1 for i in range(16)]
        tracked = {hero_slot}
    else:  # ffa / duel: slots list as given, shuffled into positions
        idx = list(range(len(slots)))
        rng.shuffle(idx)
        bots = [_bot_class(slots[i]["spec"], slots[i].get("params"), slots[i]["label"]) for i in idx]
        labels = [slots[i]["label"] for i in idx]
        teams = [None] * len(slots)
        mt = {}
        tracked = {i for i in range(len(slots)) if slots[idx[i]].get("track")}
    S = sim.Sim(_HELLO, bots, teams, job["seed"])
    S.tier = mt
    res = S.run()
    out: Dict[str, List[float]] = {}
    kills: Dict[str, List[int]] = {}
    ms: Dict[str, float] = {}
    n = len(bots)
    for mid, place, k, team, cubes in res:
        if mid not in tracked:
            continue
        lab = labels[mid]
        out.setdefault(lab, []).append(float((16 if n == 16 else n) - place) if n == 16 else float(place == 1))
        kills.setdefault(lab, []).append(k)
        m = S.mechs[mid]
        ms[lab] = max(ms.get(lab, 0.0), max(m.act_ms) if m.act_ms else 0.0)
    return {"key": job["key"], "kind": job["kind"], "seed": job["seed"], "pts": out, "kills": kills, "ms": ms,
            "wall": time.time() - t0, "end_t": S.t}


# ----------------------------------------------------------------------------- the tournament
class Evo:
    def __init__(self, run_dir: str, workers: int, pop: int, elites: int, promote_z: float):
        self.dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.workers, self.P, self.MU, self.promote_z = workers, pop, elites, promote_z
        self.state_path = os.path.join(run_dir, "state.json")
        self.eval_path = os.path.join(run_dir, "evaluations.jsonl")
        self.log_path = os.path.join(run_dir, "evolve.log")
        self.status_path = os.path.join(run_dir, "status.json")
        self.results: Dict[str, dict] = {}  # job key -> result
        if os.path.exists(self.eval_path):
            with open(self.eval_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.results[r["key"]] = r
                    except ValueError:
                        pass
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.st = json.load(f)
            self.log(f"resumed at generation {self.st['gen']} phase {self.st['phase']} "
                     f"({len(self.results)} cached match results)")
        else:
            g0 = default_genome()
            self.st = {"gen": 0, "phase": "screen", "genomes": {gid(g0): g0}, "population": [gid(g0)],
                       "champion": gid(g0), "reference": gid(g0), "history": [], "promotions": [],
                       "created": time.time(), "workers": workers}
            self.log(f"new run: reference/champion = {gid(g0)} (current brain defaults)")
            self.save()
        self.t_start = time.time()
        self.busy = 0.0
        self.done_now = 0

    # -------------------------------------------------- bookkeeping
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def save(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.st, f)
        os.replace(tmp, self.state_path)

    def status(self, **extra) -> None:
        el = time.time() - self.t_start
        s = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "gen": self.st["gen"], "phase": self.st["phase"],
             "champion": self.st["champion"], "promotions": len(self.st["promotions"]),
             "matches_cached": len(self.results), "matches_this_session": self.done_now,
             "matches_per_hour": round(self.done_now / max(el, 1) * 3600, 1),
             "worker_busy_frac": round(self.busy / max(el * self.workers, 1e-9), 3),
             "uptime_h": round(el / 3600, 2), "pid": os.getpid()}
        s.update(extra)
        with open(self.status_path, "w") as f:
            json.dump(s, f, indent=1)

    def G(self, h: str) -> Dict[str, Any]:
        return self.st["genomes"][h]

    def slot(self, h: str, track: bool = False) -> dict:
        return {"spec": "brain", "params": to_params(self.G(h)), "label": h, "track": track}

    def tiers(self) -> List[dict]:
        ref = self.st["reference"]
        return [{"spec": "tools.baseline.brain_v1", "label": "v1"}, {"spec": "tools.baseline.brain_v1", "label": "v1"},
                {"spec": "tools.baseline.brain_v2", "label": "v2"},
                {"spec": "brain", "params": to_params(self.G(ref)), "label": "ref"}]

    def squad_job(self, h: str, seed: int) -> dict:
        return {"key": f"squad:{h}:{seed}", "kind": "squad", "seed": seed, "slots": [self.slot(h, True)],
                "tiers": self.tiers()}

    def ffa_job(self, group: List[str], seed: int) -> dict:
        """4 candidates x2 copies + reference x4 + v2 x2 + v1 x2: a mixed population, not mirror copies."""
        slots = []
        for h in group:
            slots += [self.slot(h, True), self.slot(h, True)]
        ref = self.st["reference"]
        slots += [{"spec": "brain", "params": to_params(self.G(ref)), "label": "ref"} for _ in range(4)]
        slots += [{"spec": "tools.baseline.brain_v2", "label": "v2"} for _ in range(2)]
        slots += [{"spec": "tools.baseline.brain_v1", "label": "v1"} for _ in range(2)]
        while len(slots) < 16:
            slots.append({"spec": "tools.baseline.brain_v2", "label": "v2"})
        return {"key": f"ffa:{'+'.join(group)}:{seed}", "kind": "ffa", "seed": seed, "slots": slots}

    def run_jobs(self, jobs: List[dict], phase: str) -> None:
        todo = [j for j in jobs if j["key"] not in self.results]
        if not todo:
            return
        self.log(f"gen {self.st['gen']} {phase}: {len(todo)} matches ({len(jobs) - len(todo)} cached)")
        t0 = time.time()
        fails: Dict[str, int] = {}
        for attempt in range(12):
            if not todo:
                break
            try:
                with ProcessPoolExecutor(self.workers) as ex, open(self.eval_path, "a") as f:
                    futs = {ex.submit(play, j): j["key"] for j in todo}
                    for k, fu in enumerate(as_completed(futs)):
                        try:
                            r = fu.result()
                        except BrokenProcessPool:
                            raise
                        except Exception as e:  # a bug in one match: record it, never stop the run
                            fails[futs[fu]] = fails.get(futs[fu], 0) + 1
                            self.log(f"match {futs[fu]} raised {e!r}")
                            continue
                        self.results[r["key"]] = r
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        self.busy += r["wall"]
                        self.done_now += 1
                        if k % 10 == 0:
                            self.status(phase_progress=f"{k + 1}/{len(todo)}",
                                        phase_eta_min=round((time.time() - t0) / (k + 1) * (len(todo) - k - 1) / 60, 1))
            except BrokenProcessPool:
                for j in todo:
                    if j["key"] not in self.results:
                        fails[j["key"]] = fails.get(j["key"], 0) + 1
                self.st["worker_deaths"] = self.st.get("worker_deaths", 0) + 1
                self.log(f"a worker process died (total {self.st['worker_deaths']}); retrying unfinished matches")
            todo = [j for j in todo if j["key"] not in self.results and fails.get(j["key"], 0) < 6]
        skipped = [j["key"] for j in jobs if j["key"] not in self.results]
        if skipped:
            self.st.setdefault("skipped", []).extend(skipped)
            self.log(f"skipping {len(skipped)} matches that kept failing (they count as missing data)")

    # -------------------------------------------------- statistics
    def samples(self, h: str, kind: str, seeds: Optional[set] = None) -> List[float]:
        out = []
        for r in self.results.values():
            if r["kind"] != kind or (seeds is not None and r["seed"] not in seeds):
                continue
            if r["seed"] >= VALID_BASE and seeds is None:
                continue  # validation/test results never feed selection
            out += r["pts"].get(h, [])
        return out

    def fitness(self, h: str) -> Tuple[float, int, float, float]:
        """Shrunk mean: 0.5 * squad + 0.5 * ffa, each pulled toward the population prior with 6 pseudo-matches."""
        sq, ff = self.samples(h, "squad"), self.samples(h, "ffa")
        k = 6.0
        sq_m = (sum(sq) + k * PRIOR_SQ) / (len(sq) + k)
        ff_m = (sum(ff) + k * PRIOR_FFA) / (len(ff) + k)
        return 0.5 * sq_m + 0.5 * ff_m, len(sq) + len(ff), sq_m, ff_m

    # -------------------------------------------------- one generation
    def generation(self) -> None:
        gen = self.st["gen"]
        rng = random.Random(1000 + gen)
        pop: List[str] = self.st["population"]
        if self.st["phase"] == "breed" or len(pop) < self.P:
            pop = self.breed(rng, pop)
            self.st["population"] = pop
            self.st["phase"] = "screen"
            self.save()
        base = TRAIN_BASE + gen * 1000
        if self.st["phase"] == "screen":
            jobs = [self.squad_job(h, base + s) for h in pop for s in range(SCREEN_SQUAD)]
            for rnd in range(SCREEN_FFA_ROUNDS):
                order = pop[:]
                random.Random(base + 500 + rnd).shuffle(order)
                for gi in range(0, len(order), 4):
                    group = order[gi:gi + 4]
                    if len(group) == 4:
                        jobs.append(self.ffa_job(group, base + 500 + rnd * 50 + gi))
            self.run_jobs(jobs, "screen")
            self.st["phase"] = "race"
            self.save()
        if self.st["phase"] == "race":
            ranked = sorted(pop, key=lambda h: -self.fitness(h)[0])
            top = ranked[:max(4, self.P // 4)]
            jobs = [self.squad_job(h, base + 100 + s) for h in top for s in range(RACE_SQUAD)]
            for rnd in range(RACE_FFA_ROUNDS):
                order = top[:]
                random.Random(base + 700 + rnd).shuffle(order)
                for gi in range(0, len(order), 4):
                    group = order[gi:gi + 4]
                    if len(group) == 4:
                        jobs.append(self.ffa_job(group, base + 700 + rnd * 50 + gi))
            self.run_jobs(jobs, "race")
            self.st["phase"] = "select"
            self.save()
        if self.st["phase"] == "select":
            elites = self.select(pop)
            self.st["elites"] = elites
            best = elites[0]
            fb, nb, sqb, ffb = self.fitness(best)
            fc, nc, sqc, ffc = self.fitness(self.st["champion"])
            self.st["history"].append({"gen": gen, "best": best, "best_fit": round(fb, 3), "best_n": nb,
                                       "best_sq": round(sqb, 3), "best_ffa": round(ffb, 3),
                                       "champ": self.st["champion"], "champ_fit": round(fc, 3),
                                       "pop_mean": round(statistics.mean(self.fitness(h)[0] for h in pop), 3),
                                       "time": time.time()})
            self.log(f"gen {gen} select: best {best} fit {fb:.3f} (squad {sqb:.2f}, ffa {ffb:.2f}, n={nb}); "
                     f"champion {self.st['champion']} fit {fc:.3f}")
            self.st["phase"] = "challenge"
            self.save()
        if self.st["phase"] == "challenge":
            cand = next((h for h in self.st["elites"] if h != self.st["champion"]), None)
            if cand and self.fitness(cand)[0] > self.fitness(self.st["champion"])[0] + CHALLENGE_MARGIN:
                self.challenge(cand)
            self.st["phase"] = "breed"
            self.st["gen"] = gen + 1
            self.st["population"] = list(self.st["elites"])
            self.save()
        self.status()

    def select(self, pop: List[str]) -> List[str]:
        cands = set(pop) | set(self.st.get("elites", [])) | {self.st["champion"]}
        ranked = sorted(cands, key=lambda h: -self.fitness(h)[0])
        out: List[str] = []
        for h in ranked:  # keep elites diverse: skip near-duplicates of a better elite
            if all(distance(self.G(h), self.G(o)) > 0.6 for o in out):
                out.append(h)
            if len(out) >= self.MU:
                break
        if self.st["champion"] not in out:
            out.append(self.st["champion"])
        return out

    def breed(self, rng: random.Random, parents: List[str]) -> List[str]:
        pop = list(dict.fromkeys(parents))
        while len(pop) < self.P:
            a = self.G(rng.choice(parents))
            if len(parents) > 1 and rng.random() < 0.4:
                b = self.G(rng.choice(parents))
                child = mutate(crossover(a, b, rng), rng, rate=0.12)
            else:
                child = mutate(a, rng)
            h = gid(child)
            if h not in self.st["genomes"]:
                self.st["genomes"][h] = child
            if h not in pop:
                pop.append(h)
        return pop

    # -------------------------------------------------- validation challenge (paired)
    def paired(self, a: str, b: str, seed_base: int, n_sq: int, n_ffa: int, phase: str) -> Tuple[float, float, dict]:
        jobs = [self.squad_job(h, seed_base + s) for h in (a, b) for s in range(n_sq)]
        jobs += [self.ffa_job([a, a, b, b], seed_base + 5000 + s) for s in range(n_ffa)]
        self.run_jobs(jobs, phase)
        diffs_sq, diffs_ffa = [], []
        R = self.results
        for s in range(n_sq):  # paired by seed; seeds missing on either side are dropped
            ka, kb = f"squad:{a}:{seed_base + s}", f"squad:{b}:{seed_base + s}"
            if ka in R and kb in R:
                diffs_sq.append(R[ka]["pts"][a][0] - R[kb]["pts"][b][0])
        for s in range(n_ffa):
            k = f"ffa:{a}+{a}+{b}+{b}:{seed_base + 5000 + s}"
            if k in R and a in R[k]["pts"] and b in R[k]["pts"]:
                diffs_ffa.append(statistics.mean(R[k]["pts"][a]) - statistics.mean(R[k]["pts"][b]))

        def mse(x):
            return (statistics.mean(x), statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else 9.9)
        msq, ssq = mse(diffs_sq)
        mff, sff = mse(diffs_ffa)
        diff = 0.5 * msq + 0.5 * mff
        se = 0.5 * math.sqrt(ssq ** 2 + sff ** 2)
        return diff, se, {"squad_diff": round(msq, 3), "squad_se": round(ssq, 3), "ffa_diff": round(mff, 3),
                          "ffa_se": round(sff, 3), "n_sq": n_sq, "n_ffa": n_ffa}

    def latency_ok(self, h: str) -> Tuple[bool, float]:
        """Replay the kit's recorded match through the candidate in-process; max act() must stay small."""
        import gzip
        from tools import sim
        import mechfight as mf
        hello = sim.load_hello()
        with gzip.open(os.path.join(ROOT, "mechfight-kit", "check", "stream.jsonl.gz"), "rt") as f:
            msgs = [json.loads(l) for l in f if l.strip()]
        cls = _bot_class("brain", to_params(self.G(h)), h)
        mf.draw.caps = hello["config"]["debug"]
        bot = cls(hello)
        worst = 0.0
        for m in msgs[1:]:
            if m.get("type") == "obs":
                mf.draw.clear()
                s = time.perf_counter()
                bot.act(m)
                worst = max(worst, (time.perf_counter() - s) * 1000)
        return worst < LATENCY_MAX_MS, worst

    def challenge(self, cand: str) -> None:
        champ = self.st["champion"]
        diff, se, det = self.paired(cand, champ, VALID_BASE, VALID_SQUAD, VALID_FFA, "challenge")
        z = diff / se if se > 0 else 0.0
        ok, worst = self.latency_ok(cand)
        rec = {"gen": self.st["gen"], "cand": cand, "champ": champ, "diff": round(diff, 3), "se": round(se, 3),
               "z": round(z, 2), "latency_ms": round(worst, 2), "latency_ok": ok, **det, "time": time.time()}
        if z > self.promote_z and ok:
            self.st["champion"] = cand
            self.st["promotions"].append(rec)
            self.export_champion(cand, rec)
            self.log(f"PROMOTED {cand} over {champ}: +{diff:.3f} pts (z={z:.2f}, latency {worst:.1f} ms) {det}")
        else:
            self.st.setdefault("rejections", []).append(rec)
            self.log(f"challenge {cand} vs {champ}: {diff:+.3f} pts (z={z:.2f}, latency {worst:.1f} ms) -> kept champion")

    def export_champion(self, h: str, rec: dict) -> None:
        out = {"genome_id": h, "params": to_params(self.G(h)), "evidence": rec, "run": self.dir}
        with open(os.path.join(self.dir, "champion.json"), "w") as f:
            json.dump(out, f, indent=1)
        arch = os.path.join(self.dir, "archive")
        os.makedirs(arch, exist_ok=True)
        with open(os.path.join(arch, f"champion_gen{self.st['gen']:03d}_{h}.json"), "w") as f:
            json.dump(out, f, indent=1)

    def final_test(self) -> None:
        a, b = self.st["champion"], self.st["reference"]
        if a == b:
            print("champion == reference; nothing to test")
            return
        diff, se, det = self.paired(a, b, TEST_BASE, TEST_SQUAD, TEST_FFA, "final-test")
        msg = (f"FINAL TEST (untouched seeds) champion {a} vs reference {b}: {diff:+.3f} pts "
               f"(95% CI {diff - 1.96 * se:+.3f} .. {diff + 1.96 * se:+.3f}) {det}")
        self.log(msg)
        with open(os.path.join(self.dir, "final_test.json"), "w") as f:
            json.dump({"champion": a, "reference": b, "diff": diff, "se": se, **det}, f, indent=1)


# seed sets (disjoint)
TRAIN_BASE, VALID_BASE, TEST_BASE = 100_000, 5_000_000, 9_000_000
SCREEN_SQUAD, SCREEN_FFA_ROUNDS = 12, 2
RACE_SQUAD, RACE_FFA_ROUNDS = 24, 3
VALID_SQUAD, VALID_FFA = 120, 30
TEST_SQUAD, TEST_FFA = 200, 50
PRIOR_SQ, PRIOR_FFA = 2.0, 7.5
LATENCY_MAX_MS = 15.0
CHALLENGE_MARGIN = 0.05  # challenge only when the best elite's training fitness beats the champion's by this


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elites", type=int, default=6)
    ap.add_argument("--promote-z", type=float, default=2.0)
    ap.add_argument("--generations", type=int, default=10_000)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="pipeline check: tiny seed counts, forced promotion")
    a = ap.parse_args()
    if a.smoke:
        global SCREEN_SQUAD, SCREEN_FFA_ROUNDS, RACE_SQUAD, RACE_FFA_ROUNDS, VALID_SQUAD, VALID_FFA, CHALLENGE_MARGIN
        SCREEN_SQUAD, SCREEN_FFA_ROUNDS, RACE_SQUAD, RACE_FFA_ROUNDS, VALID_SQUAD, VALID_FFA = 2, 1, 2, 1, 4, 2
        CHALLENGE_MARGIN = -99.0
        a.promote_z = -99.0
    if a.status:
        with open(os.path.join(a.run, "status.json")) as f:
            print(f.read())
        st = json.load(open(os.path.join(a.run, "state.json")))
        for h in st["history"][-10:]:
            print(h)
        return
    evo = Evo(a.run, a.workers, a.pop, a.elites, a.promote_z)
    if a.final_test:
        evo.final_test()
        return
    for _ in range(a.generations):
        evo.generation()


if __name__ == "__main__":
    main()
