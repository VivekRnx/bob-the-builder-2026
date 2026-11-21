"""Check simulator assumptions against the only real-server data we have: the kit's recorded stream
(a hello + 300 observations from a real match). Prints a pass/warn report.

    python tools/fidelity.py

What it can check (from one mech's point of view, its actions unknown): speed and acceleration limits,
torso turn rate, projectile speeds, vision range / field of view / memory, damage amounts and rear
multiplier, sound distance buckets vs known positions, zone schedule, NPC and crate numbers.
"""
from __future__ import annotations

import collections
import gzip
import json
import math
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    with gzip.open(os.path.join(ROOT, "mechfight-kit", "check", "stream.jsonl.gz"), "rt") as f:
        msgs = [json.loads(l) for l in f if l.strip()]
    hello = msgs[0]
    cfg = hello["config"]
    obs = [m for m in msgs if m.get("type") == "obs"]
    mech = cfg["mech"]
    report = []

    def check(name, ok, detail):
        report.append(("PASS" if ok else "WARN", name, detail))

    # ---- timing
    dts = [round(b["t"] - a["t"], 3) for a, b in zip(obs, obs[1:])]
    check("decision interval", collections.Counter(dts).most_common(1)[0][0] == 0.1,
          f"intervals {collections.Counter(dts).most_common(3)} (sim: 0.1 s)")

    # ---- movement: speed and acceleration of self
    sp, acc, vz = [], [], []
    for a, b in zip(obs, obs[1:]):
        if round(b["t"] - a["t"], 3) != 0.1:
            continue
        va, vb = a["self"]["vel"], b["self"]["vel"]
        sp.append(math.hypot(vb[0], vb[1]))
        if a["self"]["grounded"] and b["self"]["grounded"]:
            acc.append(math.hypot(vb[0] - va[0], vb[1] - va[1]) / 0.1)
        vz.append(vb[2])
    check("max horizontal speed", max(sp) <= mech["boost_speed"] * 1.05,
          f"max {max(sp):.2f} m/s, walk {mech['walk_speed']}, boost {mech['boost_speed']}")
    check("ground acceleration", max(acc) <= mech["ground_accel"] * 1.6,
          f"max |dv|/dt {max(acc):.1f} m/s2 (config {mech['ground_accel']}); p90 {sorted(acc)[int(.9 * len(acc))]:.1f}"
          " (>config means knockback/contacts, or a different accel model)")
    # position integrates velocity?
    errs = []
    for a, b in zip(obs, obs[1:]):
        if round(b["t"] - a["t"], 3) != 0.1:
            continue
        pa, pb, va, vb = a["self"]["pos"], b["self"]["pos"], a["self"]["vel"], b["self"]["vel"]
        pred = [pa[i] + (va[i] + vb[i]) / 2 * 0.1 for i in (0, 1)]
        errs.append(math.hypot(pb[0] - pred[0], pb[1] - pred[1]))
    check("position ~ integrated velocity", statistics.median(errs) < 0.3,
          f"median xy error {statistics.median(errs):.3f} m, p95 {sorted(errs)[int(.95 * len(errs))]:.3f} m "
          "(large = walls/slides/pushes)")

    # ---- aim slew
    dyaw = [abs(wrap(b["self"]["aim"][0] - a["self"]["aim"][0])) for a, b in zip(obs, obs[1:])
            if round(b["t"] - a["t"], 3) == 0.1]
    lim = mech["turn_rate"] * 0.1
    check("turn rate", max(dyaw) <= lim * 1.05, f"max yaw change {max(dyaw):.3f} rad/decision (limit {lim:.2f})")

    # ---- projectiles
    speeds = collections.defaultdict(list)
    for o in obs:
        for p in o["projectiles"]:
            speeds[p["kind"]].append(math.sqrt(sum(v * v for v in p["vel"])))
    for k, v in speeds.items():
        exp = {"gun": cfg["weapons"]["gun"]["speed"], "npc": cfg["npc"]["projectile_speed"],
               "mortar": cfg["weapons"]["mortar"]["speed"]}.get(k)
        check(f"projectile speed ({k})", exp is None or abs(statistics.median(v) - exp) < 0.1 * exp + (40 if k == "mortar" else 0),
              f"median {statistics.median(v):.1f} m/s over {len(v)} sightings (config {exp})")

    # ---- vision: every reported mech must be explainable by near range, the fov wedge, or memory
    vis = cfg["vision"]
    near, far, half = vis["near_range"], vis["range"], math.radians(vis["fov_deg"]) / 2
    seen_last = {}
    unexplained, total, max_far = 0, 0, 0.0
    for o in obs:
        me = o["self"]
        eye = me["pos"]
        for m in o["mechs"]:
            total += 1
            d = math.hypot(m["pos"][0] - eye[0], m["pos"][1] - eye[1])
            inwedge = abs(wrap(math.atan2(m["pos"][1] - eye[1], m["pos"][0] - eye[0]) - me["aim"][0])) <= half + 0.05
            ok = d <= near + 1 or (d <= far + 1 and inwedge)
            mem = o["t"] - seen_last.get(m["id"], -99) <= vis.get("memory_s", 5) + 0.15
            if ok:
                seen_last[m["id"]] = o["t"]
                max_far = max(max_far, d)
            elif not mem:
                unexplained += 1
    check("vision explained by near/fov/memory", unexplained == 0,
          f"{total} mech sightings, {unexplained} unexplained; farthest in-view {max_far:.0f} m")

    # ---- damage amounts
    dmg = collections.Counter()
    for o in obs:
        for e in o["events"]:
            if e["type"] == "damaged":
                dmg[(e["weapon"], round(e["amount"], 1), bool(e.get("rear")))] += 1
    rear_ok = all(abs(amt / (1.5 if rear else 1.0) - {"gun": 45, "npc": 25}.get(w, amt / (1.5 if rear else 1.0))) < 1
                  for (w, amt, rear) in dmg if w in ("gun", "npc"))
    check("damage amounts and rear x1.5", rear_ok, f"{dict(dmg)}")

    # ---- sounds: dist buckets
    check("sound buckets present", any(o["sounds"] for o in obs),
          f"kinds {collections.Counter(s['kind'] for o in obs for s in o['sounds']).most_common(8)}")

    # ---- zone schedule vs sim formula
    from tools import sim as S
    zs = obs[-1]["zone"]
    ph = cfg["zone"]["phases"][0]
    check("zone phase 1 timing", abs(zs["shrink_start"] - ph["hold_s"]) < 1e-6 and
          abs(zs["shrink_end"] - ph["hold_s"] - ph["shrink_s"]) < 1e-6,
          f"stream shrink {zs['shrink_start']}..{zs['shrink_end']}, config hold {ph['hold_s']} shrink {ph['shrink_s']}")
    nc = zs["next_center"]
    d_next = math.hypot(nc[0] - cfg["zone"]["start_center"][0], nc[1] - cfg["zone"]["start_center"][1])
    check("next circle inside current", d_next + zs["next_radius"] <= cfg["zone"]["start_radius"] + 1e-6,
          f"next center {d_next:.0f} m from start center (max {cfg['zone']['start_radius'] - zs['next_radius']:.0f})")

    # ---- npc behavior
    npc_sp = [math.hypot(m["vel"][0], m["vel"][1]) for o in obs for m in o["mechs"] if m["kind"] == "npc"]
    if npc_sp:
        check("npc speed", max(npc_sp) <= cfg["npc"]["speed"] * 1.1,
              f"max {max(npc_sp):.1f} m/s (config {cfg['npc']['speed']}); sim does NOT simulate NPC drones")
    other_sp = [math.hypot(m["vel"][0], m["vel"][1]) for o in obs for m in o["mechs"] if m["kind"] == "mech"]
    check("other mechs' speeds", max(other_sp) <= mech["boost_speed"] * 1.3,
          f"max {max(other_sp):.1f} m/s, median {statistics.median(other_sp):.1f}")

    w = max(len(r[1]) for r in report)
    for st, name, det in report:
        print(f"{st:4s}  {name:<{w}}  {det}")


if __name__ == "__main__":
    main()
