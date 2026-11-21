"""Record a simulated match and write a self-contained HTML replay you can open in any browser.

    python tools/replay.py 142                       # practice-like tiered squad, seed 142 -> replays/squad_142.html
    python tools/replay.py 7 --mode ffa --open       # free-for-all, open the browser when done
    python tools/replay.py 3 --mode duel
    python tools/replay.py 142 --hero tools.baseline.brain_v2   # watch another version in the hero seat

The viewer (tools/viewer.html) shows the campus top-down: buildings shaded by height, the zone and next
circle, every mech (aim line, HP bar, weapon), bullets, mortar shells, explosions, laser/lightning hits,
kills, and our bot's own debug drawing (planned path, goal, mode label). Same setup as tools/sim.py, so a
seed you saw misbehave in `sim.py --diag` replays identically here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import sim  # noqa: E402

TIERED = "tools.baseline.brain_v1,tools.baseline.brain_v1,tools.baseline.brain_v2,tools.baseline.brain_v2"
TIER_NAMES = ["easy", "medium", "hard", "expert"]
W_CODE = {None: 0, "gun": 0, "mortar": 1, "laser": 2, "lightning": 3}


def r1(v):
    return round(v, 1)


class RecordingSim(sim.Sim):
    """Sim that also logs visual effects (beams, explosions, kills) for the replay."""

    def __init__(self, *a, **kw):
        self.fx = []
        self.feed = []
        super().__init__(*a, **kw)

    def damage(self, src, victim, amount, weapon, from_pos, effect=None, by_id=None):
        if src is not None and victim.alive and weapon in ("lightning", "laser"):
            self.fx.append(["arc" if weapon == "lightning" else "beam", r1(src.pos[0]), r1(src.pos[1]),
                            r1(victim.pos[0]), r1(victim.pos[1])])
        if src is not None and victim.alive and weapon in ("gun", "mortar", "laser", "lightning"):
            self.fx.append(["hit", r1(victim.pos[0]), r1(victim.pos[1]), round(amount)])
        super().damage(src, victim, amount, weapon, from_pos, effect, by_id)

    def fire_laser(self, m, w):
        o, d = self.muzzle(m)
        L = self.ray_wall(o, d, w.stats["range"])
        self.fx.append(["beam", r1(o[0]), r1(o[1]), r1(o[0] + d[0] * L), r1(o[1] + d[1] * L)])
        super().fire_laser(m, w)

    def explode(self, owner, at):
        self.fx.append(["boom", r1(at[0]), r1(at[1]), self.cfg["weapons"]["mortar"]["splash_radius"]])
        super().explode(owner, at)

    def kill(self, v, src):
        self.feed.append([round(self.t, 1), v.id, src.id if src is not None else -1])
        super().kill(v, src)


def build(seed, mode, hero_spec, opp_spec, max_t=None):
    sim.Sim = RecordingSim  # setup_match builds whatever sim.Sim is
    s, hero_ids = sim.setup_match(seed, mode, hero_spec, opp_spec)
    w = s.world
    buildings = []
    for p in w.prisms:
        ring = []
        for x, y in zip(p.xs, p.ys):
            ring += [r1(x), r1(y)]
        buildings.append({"k": p.kind[0], "b": r1(p.base), "t": r1(p.top), "r": ring})
    roles = {}
    for m in s.mechs:
        if m.id in hero_ids:
            roles[m.id] = "hero"
        elif mode == "squad":
            roles[m.id] = TIER_NAMES[getattr(s, "tier", {}).get(m.id, 0)] if len(opp_spec.split(",")) == 4 else "squad"
        else:
            roles[m.id] = "opp"
    frames = []
    feed_seen = [0]

    def rec(S):
        z = S.zone()
        ms = []
        for m in S.mechs:
            if not m.alive:
                continue
            w1 = m.weapons[1].kind if m.weapons[1] else None
            charging = any(x and x.kind == "laser" and x.charge > 0 for x in m.weapons)
            ms.append([m.id, r1(m.pos[0]), r1(m.pos[1]), r1(m.pos[2]), round(m.aim[0], 2), round(m.hp),
                       round(m.max_hp), m.cubes, W_CODE.get(w1, 0), 1 if charging else 0,
                       1 if m.id in S.hunters else 0])
        f = {"t": round(S.t, 2),
             "z": [r1(z["center"][0]), r1(z["center"][1]), r1(z["radius"]),
                   r1(z["next_center"][0]), r1(z["next_center"][1]), r1(z["next_radius"])],
             "m": ms,
             "p": [[r1(p["pos"][0]), r1(p["pos"][1]), 1 if p["kind"] == "mortar" else 0] for p in S.projectiles],
             "c": [[r1(c["pos"][0]), r1(c["pos"][1])] for c in S.crates.values()],
             "n": [[r1(n["pos"][0]), r1(n["pos"][1]), round(n["hp"]), 1 if n["target"] is not None else 0]
                   for n in S.npcs.values()],
             "q": [[r1(c["pos"][0]), r1(c["pos"][1])] for c in S.cubes.values()],
             "w": [[r1(p["pos"][0]), r1(p["pos"][1]), W_CODE.get(p["kind"], 0)] for p in S.pickups.values()],
             "d": [[r1(d["pos"][0]), r1(d["pos"][1]), round(max(0.0, d["lands"] - S.t), 1), d["holder"]]
                   for d in S.drops.values()]}
        if S.fx:
            f["fx"] = S.fx
            S.fx = []
        if len(S.feed) > feed_seen[0]:
            f["k"] = S.feed[feed_seen[0]:]
            feed_seen[0] = len(S.feed)
        dbg, modes = {}, {}
        for m in S.mechs:
            if not m.alive:
                continue
            mode_name = getattr(m.bot, "mode", None)
            if mode_name:
                modes[m.id] = mode_name
            if m.id in hero_ids and getattr(m, "last_debug", None):
                dbg[m.id] = m.last_debug
        if modes:
            f["mo"] = modes
        if dbg:
            f["dbg"] = dbg
        frames.append(f)

    s.recorder = rec
    t0 = time.time()
    res = s.run(max_t=max_t)
    places = {mid: place for mid, place, *_ in res}
    meta = {"seed": seed, "mode": mode, "hero": hero_spec, "opp": opp_spec, "hero_ids": hero_ids,
            "roles": {str(k): v for k, v in roles.items()},
            "names": {str(m.id): getattr(m.bot, "name", "bot") for m in s.mechs},
            "places": {str(k): v for k, v in places.items()},
            "kills": {str(m.id): m.kills for m in s.mechs},
            "bounds": [w.minX, w.minY, w.maxX, w.maxY], "end_t": round(s.t, 1),
            "wall_s": round(time.time() - t0, 1)}
    return {"meta": meta, "buildings": buildings, "frames": frames}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seed", type=int)
    ap.add_argument("--mode", choices=["squad", "ffa", "duel"], default="squad")
    ap.add_argument("--hero", default="brain")
    ap.add_argument("--opp", default=None, help="opponent bot(s); squad default = tiered easy,medium,hard,expert")
    ap.add_argument("--max-t", type=float, default=None, help="stop the match early (seconds)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--open", action="store_true", help="open the replay in your browser")
    a = ap.parse_args()
    opp = a.opp or (TIERED if a.mode == "squad" else "tools.baseline.brain_v2")
    data = build(a.seed, a.mode, a.hero, opp, a.max_t)
    out = a.out or os.path.join(ROOT, "replays", f"{a.mode}_{a.seed}.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(os.path.join(ROOT, "tools", "viewer.html"), encoding="utf-8") as f:
        html = f.read()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("/*__REPLAY_DATA__*/null", blob, 1)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    hm = data["meta"]
    hero = hm["hero_ids"][0]
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(data['frames'])} frames, match {hm['end_t']} s)")
    print(f"hero {hero}: place {hm['places'][str(hero)]}, kills {hm['kills'][str(hero)]}")
    if a.open:
        import webbrowser
        webbrowser.open("file://" + os.path.abspath(out))


if __name__ == "__main__":
    main()
