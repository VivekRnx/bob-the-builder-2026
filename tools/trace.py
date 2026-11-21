"""Trace our bot in one simulated match (same setup as tools/sim.py for that seed).

    python tools/trace.py 142 --from 100 --to 170 --every 10
    python tools/trace.py 7 --opp tools.baseline.brain_v1   # opponents (comma list = easy,medium,hard,expert)
"""
import argparse
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import sim  # noqa: E402

TIERED = "tools.baseline.brain_v1,tools.baseline.brain_v1,tools.baseline.brain_v2,tools.baseline.brain_v2"
ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int)
ap.add_argument("--from", dest="t0", type=float, default=0)
ap.add_argument("--to", dest="t1", type=float, default=60)
ap.add_argument("--every", type=int, default=5, help="print every Nth decision (and every one with events)")
ap.add_argument("--hero", default="brain")
ap.add_argument("--opp", default=TIERED)
ap.add_argument("--mode", default="squad")
a = ap.parse_args()
VEL = "--vel" in sys.argv

s, hero_ids = sim.setup_match(a.seed, a.mode, a.hero, a.opp)
H = s.mechs[hero_ids[0]]
for m in s.mechs:
    h = copy.deepcopy(s.hello)
    h["slot"] = m.id
    h["seed"] = s.rng.randrange(1 << 31)
    m.bot = m.bot(h)
n = 0
while s.t < a.t1 and H.alive and sum(m.alive for m in s.mechs) > 1:
    vis = {}
    alive = [m for m in s.mechs if m.alive]
    s.update_hunters(alive)
    for x in alive:
        for y in alive:
            if x is not y:
                vis[(x.id, y.id)] = s.visible(x, y.pos, y.height)
    for m in alive:
        obs = s.build_obs(m, vis)
        if m.offers and m.offers[0]["expires"] is None:
            m.offers[0]["expires"] = s.t + 5
        act = m.bot.act(obs) or {}
        if m is H and a.t0 <= obs["t"]:
            n += 1
            if n % a.every == 0 or obs["events"]:
                me, z = obs["self"], obs["zone"]
                see = [(o["id"], round(math.dist(o["pos"], me["pos"]))) for o in obs["mechs"] if o["kind"] == "mech"]
                ev = [(e["type"], e.get("by", e.get("target")), e.get("amount")) for e in obs["events"]]
                print(f"t={obs['t']:5.1f} {getattr(m.bot, 'mode', '?'):14s} hp={me['hp']:5.0f} e={me['energy']:3.0f} "
                      f"pos=({me['pos'][0]:.0f},{me['pos'][1]:.0f},{me['pos'][2]:.1f}) inzone={int(me['in_zone'])} "
                      f"zone=({z['center'][0]:.0f},{z['center'][1]:.0f}) r{z['radius']:.0f}->r{z['next_radius']:.0f} "
                      f"mv={act.get('move')} b={int(bool(act.get('boost')))} fire={act.get('fire')} see={see} ev={ev}")
        s.apply_action(m, act)
    for _ in range(2):
        for m in s.mechs:
            if m.alive:
                s.step_mech(m)
                s.weapons_tick(m)
        s.projectiles_tick()
        s.world_tick()
        s.t += sim.DT
        s.tick += 1
print("hero", H.id, "alive" if H.alive else f"dead at {H.diag['death_t']} by {H.diag['death_by']}",
      {k: round(v, 1) if isinstance(v, float) else v for k, v in H.diag.items()})
