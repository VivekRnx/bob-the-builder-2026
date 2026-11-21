"""Trace the hero in one simulated match: python tools/trace.py SEED [--from T] [--to T] [--every K]"""
import argparse, math, os, sys, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import sim

ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int)
ap.add_argument("--from", dest="t0", type=float, default=0)
ap.add_argument("--to", dest="t1", type=float, default=60)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--hero", default="brain")
ap.add_argument("--opp", default="tools.baseline.brain_v1")
ap.add_argument("--mode", default="squad")
a = ap.parse_args()

hero, opp = sim.import_bot(a.hero), sim.import_bot(a.opp)
rng = random.Random(a.seed)
hs = rng.randrange(16) if a.mode == "squad" else 0
bots = [hero if i == hs else opp for i in range(16)]
teams = [None if i == hs else 1 for i in range(16)] if a.mode == "squad" else [None] * 16
s = sim.Sim(sim.load_hello(), bots, teams, a.seed)
H = s.mechs[hs]
orig = None
n = [0]

def wrap_act(bot):
    f = bot.act
    def act(obs):
        out = f(obs)
        n[0] += 1
        if a.t0 <= obs["t"] <= a.t1 and (n[0] % a.every == 0 or obs["events"]):
            me = obs["self"]
            vis = [(m["id"], round(math.dist(m["pos"], me["pos"]))) for m in obs["mechs"] if m["kind"] == "mech"]
            ev = [(e["type"], e.get("by", e.get("target")), e.get("amount")) for e in obs["events"]]
            print(f"t={obs['t']:5.1f} {getattr(bot, 'mode', '?'):10s} hp={me['hp']:6.0f} pos=({me['pos'][0]:.0f},{me['pos'][1]:.0f},{me['pos'][2]:.1f}) "
                  f"aim=({me['aim'][0]:.2f},{me['aim'][1]:.2f}) cmd_aim={[round(x,2) for x in out.get('aim',[])]} "
                  f"mv={out.get('move')} fire={out.get('fire')} see={vis} ev={ev}")
        return out
    bot.act = act

old_run = s.run
# instantiate bots like Sim.run does, then wrap hero
import copy
def run():
    for m in s.mechs:
        h = copy.deepcopy(s.hello); h["slot"] = m.id; h["seed"] = 1
        m.bot = m.bot(h)
    wrap_act(H.bot)
    for m in s.mechs:
        m.bot_cls_done = True
    # reuse loop without re-instantiating
    cls_init = sim.Sim.run
    s.mechs_inst = True
    return loop()
def loop():
    while s.t < a.t1 and H.alive:
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
            s.apply_action(m, m.bot.act(obs) or {})
        for _ in range(2):
            for m in s.mechs:
                if m.alive:
                    s.step_mech(m); s.weapons_tick(m)
            s.projectiles_tick(); s.world_tick(); s.t += sim.DT; s.tick += 1
    print("hero", H.id, "alive" if H.alive else f"dead at {H.diag['death_t']} by {H.diag['death_by']}", H.diag)
run()
