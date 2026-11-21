"""Approximate local simulator of a mechfight/1 match, written from protocol.md. NOT the real server.

Use it to catch behavior bugs (stuck, zone deaths, never firing) and to A/B bot versions. Fidelity is
rough: physics, wall sliding, vision, canopy, NPC drones and many edge cases are simplified. Trust the
practice leaderboard over this.

    python tools/sim.py                               # 1 match: our bot vs 15 baseline squad (practice-like)
    python tools/sim.py -n 8 -j 4                     # 8 matches, 4 processes
    python tools/sim.py --mode ffa -n 8               # 8 ours vs 8 baseline free-for-all
    python tools/sim.py --hero tools.baseline.brain_v1 # A/B: put another bot in the hero seat

Bots run in-process (no stdin/stdout), with the same hello/obs dicts the server sends.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KIT = os.path.join(ROOT, "mechfight-kit")

import mechfight as mf  # noqa: E402
from brain.world import World  # noqa: E402

DT = 0.05
NOFIRE = [False, False]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load_hello():
    with gzip.open(os.path.join(KIT, "check", "stream.jsonl.gz"), "rt") as f:
        h = json.loads(f.readline())
    h["map"]["dir"] = os.path.join(KIT, "arena")
    return h


class Weapon:
    def __init__(self, kind, stats):
        self.kind = kind
        self.stats = stats  # shared per-mech per-kind dict (upgrades persist)
        self.cooldown = 0.0
        self.ammo = int(stats.get("mag", 0))
        self.reload = 0.0
        self.charge = 0.0
        self.held = 0.0

    def obs(self):
        d = {"kind": self.kind, "cooldown": round(max(0.0, self.cooldown), 3), "stats": dict(self.stats)}
        if self.kind == "gun":
            d["ammo"] = self.ammo
            d["reload"] = round(max(0.0, self.reload), 3)
        if self.kind == "laser":
            d["charge"] = round(self.charge, 3)
        return d


class Mech:
    def __init__(self, mid, pos, cfg, bot, team):
        self.id = mid
        self.pos = list(pos)
        self.vel = [0.0, 0.0, 0.0]
        self.aim = [random.uniform(-math.pi, math.pi), 0.0]
        self.cmd = {"move": [0, 0], "aim": list(self.aim), "boost": False}
        self.mstats = dict(cfg["mech"])
        self.wstats = {k: dict(v) for k, v in cfg["weapons"].items()}
        self.cubes = 0
        self.upgrades: List[str] = []
        self.hp = self.max_hp = float(self.mstats["hp"])
        self.energy = float(self.mstats["energy"])
        self.weapons: List[Optional[Weapon]] = [Weapon("gun", self.wstats["gun"]), None]
        self.alive = True
        self.grounded = True
        self.last_combat = -99.0
        self.last_spend = -99.0
        self.shorted_until = -1.0
        self.bot = bot
        self.team = team
        self.events: List[dict] = []
        self.sounds: List[dict] = []
        self.beams: List[dict] = []
        self.seen: Dict[int, float] = {}
        self.offers: List[dict] = []
        self.next_offer_id = 1
        self.place = None
        self.kills = 0
        self.fire = [False, False]
        self.act_ms: List[float] = []
        self.thresh_i = 0
        self.hits_log = []
        self.diag = {"death_t": None, "death_by": None, "stuck_s": 0.0, "out_s": 0.0, "dealt": 0.0, "taken": 0.0,
                     "shots": 0, "zone_dmg": 0.0}

    @property
    def scale(self):
        return 1 + 0.05 * self.cubes

    @property
    def radius(self):
        return 2.5 * self.scale

    @property
    def height(self):
        return 9.0 * self.scale


class Sim:
    def __init__(self, hello, bots: List[Any], teams: List[int], seed: int, friendly_fire_team: bool = False):
        self.rng = random.Random(seed)
        random.seed(seed)
        self.hello = hello
        self.cfg = hello["config"]
        self.world = World(hello["map"]["dir"])
        self.t = 0.0
        self.tick = 0
        self.projectiles: List[dict] = []
        self.next_id = {"proj": 1, "crate": 1000, "cube": 5000, "pickup": 8000, "drop": 20000}
        self.crates: Dict[int, dict] = {}
        self.cubes: Dict[int, dict] = {}
        self.pickups: Dict[int, dict] = {}
        self.drops: Dict[int, dict] = {}
        self.dead_order: List[int] = []
        self.squad_ff = friendly_fire_team
        self.hunters = set()
        self.omniscient = False
        self.recorder = None  # optional callable(sim) after every decision (tools/replay.py)
        self._zone_init()
        self.mechs: List[Mech] = []
        spawns = self._spawns(len(bots))
        if len(bots) == 2:
            a = spawns[0]
            for _ in range(500):
                yaw, d = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(70, 130)
                q = [a[0] + math.cos(yaw) * d, a[1] + math.sin(yaw) * d, 0]
                if self.world.in_bounds(q[0], q[1], 30):
                    g = self.world.ground(q[0], q[1])
                    if not any(p.covers(q[0], q[1]) and p.base < g + 9 for p in self.world.prisms_at(q[0], q[1])):
                        spawns[1] = [q[0], q[1], g]
                        break
        for i, (b, tm) in enumerate(zip(bots, teams)):
            self.mechs.append(Mech(i, spawns[i], self.cfg, b, tm))
        for _ in range(int(self.cfg["crates"]["count"])):
            p = self._open_point()
            self.crates[self._nid("crate")] = {"pos": p, "hp": float(self.cfg["crates"]["hp"])}
        kinds = [k for k, w in self.cfg["loot"]["weights"].items() if w > 0]
        for _ in range(int(self.cfg["loot"]["floor_count"])):
            self.pickups[self._nid("pickup")] = {"kind": self.rng.choice(kinds), "pos": self._open_point()}

    # ------------------------------------------------------------------ setup helpers
    def _nid(self, k):
        self.next_id[k] += 1
        return self.next_id[k]

    def _open_point(self, center=None, radius=None):
        w = self.world
        for _ in range(200):
            if center is None:
                x = self.rng.uniform(w.minX + 30, w.maxX - 30)
                y = self.rng.uniform(w.minY + 30, w.maxY - 30)
            else:
                a, r = self.rng.uniform(0, 2 * math.pi), radius * math.sqrt(self.rng.random())
                x, y = center[0] + r * math.cos(a), center[1] + r * math.sin(a)
                if not w.in_bounds(x, y, 30):
                    continue
            g = w.ground(x, y)
            if not any(p.covers(x, y) and p.base < g + 9 for p in w.prisms_at(x, y)):
                return [x, y, g]
        return [0.0, 0.0, w.ground(0, 0)]

    def _spawns(self, n):
        z = self.cfg["zone"]
        c, r = z["start_center"], z["start_radius"] * self.cfg["match"]["spawn_zone_fraction"]
        sep = self.cfg["match"]["spawn_min_separation"]
        pts: List[List[float]] = []
        for _ in range(4000):
            if len(pts) == n:
                break
            p = self._open_point(c, r)
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > sep for q in pts):
                pts.append(p)
            elif _ > 3000:
                sep *= 0.9
        while len(pts) < n:
            pts.append(self._open_point(c, r))
        return pts

    def _zone_init(self):
        z = self.cfg["zone"]
        self.zphases = []
        c, r, t = list(z["start_center"]), z["start_radius"], 0.0
        for ph in z["phases"]:
            nr = ph["radius"]
            a, d = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(0, max(0.0, r - nr))
            nc = [c[0] + d * math.cos(a), c[1] + d * math.sin(a)]
            self.zphases.append(dict(c=c, r=r, nc=nc, nr=nr, s0=t + ph["hold_s"], s1=t + ph["hold_s"] + ph["shrink_s"],
                                     dps=ph["dps"], t0=t))
            c, r, t = nc, nr, t + ph["hold_s"] + ph["shrink_s"]

    def zone(self):
        ph = self.zphases[-1]
        for p in self.zphases:
            if self.t < p["s1"]:
                ph = p
                break
        if self.t <= ph["s0"]:
            c, r = ph["c"], ph["r"]
        else:
            f = min(1.0, (self.t - ph["s0"]) / max(1e-6, ph["s1"] - ph["s0"]))
            c = [ph["c"][0] + (ph["nc"][0] - ph["c"][0]) * f, ph["c"][1] + (ph["nc"][1] - ph["c"][1]) * f]
            r = ph["r"] + (ph["nr"] - ph["r"]) * f
        return {"center": c, "radius": r, "next_center": ph["nc"], "next_radius": ph["nr"],
                "shrink_start": ph["s0"], "shrink_end": ph["s1"], "dps": ph["dps"]}

    # ------------------------------------------------------------------ perception
    def visible(self, a: Mech, tgt_pos, tgt_h) -> bool:
        vis = self.cfg["vision"]
        eye = (a.pos[0], a.pos[1], a.pos[2] + 8 * a.scale)
        d = math.hypot(tgt_pos[0] - eye[0], tgt_pos[1] - eye[1])
        near = vis["near_range"] + a.mstats.get("near_range_bonus", 0)
        far = vis["range"] + a.mstats.get("vision_bonus", 0)
        if d > far:
            return False
        if d > near:
            yaw = math.atan2(tgt_pos[1] - eye[1], tgt_pos[0] - eye[0])
            if abs(wrap(yaw - a.aim[0])) > math.radians(vis["fov_deg"]) / 2:
                return False
        c = (tgt_pos[0], tgt_pos[1], tgt_pos[2] + tgt_h * 0.5)
        top = (tgt_pos[0], tgt_pos[1], tgt_pos[2] + tgt_h - 0.2)
        return self.world.los(eye, c, step=3.0, skip_start=0) or self.world.los(eye, top, step=3.0, skip_start=0)

    def in_view_cheap(self, a: Mech, p) -> bool:
        vis = self.cfg["vision"]
        d = math.hypot(p[0] - a.pos[0], p[1] - a.pos[1])
        if d < vis["near_range"]:
            return True
        if d > vis["range"]:
            return False
        return abs(wrap(math.atan2(p[1] - a.pos[1], p[0] - a.pos[0]) - a.aim[0])) < math.radians(vis["fov_deg"]) / 2

    def build_obs(self, m: Mech, vis_cache) -> dict:
        mem = self.cfg["vision"].get("memory_s", 5)
        mechs = []
        for o in self.mechs:
            if o is m or not o.alive:
                continue
            if o.team is not None and o.team == m.team:
                continue  # squad mates are invisible to each other's targeting (they never fight)
            seen = vis_cache.get((m.id, o.id))
            if m.team is not None:
                # Practice-like squad: only `hunters` chase the lone bot (1 at the start, +1 per squad loss,
                # everyone after hunt_all_t). Hunters always know where it is; others only react up close.
                if m.id in self.hunters:
                    seen = self.omniscient or any(vis_cache.get((mm.id, o.id)) for mm in self.mechs
                                                  if mm.alive and mm.team == m.team)
                elif seen and math.dist(m.pos, o.pos) > self.cfg["vision"]["near_range"] and self.t < self.hunt_all_t:
                    seen = False
            if seen:
                m.seen[o.id] = self.t
            if seen or self.t - m.seen.get(o.id, -99) < mem:
                mechs.append({"id": o.id, "kind": "mech", "pos": [round(v, 2) for v in o.pos],
                              "vel": [round(v, 2) for v in o.vel], "aim": [round(v, 3) for v in o.aim],
                              "hp": round(o.hp), "max_hp": round(o.max_hp), "cubes": o.cubes,
                              "weapons": [w.kind for w in o.weapons if w],
                              "charging": any(w and w.kind == "laser" and w.charge > 0 for w in o.weapons)})
        crates = [{"id": i, "pos": c["pos"], "hp": c["hp"]} for i, c in self.crates.items() if self.in_view_cheap(m, c["pos"])]
        cubes = [{"id": i, "pos": c["pos"]} for i, c in self.cubes.items() if self.in_view_cheap(m, c["pos"])]
        pickups = [{"id": i, "kind": p["kind"], "pos": p["pos"]} for i, p in self.pickups.items() if self.in_view_cheap(m, p["pos"])]
        projs = [{"id": p["id"], "kind": p["kind"], "pos": [round(v, 2) for v in p["pos"]], "vel": [round(v, 2) for v in p["vel"]]}
                 for p in self.projectiles if p["owner"] != m.id and self.in_view_cheap(m, p["pos"])]
        drops = [{"id": i, "pos": d["pos"], "radius": self.cfg["drops"]["radius"],
                  "lands_in": round(max(0.0, d["lands"] - self.t), 2), "capture_s": self.cfg["drops"]["capture_s"],
                  "progress": round(d["progress"], 2), "holder": d["holder"]} for i, d in self.drops.items()]
        offer = m.offers[0] if m.offers else None
        z = self.zone()
        o = {"type": "obs", "tick": self.tick, "t": round(self.t, 2), "alive": sum(x.alive for x in self.mechs),
             "self": {"id": m.id, "pos": [round(v, 2) for v in m.pos], "vel": [round(v, 2) for v in m.vel],
                      "aim": [round(v, 3) for v in m.aim], "hp": round(m.hp, 1), "max_hp": round(m.max_hp, 1),
                      "energy": round(m.energy, 1), "max_energy": m.mstats["energy"], "grounded": m.grounded,
                      "cubes": m.cubes, "upgrades": list(m.upgrades),
                      "weapons": [w.obs() if w else None for w in m.weapons],
                      "in_zone": math.hypot(m.pos[0] - z["center"][0], m.pos[1] - z["center"][1]) <= z["radius"],
                      "stats": dict(m.mstats, height=m.height, radius=m.radius, eye_height=8 * m.scale,
                                    muzzle_height=7 * m.scale)},
             "mechs": mechs, "projectiles": projs, "crates": crates, "cubes": cubes, "pickups": pickups,
             "sounds": m.sounds, "beams": m.beams, "events": m.events, "drops": drops, "zone": z,
             "offer": {k: v for k, v in offer.items() if k != "_ids"} if offer else None}
        m.sounds, m.beams, m.events = [], [], []
        return o

    def sound(self, kind, pos, src_id=None):
        rng = self.cfg["sound"]["range"].get(kind, 100)
        for m in self.mechs:
            if not m.alive or m.id == src_id:
                continue
            d = math.hypot(pos[0] - m.pos[0], pos[1] - m.pos[1])
            if d <= rng:
                sig = math.radians(self.cfg["sound"]["bearing_noise_deg"]) * m.mstats.get("hearing", 1)
                b = math.atan2(pos[1] - m.pos[1], pos[0] - m.pos[0]) + self.rng.gauss(0, sig)
                m.sounds.append({"kind": kind, "bearing": round(wrap(b), 3),
                                 "dist": "near" if d < 60 else "mid" if d < 160 else "far", "age": 0.0})

    # ------------------------------------------------------------------ damage
    def damage(self, src: Optional[Mech], victim: Mech, amount, weapon, from_pos, effect=None):
        if not victim.alive:
            return
        if src is not None and src.team is not None and src.team == victim.team and not self.squad_ff:
            return
        rear = False
        if from_pos is not None and weapon != "zone":
            bearing = math.atan2(from_pos[1] - victim.pos[1], from_pos[0] - victim.pos[0])
            rear = abs(wrap(bearing - victim.aim[0])) > math.radians(victim.mstats["front_arc_deg"]) / 2
            if src is not None:
                amount *= 1 + self.cfg["cubes"]["damage_per_cube"] * min(src.cubes, self.cfg["cubes"]["stat_cap"])
            if rear:
                amount *= victim.mstats["rear_mult"]
            amount *= max(0.0, 1 - victim.mstats.get("armor", 0))
        else:
            bearing = None
        victim.hp -= amount
        victim.diag["taken"] += amount
        if src is not None:
            victim.hits_log.append((self.t, src.id))
        if weapon == "zone":
            victim.diag["zone_dmg"] += amount
        if src is not None:
            src.diag["dealt"] += amount
            if weapon == "gun":
                src.diag["gun_hits"] = src.diag.get("gun_hits", 0) + 1
        victim.last_combat = self.t
        ev = {"type": "damaged", "amount": round(amount, 1), "weapon": weapon,
              "bearing": None if bearing is None else round(bearing, 3), "by": src.id if src else None}
        if rear:
            ev["rear"] = True
        victim.events.append(ev)
        if src is not None:
            src.last_combat = self.t
            h = {"type": "hit", "target": victim.id, "amount": round(amount, 1), "weapon": weapon}
            if effect:
                h["effect"] = effect
            src.events.append(h)
        if victim.hp <= 0:
            self.kill(victim, src)

    def kill(self, v: Mech, src: Optional[Mech]):
        v.alive = False
        v.hp = 0
        v.diag["death_t"] = round(self.t, 1)
        v.diag["attackers8s"] = len({sid for (tt, sid) in v.hits_log if self.t - tt < 8.0})
        v.diag["hp_at_fight"] = None
        v.diag["death_by"] = (src.id if src else "zone")
        self.dead_order.append(v.id)
        v.place = sum(m.alive for m in self.mechs) + 1
        if src is not None and src is not v:
            src.kills += 1
            src.events.append({"type": "kill", "target": v.id})
        for w in v.weapons:
            if w and w.kind != "gun":
                self.pickups[self._nid("pickup")] = {"kind": w.kind, "pos": list(v.pos)}
        for _ in range(v.cubes + int(self.cfg["cubes"]["death_drop_extra"])):
            self.cubes[self._nid("cube")] = {"pos": [v.pos[0] + self.rng.uniform(-3, 3), v.pos[1] + self.rng.uniform(-3, 3), v.pos[2]]}
        self.sound("death", v.pos)

    def damage_crate(self, cid, amount, src: Optional[Mech]):
        c = self.crates.get(cid)
        if not c:
            return
        if src:
            amount *= 1 + self.cfg["cubes"]["damage_per_cube"] * min(src.cubes, self.cfg["cubes"]["stat_cap"])
            src.events.append({"type": "hit", "target": cid, "amount": round(amount, 1), "weapon": "gun"})
        c["hp"] -= amount
        if c["hp"] <= 0:
            del self.crates[cid]
            self.cubes[self._nid("cube")] = {"pos": list(c["pos"])}
            if self.rng.random() < self.cfg["loot"]["crate_weapon_chance"]:
                kinds = [k for k, w in self.cfg["loot"]["weights"].items() if w > 0]
                self.pickups[self._nid("pickup")] = {"kind": self.rng.choice(kinds), "pos": list(c["pos"])}
            self.sound("crate", c["pos"])

    def add_cube(self, m: Mech):
        m.cubes += 1
        old = m.max_hp
        m.max_hp = m.mstats["hp"] * (1 + self.cfg["cubes"]["hp_per_cube"] * min(m.cubes, self.cfg["cubes"]["stat_cap"]))
        m.hp += m.max_hp - old
        m.events.append({"type": "pickup", "cubes": m.cubes})
        th = self.cfg["cubes"]["upgrade_thresholds"]
        while m.thresh_i < len(th) and m.cubes >= th[m.thresh_i]:
            m.thresh_i += 1
            self.make_offer(m)

    def make_offer(self, m: Mech):
        held = {w.kind for w in m.weapons if w}
        el = [u for u in self.cfg["upgrades"] if m.upgrades.count(u["id"]) < u["max"]
              and (not u.get("requires") or u["requires"] in held)]
        ch = self.rng.sample(el, min(3, len(el)))
        if ch:
            m.offers.append({"id": m.next_offer_id, "expires": None, "choices": [
                {"id": u["id"], "name": u["name"], "desc": u["desc"]} for u in ch]})
            m.next_offer_id += 1

    def apply_upgrade(self, m: Mech, uid):
        u = next(x for x in self.cfg["upgrades"] if x["id"] == uid)
        m.upgrades.append(uid)
        for e in u["effects"]:
            parts = e["path"].split(".")
            d = m.mstats if parts[0] == "mech" else m.wstats[parts[1]]
            key = parts[-1]
            if "mul" in e:
                d[key] = d.get(key, 0) * e["mul"]
            if "add" in e:
                d[key] = d.get(key, 0) + e["add"]
        old = m.max_hp
        m.max_hp = m.mstats["hp"] * (1 + self.cfg["cubes"]["hp_per_cube"] * min(m.cubes, self.cfg["cubes"]["stat_cap"]))
        m.hp += max(0.0, m.max_hp - old)
        m.events.append({"type": "upgrade", "id": uid})

    # ------------------------------------------------------------------ actions
    def apply_action(self, m: Mech, a: dict):
        m.cmd = {"move": a.get("move", [0, 0]), "aim": a.get("aim", m.cmd["aim"]), "boost": bool(a.get("boost")),
                 "jump": bool(a.get("jump"))}
        f = a.get("fire") or NOFIRE
        m.fire = [bool(f[0]), bool(f[1])] if len(f) >= 2 else [bool(f[0]), False]
        if a.get("reload"):
            g = m.weapons[0] if m.weapons[0] and m.weapons[0].kind == "gun" else None
            if g and g.reload <= 0 and g.ammo < g.stats["mag"]:
                g.reload = g.stats["reload"]
        if "pick" in a and m.offers:
            o = m.offers.pop(0)
            i = int(a["pick"]) if 0 <= int(a["pick"]) < len(o["choices"]) else 0
            self.apply_upgrade(m, o["choices"][i]["id"])
        if "take" in a:
            pid = int(a["take"])
            p = self.pickups.get(pid)
            slot = a.get("drop")
            reason = None
            if not p:
                reason = "gone"
            elif math.hypot(p["pos"][0] - m.pos[0], p["pos"][1] - m.pos[1]) > self.cfg["loot"]["pickup_radius"] or \
                    abs(p["pos"][2] - m.pos[2]) > 4:
                reason = "range"
            elif any(w and w.kind == p["kind"] for w in m.weapons):
                reason = "have"
            else:
                if slot is None:
                    slot = next((i for i, w in enumerate(m.weapons) if w is None), None)
                if slot is None:
                    reason = "full"
                elif m.weapons[slot] and m.weapons[slot].kind == "laser" and m.weapons[slot].charge > 0:
                    reason = "busy"
            if reason:
                m.events.append({"type": "take_failed", "id": pid, "reason": reason})
            else:
                old = m.weapons[slot]
                if old:
                    nid = self._nid("pickup")
                    self.pickups[nid] = {"kind": old.kind, "pos": list(m.pos)}
                    m.events.append({"type": "dropped", "id": nid, "kind": old.kind, "slot": slot})
                del self.pickups[pid]
                w = Weapon(p["kind"], m.wstats[p["kind"]])
                w.cooldown = max(self.cfg["loot"]["equip_s"], w.stats.get("cooldown", 0))
                m.weapons[slot] = w
                m.events.append({"type": "took", "id": pid, "kind": p["kind"], "slot": slot})

    # ------------------------------------------------------------------ physics
    def overlap(self, x, y, feet, h, r):
        """How many body sample points are inside walls (0 = free)."""
        w = self.world
        if not w.in_bounds(x, y, r):
            return 99
        n = 0
        for ox, oy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r), (r * .7, r * .7), (-r * .7, r * .7),
                       (r * .7, -r * .7), (-r * .7, -r * .7)):
            px, py = x + ox, y + oy
            if w.ground(px, py) > feet + 1.2 + 0.6:
                n += 1
                continue
            for p in w.prisms_at(px, py):
                if p.base < feet + h and p.top > feet + 1.2 and p.covers(px, py):
                    n += 1
                    break
        return n

    def blocked(self, x, y, feet, h, r, cur=0):
        """A move is blocked if it puts more of the body into walls than now (lets a wedged mech slide out)."""
        n = self.overlap(x, y, feet, h, r)
        return n > 0 and n >= max(1, cur)

    def step_mech(self, m: Mech):
        ms = m.mstats
        mv = m.cmd["move"]
        n = math.hypot(mv[0], mv[1])
        if n > 1:
            mv = [mv[0] / n, mv[1] / n]
        charging = any(w and w.kind == "laser" and w.charge > 0 for w in m.weapons)
        boost = m.cmd["boost"] and m.energy > 0.5
        spd = ms["boost_speed"] if (boost and m.grounded and n > 0.1) else ms["walk_speed"]
        if charging:
            spd *= ms["charge_move_mult"]
        acc = (ms["ground_accel"] if m.grounded else ms["air_accel"]) * DT
        for i in (0, 1):
            dv = mv[i] * spd - m.vel[i]
            m.vel[i] += max(-acc, min(acc, dv))
        if boost:
            m.last_spend = self.t
            if m.grounded and n > 0.1:
                m.energy -= ms["boost_drain"] * DT
            elif not m.grounded:
                m.energy -= ms["jet_drain"] * DT
                m.vel[2] = min(ms["max_rise_speed"], m.vel[2] + ms["jet_accel"] * DT)
        if m.cmd.get("jump") and m.grounded:
            m.vel[2] = ms["jump_speed"]
            m.grounded = False
        if self.t - m.last_spend > ms["energy_regen_delay"]:
            m.energy = min(ms["energy"], m.energy + ms["energy_regen"] * DT)
        m.energy = max(0.0, m.energy)
        # horizontal with sliding
        nx, ny = m.pos[0] + m.vel[0] * DT, m.pos[1] + m.vel[1] * DT
        h, r = m.height, m.radius
        cur = self.overlap(m.pos[0], m.pos[1], m.pos[2], h, r)
        if not self.blocked(nx, ny, m.pos[2], h, r, cur):
            m.pos[0], m.pos[1] = nx, ny
        elif not self.blocked(nx, m.pos[1], m.pos[2], h, r, cur):
            m.pos[0] = nx
            m.vel[1] = 0
        elif not self.blocked(m.pos[0], ny, m.pos[2], h, r, cur):
            m.pos[1] = ny
            m.vel[0] = 0
        else:
            m.vel[0] = m.vel[1] = 0
        floor = self.world.floor_under(m.pos[0], m.pos[1], m.pos[2], ms["step_up"])
        if not m.grounded or floor < m.pos[2] - 0.3:
            m.vel[2] -= ms["gravity"] * DT
            nz = m.pos[2] + m.vel[2] * DT
            if nz <= floor:
                m.pos[2], m.vel[2], m.grounded = floor, 0.0, True
            else:
                m.pos[2], m.grounded = nz, False
        else:
            m.pos[2], m.vel[2] = floor, 0.0
        # aim slew
        ty, tp = m.cmd["aim"]
        tp = max(ms["min_pitch"], min(ms["max_pitch"], tp))
        dy = wrap(ty - m.aim[0])
        st = ms["turn_rate"] * DT
        m.aim[0] = wrap(m.aim[0] + max(-st, min(st, dy)))
        sp = ms["pitch_rate"] * DT
        m.aim[1] += max(-sp, min(sp, tp - m.aim[1]))

    def muzzle(self, m: Mech, tube=False):
        z = m.pos[2] + (m.height + 0.5 if tube else 7 * m.scale)
        cy, sy, cp, spp = math.cos(m.aim[0]), math.sin(m.aim[0]), math.cos(m.aim[1]), math.sin(m.aim[1])
        d = (cp * cy, cp * sy, spp)
        o = m.radius + 0.5
        return [m.pos[0] + d[0] * o, m.pos[1] + d[1] * o, z + d[2] * o], d

    def hit_actor(self, a, b, exclude):
        """First mech/crate hit by segment a->b: (kind, obj, t) or None."""
        best = None
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        L2 = dx * dx + dy * dy
        for m in self.mechs:
            if not m.alive or m.id == exclude:
                continue
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((m.pos[0] - a[0]) * dx + (m.pos[1] - a[1]) * dy) / L2))
            px, py, pz = a[0] + dx * t, a[1] + dy * t, a[2] + dz * t
            if math.hypot(px - m.pos[0], py - m.pos[1]) < m.radius and m.pos[2] <= pz <= m.pos[2] + m.height:
                if best is None or t < best[2]:
                    best = ("mech", m, t)
        cr, ch = self.cfg["crates"]["radius"], self.cfg["crates"]["height"]
        for cid, c in self.crates.items():
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((c["pos"][0] - a[0]) * dx + (c["pos"][1] - a[1]) * dy) / L2))
            px, py, pz = a[0] + dx * t, a[1] + dy * t, a[2] + dz * t
            if math.hypot(px - c["pos"][0], py - c["pos"][1]) < cr and c["pos"][2] <= pz <= c["pos"][2] + ch:
                if best is None or t < best[2]:
                    best = ("crate", cid, t)
        return best

    def ray_wall(self, a, d, rng, step=2.0):
        """Distance along d to first solid, or rng."""
        s = step
        while s < rng:
            if self.world.solid(a[0] + d[0] * s, a[1] + d[1] * s, a[2] + d[2] * s):
                return s
            s += step
        return rng

    def weapons_tick(self, m: Mech):
        for i, w in enumerate(m.weapons):
            if not w:
                continue
            w.cooldown -= DT
            if w.kind == "gun" and w.reload > 0:
                w.reload -= DT
                if w.reload <= 0:
                    w.ammo = int(w.stats["mag"])
                    w.reload = 0
            held = m.fire[i]
            ws = w.stats
            if w.kind == "gun":
                if held and w.cooldown <= 0 and w.ammo > 0 and w.reload <= 0:
                    o, d = self.muzzle(m)
                    sp = math.radians(ws["spread_deg"])
                    yaw, pit = m.aim[0] + self.rng.gauss(0, sp), m.aim[1] + self.rng.gauss(0, sp)
                    d = (math.cos(pit) * math.cos(yaw), math.cos(pit) * math.sin(yaw), math.sin(pit))
                    self.projectiles.append({"id": self._nid("proj"), "kind": "gun", "owner": m.id, "pos": o,
                                             "vel": [d[0] * ws["speed"], d[1] * ws["speed"], d[2] * ws["speed"]],
                                             "left": ws["range"], "dmg": ws["damage"]})
                    w.cooldown = ws["cooldown"]
                    m.diag["shots"] += 1
                    w.ammo -= 1
                    if w.ammo <= 0:
                        w.reload = ws["reload"]
                    self.sound("gun", m.pos, m.id)
            elif w.kind == "mortar":
                if held and w.cooldown <= 0:
                    o, d = self.muzzle(m, tube=True)
                    self.projectiles.append({"id": self._nid("proj"), "kind": "mortar", "owner": m.id, "pos": o,
                                             "vel": [d[0] * ws["speed"], d[1] * ws["speed"], d[2] * ws["speed"]],
                                             "left": ws["ttl"], "dmg": ws["damage"]})
                    w.cooldown = ws["cooldown"]
                    self.sound("mortar", m.pos, m.id)
            elif w.kind == "laser":
                if self.t < m.shorted_until:
                    w.charge = 0
                    continue
                if w.charge > 0 or (held and w.cooldown <= 0):
                    w.charge = min(ws["charge"], w.charge + DT)
                    if w.charge >= ws["charge"]:
                        w.held += DT
                    release = (not held and w.charge >= ws["min_charge"]) or w.held >= ws["hold_s"]
                    if release:
                        self.fire_laser(m, w)
            elif w.kind == "lightning":
                if held and w.cooldown <= 0 and m.energy >= ws["energy"]:
                    m.energy -= ws["energy"]
                    m.last_spend = self.t
                    w.cooldown = ws["cooldown"]
                    self.fire_lightning(m, w)
                    self.sound("lightning", m.pos, m.id)

    def fire_laser(self, m: Mech, w: Weapon):
        ws = w.stats
        frac = ws["min_frac"] + (1 - ws["min_frac"]) * min(1.0, w.charge / ws["charge"])
        o, d = self.muzzle(m)
        L = self.ray_wall(o, d, ws["range"])
        end = [o[0] + d[0] * L, o[1] + d[1] * L, o[2] + d[2] * L]
        hit = self.hit_actor(o, end, m.id)
        if hit:
            if hit[0] == "mech":
                v = hit[1]
                mult = 1 + ws["cookoff"] if any(x and x.kind == "mortar" for x in v.weapons) else 1
                self.damage(m, v, ws["damage"] * frac * mult, "laser", m.pos, "cookoff" if mult > 1 else None)
            else:
                self.damage_crate(hit[1], ws["damage"] * frac, m)
        for o2 in self.mechs:
            if o2.alive and o2 is not m:
                o2.beams.append({"by": m.id, "kind": "laser", "path": [o, end], "age": 0.0})
        w.charge = 0
        w.held = 0
        w.cooldown = ws["cooldown"]
        self.sound("laser", m.pos, m.id)

    def fire_lightning(self, m: Mech, w: Weapon):
        ws = w.stats
        o, _ = self.muzzle(m)
        cands = []
        for v in self.mechs:
            if not v.alive or v is m or (m.team is not None and v.team == m.team):
                continue
            c = (v.pos[0], v.pos[1], v.pos[2] + v.height / 2)
            d = math.dist(o, c)
            if d < ws["range"] and abs(wrap(math.atan2(c[1] - o[1], c[0] - o[0]) - m.aim[0])) < math.radians(ws["cone_deg"]):
                if self.world.los(o, c, skip_start=0):
                    cands.append((d, v))
        if not cands:
            return
        cands.sort(key=lambda x: x[0])
        tgt = cands[0][1]
        dmg = ws["damage"]
        hitset = {tgt.id}
        for _ in range(int(ws["chain"]) + 1):
            if tgt.weapons[1] and tgt.weapons[1].kind == "laser" or tgt.weapons[0] and tgt.weapons[0].kind == "laser":
                tgt.shorted_until = self.t + ws["short_s"]
            self.damage(m, tgt, dmg, "lightning", m.pos)
            dmg *= ws["chain_falloff"]
            nxt = [v for v in self.mechs if v.alive and v.id not in hitset and v is not m and
                   math.dist(v.pos, tgt.pos) < ws["chain_range"]]
            if not nxt:
                break
            tgt = min(nxt, key=lambda v: math.dist(v.pos, tgt.pos))
            hitset.add(tgt.id)

    def projectiles_tick(self):
        keep = []
        g = self.cfg["weapons"]["mortar"]["gravity"]
        for p in self.projectiles:
            owner = self.mechs[p["owner"]] if p["owner"] < len(self.mechs) else None
            a = list(p["pos"])
            if p["kind"] == "mortar":
                p["vel"][2] -= g * DT
            b = [a[0] + p["vel"][0] * DT, a[1] + p["vel"][1] * DT, a[2] + p["vel"][2] * DT]
            seg = math.dist(a, b)
            d = [(b[i] - a[i]) / max(seg, 1e-9) for i in range(3)]
            wall = self.ray_wall(a, d, seg, step=2.5)
            end = b if wall >= seg else [a[0] + d[0] * wall, a[1] + d[1] * wall, a[2] + d[2] * wall]
            hit = self.hit_actor(a, end, p["owner"])
            done = False
            if p["kind"] == "gun":
                if hit:
                    if hit[0] == "mech":
                        self.damage(owner, hit[1], p["dmg"], "gun", a)
                    else:
                        self.damage_crate(hit[1], p["dmg"], owner)
                    done = True
                elif wall < seg:
                    done = True
                p["left"] -= seg
                done = done or p["left"] <= 0
            else:
                p["left"] -= DT
                if hit or wall < seg or p["left"] <= 0:
                    hp = end
                    if hit:
                        t = hit[2]
                        hp = [a[0] + (end[0] - a[0]) * t, a[1] + (end[1] - a[1]) * t, a[2] + (end[2] - a[2]) * t]
                        if hit[0] == "mech":
                            self.damage(owner, hit[1], p["dmg"], "mortar", a)
                    self.explode(owner, hp)
                    done = True
            p["pos"] = end
            if not done:
                keep.append(p)
        self.projectiles = keep

    def explode(self, owner: Optional[Mech], at):
        ws = self.cfg["weapons"]["mortar"] if owner is None else owner.wstats["mortar"]
        R = ws["splash_radius"]
        for v in self.mechs:
            if not v.alive:
                continue
            c = (v.pos[0], v.pos[1], v.pos[2] + v.height / 2)
            d = max(0.0, math.dist(c, at) - v.radius)
            if d > R or not self.world.los(at, c, skip_start=0.5):
                continue
            f = 1 - (1 - ws["splash_edge"]) * d / R
            dmg = ws["splash"] * f * (self.cfg["mech"]["self_damage_mult"] if v is owner else 1)
            self.damage(owner if v is not owner else None, v, dmg, "mortar", at, "concuss")
            yaw = math.atan2(c[1] - at[1], c[0] - at[0])
            v.vel[0] += math.cos(yaw) * ws["knock"] * f
            v.vel[1] += math.sin(yaw) * ws["knock"] * f
            v.vel[2] += ws["knock_up"] * f
            v.grounded = False
            v.energy = max(0.0, v.energy - ws["drain"] * f)
        for cid in [cid for cid, c in self.crates.items() if math.dist(c["pos"], at) < R]:
            self.damage_crate(cid, ws["splash"] * 0.6, owner)
        self.sound("explosion", at)

    def world_tick(self):
        z = self.zone()
        for m in self.mechs:
            if not m.alive:
                continue
            out = math.hypot(m.pos[0] - z["center"][0], m.pos[1] - z["center"][1]) > z["radius"]
            if out:
                m.diag["out_s"] += DT
            if math.hypot(*m.cmd["move"]) > 0.5 and math.hypot(m.vel[0], m.vel[1]) < 1.0:
                m.diag["stuck_s"] += DT
            if out:
                self.damage(None, m, z["dps"] * DT, "zone", None)
                m.last_combat = self.t
            elif self.t - m.last_combat > m.mstats["regen_delay"]:
                m.hp = min(m.max_hp, m.hp + m.mstats["regen_per_s"] * DT)
        # cubes
        pr = self.cfg["cubes"]["pickup_radius"]
        for cid in list(self.cubes):
            c = self.cubes[cid]
            near = [m for m in self.mechs if m.alive and math.hypot(m.pos[0] - c["pos"][0], m.pos[1] - c["pos"][1]) < pr
                    and abs(m.pos[2] - c["pos"][2]) < 4]
            if near:
                m = min(near, key=lambda m: (math.hypot(m.pos[0] - c["pos"][0], m.pos[1] - c["pos"][1]), m.id))
                del self.cubes[cid]
                self.add_cube(m)
        # drops
        dc = self.cfg.get("drops")
        if dc:
            for tt in dc["times"]:
                if self.t - DT < tt <= self.t:
                    zz = self.zone()
                    pos = self._open_point(zz["next_center"], zz["next_radius"] * dc["spread"])
                    self.drops[self._nid("drop")] = {"pos": pos, "lands": self.t + dc["fall_s"], "progress": 0.0, "holder": None}
            for did in list(self.drops):
                d = self.drops[did]
                if self.t < d["lands"]:
                    continue
                inside = [m for m in self.mechs if m.alive and math.hypot(m.pos[0] - d["pos"][0], m.pos[1] - d["pos"][1]) < dc["radius"]]
                if len(inside) == 1:
                    if d["holder"] != inside[0].id:
                        d["holder"], d["progress"] = inside[0].id, 0.0
                    d["progress"] += DT
                    if d["progress"] >= dc["capture_s"]:
                        m = inside[0]
                        for _ in range(int(dc["cubes"])):
                            self.add_cube(m)
                        m.hp, m.energy = m.max_hp, m.mstats["energy"]
                        m.events.append({"type": "drop", "id": did, "cubes": m.cubes})
                        del self.drops[did]
                elif not inside:
                    d["holder"], d["progress"] = None, 0.0

    hunt_all_t = 330.0

    def update_hunters(self, alive):
        squad = [m for m in alive if m.team is not None]
        lone = [m for m in alive if m.team is None]
        self.hunters = set()
        if not squad or not lone:
            return
        dead = sum(1 for m in self.mechs if m.team is not None and not m.alive)
        k = len(squad) if self.t >= self.hunt_all_t else min(len(squad), 1 + dead)
        tgt = lone[0]
        tier = getattr(self, "tier", {})
        squad.sort(key=lambda m: (tier.get(m.id, 0), math.dist(m.pos, tgt.pos)))
        self.hunters = {m.id for m in squad[:k]}

    # ------------------------------------------------------------------ loop
    def run(self, max_t=None, verbose=False):
        max_t = max_t or self.cfg["match"]["duration_s"]
        for m in self.mechs:
            h = copy.deepcopy(self.hello)
            h["slot"] = m.id
            h["seed"] = self.rng.randrange(1 << 31)
            mf.draw.caps = h["config"]["debug"]
            m.bot = m.bot(h)
        while self.t < max_t and sum(m.alive for m in self.mechs) > 1:
            vis = {}
            alive = [m for m in self.mechs if m.alive]
            self.update_hunters(alive)
            for a in alive:
                for b in alive:
                    if a is not b:
                        vis[(a.id, b.id)] = self.visible(a, b.pos, b.height)
            for m in alive:
                obs = self.build_obs(m, vis)
                for off in m.offers:
                    if off["expires"] is None:
                        off["expires"] = self.t + self.cfg["bot"]["offer_timeout_s"]
                if m.offers and self.t > m.offers[0]["expires"]:
                    o = m.offers.pop(0)
                    self.apply_upgrade(m, o["choices"][0]["id"])
                mf.draw.clear()
                s = time.perf_counter()
                try:
                    a = m.bot.act(obs) or {}
                except Exception as e:  # match goes on, mech coasts
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    a = {"move": m.cmd["move"], "aim": m.cmd["aim"], "boost": m.cmd["boost"]}
                m.act_ms.append((time.perf_counter() - s) * 1000)
                m.last_debug = a.get("debug") or mf.draw.take()  # what the bot drew this decision
                m.last_action = a
                self.apply_action(m, a)
            for _ in range(int(self.cfg["match"]["decide_every"])):
                for m in self.mechs:
                    if m.alive:
                        self.step_mech(m)
                        self.weapons_tick(m)
                self.projectiles_tick()
                self.world_tick()
                self.t += DT
                self.tick += 1
            if self.recorder is not None:
                self.recorder(self)
        # placement: survivors ranked by hp
        alive = sorted([m for m in self.mechs if m.alive], key=lambda m: -m.hp)
        for i, m in enumerate(alive):
            m.place = i + 1
        return [(m.id, m.place, m.kills, m.team, m.cubes) for m in self.mechs]


def import_bot(spec: str):
    mod = importlib.import_module(spec)
    return getattr(mod, "Brain")


def setup_match(seed, mode, hero_spec, opp_spec, omni=False):
    """Build a Sim exactly like run_match does: (sim, hero_ids)."""
    hello = load_hello()
    hero = import_bot(hero_spec)
    # opp may be "easy,medium,hard,expert" specs (4/4/4/3 bots, hunting weakest tier first, like practice)
    tiers = [import_bot(x) for x in opp_spec.split(",")]
    tier_of = [min(len(tiers) - 1, i * len(tiers) // 15) for i in range(15)] if len(tiers) > 1 else [0] * 15
    if len(tiers) == 4:
        tier_of = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 3
    opp = tiers[0]
    rng = random.Random(seed)
    hero_slot = None
    mech_tier = {}
    if mode == "squad":
        hero_slot = rng.randrange(16)
        order = [i for i in range(16) if i != hero_slot]
        rng.shuffle(order)
        bots = [hero] * 16
        for k, i in enumerate(order):
            bots[i] = tiers[tier_of[k]]
            mech_tier[i] = tier_of[k]
        teams = [None if i == hero_slot else 1 for i in range(16)]
    elif mode == "duel":
        bots, teams = [hero, opp], [None, None]
    else:
        bots = [hero if i % 2 == 0 else opp for i in range(16)]
        teams = [None] * 16
    sim = Sim(hello, bots, teams, seed)
    sim.omniscient = omni
    sim.tier = mech_tier
    hero_ids = [hero_slot] if mode == "squad" else [0] if mode == "duel" else [i for i in range(16) if i % 2 == 0]
    return sim, hero_ids


def run_match(args):
    seed, mode, hero_spec, opp_spec, omni = args
    sim, hero_ids = setup_match(seed, mode, hero_spec, opp_spec, omni)
    t0 = time.time()
    res = sim.run()
    acts = [x for m in sim.mechs for x in m.act_ms]
    hero_ms = max((x for m in sim.mechs if m.id in hero_ids for x in m.act_ms), default=0)
    diags = {m.id: dict(m.diag, place=m.place, kills=m.kills, cubes=m.cubes, w=[w.kind for w in m.weapons if w])
             for m in sim.mechs}
    return {"seed": seed, "res": res, "diag": diags, "hero": hero_ids, "t": sim.t,
            "wall": time.time() - t0, "hero_max_ms": hero_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1)
    ap.add_argument("-j", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=["squad", "ffa", "duel"], default="squad")
    ap.add_argument("--hero", default="brain")
    ap.add_argument("--opp", default="tools.baseline.brain_v1")
    ap.add_argument("--diag", action="store_true", help="print per-hero diagnostics")
    ap.add_argument("--omniscient", action="store_true", help="squad hunters always know where you are")
    a = ap.parse_args()
    jobs = [(a.seed + i, a.mode, a.hero, a.opp, a.omniscient) for i in range(a.n)]
    if a.j > 1:
        from multiprocessing import Pool
        with Pool(a.j) as p:
            outs = []
            for o in p.imap_unordered(run_match, jobs):
                print(f"done seed {o['seed']} wall={o['wall']:.0f}s", file=sys.stderr, flush=True)
                outs.append(o)
            outs.sort(key=lambda o: o["seed"])
    else:
        outs = [run_match(j) for j in jobs]
    hero_pts, opp_pts, kills = [], [], []
    for o in outs:
        for mid, place, k, team, cubes in o["res"]:
            if mid in o["hero"]:
                hero_pts.append(16 - place)
                kills.append(k)
            else:
                opp_pts.append(16 - place)
        hp = [16 - p for mid, p, *_ in o["res"] if mid in o["hero"]]
        for h in o["hero"]:
            if a.diag:
                d = o["diag"][h]
                print(f"   hero {h}: " + " ".join(f"{k}={round(v, 1) if isinstance(v, float) else v}" for k, v in d.items()))
        print(f"seed {o['seed']}: hero pts {hp} end t={o['t']:.0f}s wall={o['wall']:.0f}s "
              f"hero max act {o['hero_max_ms']:.1f} ms", flush=True)
    acc_h, acc_o = [0, 0], [0, 0]
    for o in outs:
        for mid, d in o["diag"].items():
            tgt = acc_h if mid in o["hero"] else acc_o
            tgt[0] += d.get("gun_hits", 0)
            tgt[1] += d.get("shots", 0)
    print(f"gun accuracy: hero {acc_h[0] / max(1, acc_h[1]):.3f}  opp {acc_o[0] / max(1, acc_o[1]):.3f}")
    print(f"\n{a.mode}: hero={a.hero} mean pts {statistics.mean(hero_pts):.2f} "
          f"(kills/match {statistics.mean(kills):.2f})  opp={a.opp} mean {statistics.mean(opp_pts):.2f}  "
          f"over {len(outs)} matches")


if __name__ == "__main__":
    main()
