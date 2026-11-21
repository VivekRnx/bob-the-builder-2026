"""The bot's decision loop. One Brain per match; act() is called ~10x per match-second.

Pipeline each tick:
    perceive()      memory: enemy tracks (beyond the server's 5 s), attackers from damage events,
                    sound contacts, failed pickups, stuck detection
    pick_enemy()    who to fight (visible/remembered mechs, weighted by threat, HP, line of sight)
    decide()        mode + goal: zone emergency > dodge > fight / take cover / retreat > loot > cubes >
                    supply drop > farm crates & drones > zone drift
    navigate()      goal -> move vector: direct when clear, else A* on the nav grid; walls via
                    world.probe; jets when stuck; dodge overlay for lasers and mortar shells
    combat()        aim + fire per slot: gun lead, mortar ballistics (lob over cover), laser
                    charge/release, lightning cone; always face the threat (rear hits do x1.5)

Every number worth tuning is in TUNING below. Rule numbers come from hello.config, never hard-coded.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf

from .geom import aim_at, ballistic_pitch, flight_time, hdist, lead_point, unit, wrap, yaw_to
from .world import World, load_world

TUNING = {
    # weapons: higher = better to hold in slot 1 (slot 0 keeps the gun)
    "weapon_rank": {"laser": 3, "mortar": 2, "lightning": 1, "gun": 0},
    # Upgrade ids by preference; unknown ids go last. Max HP first: the zone ignores armor, so the
    # biggest HP pool survives the final burn. Armor second: it cuts every weapon hit.
    "upgrade_priority": ["plating", "armor", "nanites", "gun_rate", "gun_damage", "laser_damage",
                         "laser_charge", "gyros", "mortar_reload", "lightning_efficiency", "gun_mag",
                         "gun_velocity", "servos", "sensors", "reactor", "capacitor", "jets",
                         "mortar_splash", "mortar_velocity", "lightning_range", "lightning_chain",
                         "laser_pierce"],
    "loot_radius": 80.0,          # detour this far for a better weapon (when safe)
    "cube_radius": 90.0,          # detour this far for a power cube (when safe)
    "crate_radius": 120.0,        # shoot crates within this range when safe
    "npc_farm_min_hp_frac": 0.7,  # only farm NPC drones (they shoot back) when healthy
    "drop_max_dist": 220.0,       # consider supply drops this close
    "drop_enemy_clear": 70.0,     # ...only if no enemy seen in the last 20 s is this close to the ring
    "drop_min_hp_frac": 0.0,      # ...and only when this healthy
    "zone_inner_frac": 0.55,      # aim for this fraction of the next circle's radius (deeper = safer)
    "zone_slack_s": 15.0,         # start moving for the next circle this many seconds early
    "boost_min_energy": 30.0,     # keep this much energy in reserve (for jets / escapes)
    "lightning_min_energy": 15.0,
    "strafe_period": (0.45, 1.3), # seconds between strafe-direction changes (random jukes)
    "fight_range": {"lightning": 26.0, "laser": 120.0, "mortar": 110.0, "gun": 65.0},
    "mortar_min_range": 26.0,     # don't splash ourselves
    "disengage_ratio": 0.75,      # leave a fight when my time-to-die < ratio * their time-to-die
    "cover_hp_frac": 0.45,        # below this, prefer cover to regen when anything threatens
    "reengage_hp_frac": 0.85,     # leave cover (if safe) once healed to this
    # Caution knobs above were A/B'd at cautious values (zone 0.45/25 s, drops 120 m/0.7 HP, drones 0.9,
    # disengage 0.9, cover 0.55/0.95, no chase, farm_quiet 10 s): 1.68 vs 2.25 pts over 110 sim matches.
    "chase": True,                # advance on enemies beyond weapon range? (False = hold/back off)
    "two_front": False,           # with 2+ enemies, strafe away from the second one (off: 1.88 vs 2.31 in sim A/B)
    "finish_hp_frac": 0.3,        # ...except to finish an enemy below this HP fraction
    "farm_quiet_s": 0.0,          # shoot crates only if no enemy was seen for this long (0 = off)
    "safe_after_s": 4.0,          # "safe" = no damage taken and no enemy seen for this long
    "track_keep_s": 20.0,         # remember enemies this long after the server stops reporting them
    "time_guard_ms": 28.0,        # skip optional heavy work past this per-tick time
    "heal_below_frac": 0.0,       # below this HP fraction with no enemy in view: stop farming, hide, regen (0 = off; lost in sim A/B)
    "early_farm_s": 90.0,         # before this match time...
    "early_farm_mult": 1.7,       # ...crates/cubes this much further away are worth a detour
    "reload_quiet_s": 2.0,        # reload a non-full magazine once no enemy has been seen this long
    "fight_boost_energy": 999.0,  # boost while strafing in a fight above this energy (999 = never)
    "debug": True,
}

# Local experiments only: MECH_TUNING='{"fight_boost_energy": 50}' overrides knobs (never set in a match).
try:
    import json as _json
    import os as _os
    TUNING.update(_json.loads(_os.environ.get("MECH_TUNING", "{}") or "{}"))
except Exception:  # a bad override must never break the bot
    pass

# rough sustained DPS per weapon (base stats), used only to judge fights
BASE_DPS = {"gun": 45 * 12 / (12 * 0.24 + 2.4), "laser": 315 / (1.65 + 0.75), "mortar": 230 / 2.6,
            "lightning": 60 / 0.25 * 0.8}
WEAPON_REACH = {"gun": 230.0, "laser": 280.0, "mortar": 230.0, "lightning": 45.0}


class Track:
    __slots__ = ("id", "pos", "vel", "t", "seen_t", "hp", "max_hp", "weapons", "charging", "aim", "cubes", "est")

    def __init__(self, mid):
        self.id = mid
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.t = -99.0       # last update of any kind
        self.seen_t = -99.0  # last time the server reported it in obs.mechs
        self.hp = 1400.0
        self.max_hp = 1400.0
        self.weapons: List[str] = ["gun"]
        self.charging = False
        self.aim = [0.0, 0.0]
        self.cubes = 0
        self.est = False     # position is a guess (from a damage bearing), not observed


class Brain:
    name = "osu-mech"

    def __init__(self, hello: mf.Hello):
        self.cfg = hello["config"]
        self.me_id = int(hello["slot"])
        self.rng = random.Random(hello["seed"])
        t0 = time.perf_counter()
        self.world: Optional[World] = load_world(hello.get("map", {}).get("dir"))
        mech = self.cfg["mech"]
        self.step_up = mech.get("step_up", 1.2)
        self.max_slope = mech.get("max_slope_deg", 50)
        self.nav = None
        if self.world is not None:
            try:
                from .nav import NavGrid
                self.nav = NavGrid(self.world, radius=mech.get("radius", 2.5), height=mech.get("height", 9.0),
                                   step_up=self.step_up, max_slope_deg=self.max_slope)
            except Exception as e:  # never die over navigation
                mf.log(f"brain: nav grid failed: {e!r}")
        mf.log(f"brain: slot {self.me_id}, map {'ok' if self.world else 'MISSING'}, "
               f"nav {'ok' if self.nav else 'off'} ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        # memory
        self.tracks: Dict[int, Track] = {}
        self.threat: Optional[Tuple[float, float, Optional[int]]] = None  # (bearing, t, attacker id)
        self.last_hit_taken = -99.0
        self.last_hit_dealt = -99.0
        self.sound_contacts: List[Tuple[float, float, str]] = []  # (bearing, t, dist bucket)
        self.bad_pickups: Dict[int, float] = {}
        self.strafe = 1.0
        self.radial_bias = 0.0
        self.next_flip = 0.0
        self.stuck_since: Optional[float] = None
        self.unstick_until = 0.0
        self.side = 1.0
        self.mode = "start"
        self.path: List[Tuple[float, float]] = []
        self.path_goal: Optional[Tuple[float, float]] = None  # goal the current path was planned to
        self.path_t = -99.0
        self.cover: Optional[Tuple[float, float]] = None
        self.cover_t = -99.0
        self.in_cover_mode = False
        self.t = 0.0
        self.tick_start = 0.0
        self.rad = 2.5
        self.zone_urgent_until = -1.0
        self.climbing = False
        self.target_id: Optional[int] = None
        self.goal_bad: Dict[Tuple[int, int], float] = {}  # goal cell -> time it was given up
        self.goal_key: Optional[Tuple[int, int]] = None
        self.goal_best_d = 1e9
        self.goal_best_t = 0.0

    # ================================================================== main
    def act(self, obs: mf.Observation) -> mf.Action:
        self.tick_start = time.perf_counter()
        self.t = t = obs["t"]
        me = obs["self"]
        self.rad = float(me["stats"].get("radius", 2.5))
        self.perceive(obs)
        act: Dict[str, Any] = {}

        if obs.get("offer"):
            act["pick"] = self.choose_upgrade(obs["offer"])

        enemy = self.pick_enemy(obs)
        # Being shot by someone we can't see (often in the back, x1.5): turn to them first.
        blind = self.unseen_attacker(obs)
        if blind and (enemy is None or enemy.id != blind):
            enemy = None
            self.target_id = None
        safe = self.is_safe(obs)
        # Regen needs regen_delay s without dealing OR taking damage: when hurt, don't shoot crates/drones.
        healing = me["hp"] < TUNING["heal_below_frac"] * me["max_hp"] and enemy is None
        farm = self.pick_farm_target(obs) if (enemy is None and safe and not healing) else None
        act.update(self.loot_action(obs))

        goal, urgent, mode, fight_move = self.decide(obs, enemy, farm, safe)
        self.mode = mode
        self.watch_goal(obs, goal, mode)
        move, jump, boost = self.navigate(obs, goal, urgent, enemy, fight_move)
        dodge = self.dodge(obs)
        if dodge is not None:
            move, boost = dodge
            self.mode = mode = mode + "+dodge"
        if not me["grounded"] and t >= self.unstick_until and not mode.startswith("zone!") and not self.climbing:
            boost = False  # boost in the air = jets: only on purpose
        aim, fire, reload = self.combat(obs, enemy, farm)
        if mode.startswith("cover") and enemy is not None and not self.has_los_to(me, enemy):
            fire = [False, False]  # don't give away the hide; let regen start

        act.update(move=move, jump=jump, boost=boost, aim=aim, fire=fire)
        if reload:
            act["reload"] = True

        if TUNING["debug"]:
            p = me["pos"]
            if goal:
                mf.draw.point([goal[0], goal[1], p[2] + 1], "#40c0ff", 6)
            for a, b in zip(self.path[:12], self.path[1:12]):
                mf.draw.line([a[0], a[1], p[2] + 1], [b[0], b[1], p[2] + 1], "#30a0ff")
            if enemy:
                ep = enemy.pos
                mf.draw.line([p[0], p[1], p[2] + 7], [ep[0], ep[1], ep[2] + 4.5], "#ff5040")
            mf.draw.label(f"{mode} hp{me['hp']:.0f} e{me['energy']:.0f}")
        return act

    @staticmethod
    def gkey(pt) -> Tuple[int, int]:
        return int(round(pt[0] / 5.0)), int(round(pt[1] / 5.0))

    def goal_ok(self, pt) -> bool:
        bad = self.goal_bad.get(self.gkey(pt))
        return bad is None or self.t - bad > 30.0

    def give_up(self, pt) -> None:
        self.goal_bad[self.gkey(pt)] = self.t
        self.path = []

    def watch_goal(self, obs, goal, mode) -> None:
        """Optional goals (loot, cubes, crates, drops) that stop getting closer are dropped for 30 s."""
        if goal is None or mode.split(":")[0] not in ("loot", "cube", "farm", "drop", "drift"):
            self.goal_key = None
            return
        d = hdist(obs["self"]["pos"], goal)
        k = self.gkey(goal)
        if k != self.goal_key:
            self.goal_key, self.goal_best_d, self.goal_best_t = k, d, self.t
        elif d < self.goal_best_d - 2.0:
            self.goal_best_d, self.goal_best_t = d, self.t
        elif self.t - self.goal_best_t > 6.0:
            self.give_up(goal)
            self.goal_key = None

    def time_ok(self) -> bool:
        return (time.perf_counter() - self.tick_start) * 1000 < TUNING["time_guard_ms"]

    # ================================================================== perception
    def perceive(self, obs: mf.Observation) -> None:
        t = obs["t"]
        me = obs["self"]
        for m in obs["mechs"]:
            if m["kind"] != "mech" or m["id"] == me["id"]:
                continue
            tr = self.tracks.get(m["id"]) or Track(m["id"])
            self.tracks[m["id"]] = tr
            tr.pos, tr.vel, tr.aim = list(m["pos"]), list(m["vel"]), list(m["aim"])
            tr.t = tr.seen_t = t
            tr.hp, tr.max_hp = m.get("hp", tr.hp), m.get("max_hp", tr.max_hp)
            tr.weapons = m.get("weapons", tr.weapons)
            tr.charging = bool(m.get("charging"))
            tr.cubes = m.get("cubes", 0)
            tr.est = False
        for ev in obs.get("events", []):
            et = ev.get("type")
            if et == "damaged":
                if ev.get("weapon") != "zone":
                    self.last_hit_taken = t
                if ev.get("bearing") is not None and ev.get("by") is not None:
                    by = int(ev["by"])
                    self.threat = (ev["bearing"], t, by)
                    if by < 100:  # a mech, not an NPC drone
                        tr = self.tracks.get(by)
                        if tr is None or t - tr.seen_t > 0.5:
                            # unseen attacker: guess its position along the bearing
                            tr = tr or Track(by)
                            self.tracks[by] = tr
                            d = {"lightning": 35.0, "gun": 90.0, "laser": 150.0, "mortar": 140.0}.get(ev.get("weapon"), 90.0)
                            if not tr.est and t - tr.seen_t < 3:
                                d = max(20.0, hdist(me["pos"], tr.pos))
                            b = ev["bearing"]
                            tr.pos = [me["pos"][0] + math.cos(b) * d, me["pos"][1] + math.sin(b) * d, me["pos"][2]]
                            tr.vel = [0.0, 0.0, 0.0]
                            tr.t = t
                            tr.est = True
                            if ev.get("weapon") in WEAPON_REACH and ev["weapon"] not in tr.weapons:
                                tr.weapons = list(tr.weapons) + [ev["weapon"]]
            elif et == "hit":
                self.last_hit_dealt = t
            elif et == "take_failed":
                self.bad_pickups[int(ev["id"])] = t
            elif et == "kill":
                self.tracks.pop(int(ev["target"]), None)
                mf.log(f"t={t:.1f} killed {ev['target']}")
        # sounds that matter (weapons/steps), kept 2 s
        for s in obs.get("sounds", []):
            if s["kind"] in ("gun", "laser", "laser_charge", "mortar", "lightning", "step", "boost", "jet") \
                    and s["dist"] != "far":
                self.sound_contacts.append((s["bearing"], t - s.get("age", 0), s["dist"]))
        self.sound_contacts = [c for c in self.sound_contacts if t - c[1] < 2.0][-12:]
        # forget stale tracks
        for k in [k for k, tr in self.tracks.items() if t - tr.t > TUNING["track_keep_s"]]:
            del self.tracks[k]
        # stuck detection: we asked to move but are barely moving on the ground
        speed = math.hypot(me["vel"][0], me["vel"][1])
        wants = self.mode.split("+")[0] not in ("hold", "capture", "cover-hold")
        if wants and me["grounded"] and speed < 1.5:
            if self.stuck_since is None:
                self.stuck_since = t
            elif t - self.stuck_since > 0.6 and t > self.unstick_until:
                self.unstick_until = t + 1.0
                self.side = -self.side
                self.stuck_since = None
                self.path = []  # replan
                self.path_t = -99
        else:
            self.stuck_since = None

    def unseen_attacker(self, obs) -> Optional[int]:
        if not self.threat or obs["t"] - self.threat[1] > 1.2 or self.threat[2] is None:
            return None
        by = self.threat[2]
        if any(m["id"] == by for m in obs["mechs"]):
            return None
        return by

    def is_safe(self, obs) -> bool:
        t = obs["t"]
        if t - self.last_hit_taken < TUNING["safe_after_s"]:
            return False
        for tr in self.tracks.values():
            if t - tr.seen_t < TUNING["safe_after_s"] and hdist(tr.pos, obs["self"]["pos"]) < 200:
                return False
        return True

    # ================================================================== targets
    def visible_enemies(self, obs) -> List[dict]:
        me = obs["self"]
        return [m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]

    def pick_enemy(self, obs: mf.Observation) -> Optional[Track]:
        """Best mech to engage: reported by the server now (it tracks real positions 5 s after view)."""
        me = obs["self"]
        best, best_score = None, 1e9
        cur_score = None
        for m in self.visible_enemies(obs):
            tr = self.tracks[m["id"]]
            d = math.dist(m["pos"], me["pos"])
            hp_frac = tr.hp / max(1.0, tr.max_hp)
            score = d + 90 * hp_frac
            if self.threat and self.threat[2] == tr.id and obs["t"] - self.threat[1] < 4:
                score -= 70  # whoever is shooting us
            if tr.charging and self.aimed_at_me(tr, me):
                score -= 40
            if d > 240:
                score += 200
            if tr.id == self.target_id:
                cur_score = score
            if score < best_score:
                best, best_score = tr, score
        if cur_score is not None and cur_score < best_score + 40:
            best = self.tracks[self.target_id]  # stick with the current target unless clearly worse
        self.target_id = best.id if best else None
        return best

    def aimed_at_me(self, tr: Track, me) -> bool:
        d = max(1.0, hdist(tr.pos, me["pos"]))
        return abs(wrap(tr.aim[0] - yaw_to(tr.pos, me["pos"]))) < 0.1 + math.atan2(4.0, d)

    def pick_farm_target(self, obs: mf.Observation) -> Optional[dict]:
        me = obs["self"]
        p = me["pos"]
        if any(obs["t"] - tr.seen_t < TUNING["farm_quiet_s"] for tr in self.tracks.values()):
            return None  # shooting crates is loud and blocks regen: not while enemies are around
        # Early on (before the hunters arrive) cubes are worth a detour: 1 cube = +4% HP/damage and the
        # first upgrade offer (plating, +15% max HP), which decides close duels.
        early = TUNING["early_farm_mult"] if obs["t"] < TUNING["early_farm_s"] else 1.0
        best, bd = None, TUNING["crate_radius"] * early
        for c in obs["crates"]:
            d = hdist(p, c["pos"])
            if d < bd and abs(c["pos"][2] - p[2]) < 20 and self.goal_ok(c["pos"]):
                best, bd = dict(c, kind="crate", vel=[0, 0, 0]), d
        if me["hp"] / me["max_hp"] >= TUNING["npc_farm_min_hp_frac"]:
            for n in obs["mechs"]:
                if n["kind"] == "npc":
                    d = hdist(p, n["pos"])
                    if d < min(bd, 100) and self.goal_ok(n["pos"]):
                        best, bd = n, d
        return best

    def has_los_to(self, me, tr) -> bool:
        if self.world is None:
            return True
        p = me["pos"]
        return self.world.los((p[0], p[1], p[2] + me["stats"].get("muzzle_height", 7)),
                              (tr.pos[0], tr.pos[1], tr.pos[2] + 4.5))

    # ================================================================== loot & upgrades
    def choose_upgrade(self, offer: dict) -> int:
        prio = TUNING["upgrade_priority"]
        ranks = [prio.index(c["id"]) if c["id"] in prio else 99 for c in offer["choices"]]
        return ranks.index(min(ranks))

    def held(self, me: dict) -> List[Optional[str]]:
        return [w["kind"] if w else None for w in me["weapons"]]

    def pickup_value(self, me: dict, kind: str) -> Tuple[float, Optional[int]]:
        """(gain, slot to drop or None). gain <= 0 means not worth it."""
        rank = TUNING["weapon_rank"]
        held = self.held(me)
        if kind in held:
            return 0, None
        for i, h in enumerate(held):
            if h is None:
                return rank.get(kind, 0) + 1, None
        cand = [(rank.get(h, 0), i) for i, h in enumerate(held) if h and h != "gun"]
        if not cand:
            return 0, None
        worst, slot = min(cand)
        busy = me["weapons"][slot] and me["weapons"][slot].get("charge", 0) > 0
        gain = rank.get(kind, 0) - worst
        return (gain, slot) if gain > 0 and not busy else (0, None)

    def loot_action(self, obs: mf.Observation) -> Dict[str, Any]:
        me = obs["self"]
        reach = self.cfg["loot"]["pickup_radius"] - 0.3
        for pk in obs["pickups"]:
            if int(pk["id"]) in self.bad_pickups and obs["t"] - self.bad_pickups[int(pk["id"])] < 5:
                continue
            if hdist(me["pos"], pk["pos"]) < reach and abs(pk["pos"][2] - me["pos"][2]) < 4:
                gain, drop = self.pickup_value(me, pk["kind"])
                if gain > 0:
                    out = {"take": pk["id"]}
                    if drop is not None:
                        out["drop"] = drop
                    return out
        return {}

    # ================================================================== fight assessment
    def my_dps(self, me) -> float:
        cube = 1 + self.cfg["cubes"].get("damage_per_cube", 0.04) * min(me["cubes"], self.cfg["cubes"].get("stat_cap", 10))
        return cube * sum(BASE_DPS.get(w["kind"], 0) for w in me["weapons"] if w)

    def their_dps(self, tr: Track, dist: float) -> float:
        cube = 1 + 0.04 * min(tr.cubes, 10)
        return cube * sum(BASE_DPS.get(k, 0) for k in tr.weapons if dist < WEAPON_REACH.get(k, 230))

    def threats(self, obs) -> List[Track]:
        t = obs["t"]
        return [tr for tr in self.tracks.values() if t - tr.t < 6.0 and hdist(tr.pos, obs["self"]["pos"]) < 260]

    def should_disengage(self, obs, enemy: Track) -> bool:
        me = obs["self"]
        threats = self.threats(obs)
        incoming = sum(self.their_dps(tr, hdist(tr.pos, me["pos"])) for tr in threats) or 1.0
        armor = 1 - me["stats"].get("armor", 0)
        t_me = me["hp"] / (incoming * armor)
        t_them = enemy.hp / max(1.0, self.my_dps(me))
        return t_me < TUNING["disengage_ratio"] * t_them

    # ================================================================== goals
    def zone_status(self, obs: mf.Observation) -> Tuple[Optional[List[float]], bool]:
        """(point to head for or None, urgent?)"""
        me, z, t = obs["self"], obs["zone"], obs["t"]
        p = me["pos"]
        walk = me["stats"].get("walk_speed", 11.7) * 0.8  # paths aren't straight
        nc, nr = z["next_center"], z["next_radius"]
        d_next = hdist(p, nc)
        inner = nr * TUNING["zone_inner_frac"]
        need_s = max(0.0, d_next - inner) / walk
        outside = not me["in_zone"]
        time_left = z["shrink_end"] - t
        urgent = outside or need_s + TUNING["zone_slack_s"] > time_left
        if need_s <= 0 and not outside:
            return None, False
        if d_next < 1e-3 or nr <= 0:
            return [nc[0], nc[1]], urgent
        yaw = yaw_to(nc, p)
        return [nc[0] + math.cos(yaw) * inner, nc[1] + math.sin(yaw) * inner], urgent

    def in_next_circle(self, obs, pt, margin=0.0) -> bool:
        z = obs["zone"]
        return hdist(pt, z["next_center"]) < z["next_radius"] - margin

    def decide(self, obs, enemy: Optional[Track], farm, safe: bool):
        """-> (goal xy or None, urgent, mode name, fight_move flag)"""
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        hp_frac = me["hp"] / me["max_hp"]
        zone_pt, urgent = self.zone_status(obs)
        if urgent:
            self.zone_urgent_until = t + 3.0
        elif zone_pt and t < self.zone_urgent_until:
            urgent = True  # sticky: don't toggle between running for the zone and other goals
        if urgent and zone_pt:
            self.in_cover_mode = False
            return zone_pt, True, "zone!", False

        threats = self.threats(obs)
        recently_hit = t - self.last_hit_taken < 3.0

        # ---- cover / regen: losing the trade, or hurt with threats around
        want_cover = False
        if enemy is not None and self.should_disengage(obs, enemy):
            want_cover = True
        if hp_frac < TUNING["cover_hp_frac"] and (threats or recently_hit):
            want_cover = True
        if self.in_cover_mode and hp_frac < TUNING["reengage_hp_frac"] and (threats or recently_hit):
            want_cover = True
        if want_cover and threats:
            cv = self.find_cover(obs, threats)
            if cv is not None:
                self.in_cover_mode = True
                if hdist(p, cv) < 2.5:
                    return None, False, "cover-hold", False
                return cv, False, "cover", False
            # no cover: back off from the threats while facing them
            if enemy is not None:
                return None, False, "retreat", True
        self.in_cover_mode = False

        if enemy is not None:
            return None, False, "fight", True

        # an attacker we can't see: turn to it (combat) and keep moving sideways
        if recently_hit:
            return zone_pt, False, "evade", False

        # hurt and nobody in view: heal before the next hunter arrives. Hide from recently seen enemies
        # (cubes/weapons right next to us are still fine: picking up deals no damage).
        if hp_frac < TUNING["heal_below_frac"] and not farm:
            recent = [tr for tr in self.tracks.values() if t - tr.t < 15.0]
            near_cube = next((c for c in obs["cubes"] if hdist(p, c["pos"]) < 25 and abs(c["pos"][2] - p[2]) < 6
                              and self.goal_ok(c["pos"])), None)
            if near_cube:
                return near_cube["pos"], False, "heal:cube", False
            if recent:
                cv = self.find_cover(obs, recent)
                if cv is not None:
                    return (cv if hdist(p, cv) > 2.5 else None), False, "heal:cover", False
            return None, False, "heal", False

        if not (safe or not threats):
            # Not fully safe, but grab cubes close by (e.g. a kill's drop) if no enemy is near them.
            near = [tr for tr in threats if t - tr.seen_t < 3.0]
            best, bd = None, 40.0
            for c in obs["cubes"]:
                d = hdist(p, c["pos"])
                if d < bd and abs(c["pos"][2] - p[2]) < 6 and self.goal_ok(c["pos"]) and \
                        all(hdist(tr.pos, c["pos"]) > 60 for tr in near):
                    best, bd = c, d
            if best:
                return best["pos"], False, "cube", False
        if safe or not threats:
            # better weapon nearby
            best, bd = None, TUNING["loot_radius"]
            for pk in obs["pickups"]:
                if int(pk["id"]) in self.bad_pickups and t - self.bad_pickups[int(pk["id"])] < 5:
                    continue
                if abs(pk["pos"][2] - p[2]) > 6 or not self.goal_ok(pk["pos"]):
                    continue
                d = hdist(p, pk["pos"])
                if d < bd and self.pickup_value(me, pk["kind"])[0] > 0:
                    best, bd = pk, d
            if best:
                return best["pos"], False, f"loot:{best['kind']}", False
            best, bd = None, TUNING["cube_radius"] * (TUNING["early_farm_mult"] if t < TUNING["early_farm_s"] else 1.0)
            for c in obs["cubes"]:
                d = hdist(p, c["pos"])
                if d < bd and abs(c["pos"][2] - p[2]) < 6 and self.goal_ok(c["pos"]):
                    best, bd = c, d
            if best:
                return best["pos"], False, "cube", False

        # supply drop: 3 cubes + full heal for standing alone in the ring
        for dr in obs.get("drops", []):
            d = hdist(p, dr["pos"])
            walk = me["stats"].get("walk_speed", 11.7)
            if d > TUNING["drop_max_dist"] or dr["lands_in"] > d / walk + 10:
                continue
            if hp_frac < TUNING["drop_min_hp_frac"]:
                continue
            if any(hdist(tr.pos, dr["pos"]) < TUNING["drop_enemy_clear"] for tr in self.tracks.values()
                   if t - tr.t < 20.0):
                continue
            if dr.get("holder") not in (None, me["id"]) or not self.goal_ok(dr["pos"]):
                continue
            if d < dr["radius"] * 0.5:
                return None, False, "capture", False
            return dr["pos"], False, "drop", False

        if farm:
            d = hdist(p, farm["pos"])
            fh = 2.4 if farm.get("kind") == "crate" else 3.5
            fp = farm["pos"]
            if d < 110 and self.world is not None and self.world.los(
                    (p[0], p[1], p[2] + me["stats"].get("muzzle_height", 7)), (fp[0], fp[1], fp[2] + fh * 0.5)):
                return zone_pt, False, "farm-shoot", False  # shoot from here (combat aims at it)
            if d > 25:
                return farm["pos"], False, "farm", False
            return None, False, "hold", False

        if zone_pt:
            return zone_pt, False, "zone", False
        # drift toward the middle of the next circle (fewer flanks at the end)
        z = obs["zone"]
        if hdist(p, z["next_center"]) > z["next_radius"] * 0.35:
            return list(z["next_center"]), False, "drift", False
        return None, False, "hold", False

    def find_cover(self, obs, threats: List[Track]) -> Optional[Tuple[float, float]]:
        """Nearby walkable spot with no line of sight from the closest threats. Cached ~0.6 s."""
        if self.world is None or not self.time_ok():
            return self.cover
        t = obs["t"]
        me = obs["self"]
        p = me["pos"]
        if self.cover and t - self.cover_t < 0.6:
            return self.cover
        th = sorted(threats, key=lambda tr: hdist(tr.pos, p))[:3]
        best, best_s = None, 1e9
        w = self.world
        for r in (7.0, 13.0, 20.0, 30.0):
            for k in range(12):
                if not self.time_ok():
                    break
                yaw = k * math.pi / 6
                x, y = p[0] + math.cos(yaw) * r, p[1] + math.sin(yaw) * r
                if self.nav is not None and not self.nav.is_free(*self.nav.cell_of(x, y)):
                    continue
                if w.probe(p[0], p[1], p[2], yaw, r, self.step_up, 9.0, self.max_slope, radius=self.rad) < r - 0.5:
                    continue
                gz = w.floor_under(x, y, p[2], self.step_up)
                mid = (x, y, gz + 4.5)
                top = (x, y, gz + 8.5)
                hidden = True
                for tr in th:
                    eye = (tr.pos[0], tr.pos[1], tr.pos[2] + 8)
                    if w.los(eye, mid, skip_start=0) or w.los(eye, top, skip_start=0):
                        hidden = False
                        break
                if not hidden:
                    continue
                s = r
                if not self.in_next_circle(obs, (x, y), 5):
                    s += 40
                # prefer spots that don't move toward threats
                for tr in th:
                    s -= 0.3 * (hdist((x, y), tr.pos) - hdist(p, tr.pos))
                if s < best_s:
                    best, best_s = (x, y), s
            if best is not None and r >= 13:
                break
        self.cover, self.cover_t = best, t
        return best

    # ================================================================== movement
    def navigate(self, obs, goal, urgent: bool, enemy: Optional[Track], fight_move: bool):
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        energy = me["energy"]
        boost = jump = False
        mvec = (0.0, 0.0)

        self.climbing = False
        exit_pt = self.pocket_exit(p, goal) if goal is not None else None
        if exit_pt is not None and not (fight_move and enemy is not None and hdist(p, enemy.pos) < 60):
            # Walled-in pocket (courtyard): the only way out is up. Fill energy, then jump + jet over the wall.
            self.climbing = True
            self.mode += "+climb"
            full = me["energy"] >= 0.9 * me.get("max_energy", 100)
            airborne = not me["grounded"]
            if airborne or full:
                mvec = unit(yaw_to(p, exit_pt))
                jump = me["grounded"]
                boost = me["energy"] > 1
            n = math.hypot(*mvec)
            return [round(mvec[0], 3), round(mvec[1], 3)], jump, boost
        if fight_move and enemy is not None:
            mvec, boost = self.fight_move(obs, enemy, retreat=self.mode == "retreat")
        elif goal is not None:
            d = hdist(p, goal)
            if d > 1.5:
                y = self.route_yaw(obs, goal)
                mvec = unit(y)
                if d < 6:
                    mvec = (mvec[0] * max(0.4, d / 6), mvec[1] * max(0.4, d / 6))
                boost = (urgent or not me["in_zone"] or self.mode in ("cover",)) and \
                    energy > TUNING["boost_min_energy"] * (0.3 if not me["in_zone"] else 1)
            # an attacker we can't see / running for the zone under fire: weave
            if self.mode in ("evade", "zone!") and t - self.last_hit_taken < 3:
                self.update_strafe(t)
                b = self.threat[0] if self.threat else 0.0
                mvec = (mvec[0] * 0.8 + math.cos(b + math.pi / 2) * 0.5 * self.strafe,
                        mvec[1] * 0.8 + math.sin(b + math.pi / 2) * 0.5 * self.strafe)
        elif self.mode == "evade" and self.threat:
            self.update_strafe(t)
            b = self.threat[0] + math.pi / 2 * self.strafe
            mvec = unit(self.steer(p, b, me))

        # unstick: jump and jet while pushing on, then try the other side
        if t < self.unstick_until:
            jump = me["grounded"]
            boost = energy > 8
            if mvec == (0.0, 0.0) and goal is not None:
                mvec = unit(yaw_to(p, goal))
            esc = self.escape_point(p)
            if esc is not None:  # wedged somewhere unwalkable: back out to open ground first
                mvec = unit(yaw_to(p, esc))
        n = math.hypot(*mvec)
        if n > 1:
            mvec = (mvec[0] / n, mvec[1] / n)
        return [round(mvec[0], 3), round(mvec[1], 3)], jump, boost

    def pocket_exit(self, p, goal) -> Optional[Tuple[float, float]]:
        """If we're on the ground in a small walled-in region and the goal is outside it: where to jet to."""
        nav = self.nav
        if nav is None or nav.main < 0:
            return None
        i, j = nav.cell_of(p[0], p[1])
        if p[2] > nav.ground[j * nav.nx + i] + 3.0:
            return None  # on a roof / in the air above ground level: normal steering handles it
        mine = nav.component_at(p[0], p[1])
        if mine < 0 or mine == nav.main or nav.comp_size[mine] > 2500:
            return None
        if nav.component_at(goal[0], goal[1]) == mine:
            return None
        return nav.nearest_in_component(p[0], p[1], nav.main)

    def escape_point(self, p) -> Optional[Tuple[float, float]]:
        """If we stand in a cell the nav grid calls unwalkable (wedged in a gap, on a ledge), the center
        of the nearest walkable cell; else None."""
        if self.nav is None:
            return None
        i, j = self.nav.cell_of(p[0], p[1])
        if self.nav.is_free(i, j):
            return None
        g = self.nav.ground[j * self.nav.nx + i]
        if p[2] > g + 3.0:
            return None  # on a roof: walking off the edge is fine
        c = self.nav.nearest_free(i, j, 6)
        return self.nav.center(*c) if c else None

    def route_yaw(self, obs, goal) -> float:
        """Heading toward goal: straight if walkable, else along an A* path."""
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        gy = yaw_to(p, goal)
        d = hdist(p, goal)
        if self.world is None:
            return gy
        h = me["stats"].get("height", 9.0)
        look = min(d, 24.0)
        on_roof = self.nav is not None and p[2] > self.nav.ground[self.nav.cell_of(p[0], p[1])[1] * self.nav.nx +
                                                                   self.nav.cell_of(p[0], p[1])[0]] + 3.0
        have_path = bool(self.path) and self.path_goal is not None and \
            hdist(self.path_goal, goal) <= 20.0 and t - self.path_t <= 10.0
        # Go straight only when we aren't already following a path, or the goal is close and clear.
        # (Hysteresis: a path exists because the straight line is blocked further on than the probe
        # looks; dropping it whenever the next 24 m look clear makes the bot dither in place.)
        if self.nav is None or on_roof or (
                (not have_path or d < 25) and
                self.world.probe(p[0], p[1], p[2], gy, look, self.step_up, h, self.max_slope,
                                 radius=self.rad) >= look - 0.1):
            self.path = []
            return self.steer(p, gy, me)
        # need a path: (re)plan when there is none, the goal moved > 20 m from where the path ends,
        # or the path is old. Small goal drift (a shrinking zone) must not trigger replans.
        stale = not have_path
        if stale and self.time_ok():
            res = self.nav.plan((p[0], p[1]), (goal[0], goal[1]), budget_ms=min(10.0, max(2.0, TUNING["time_guard_ms"] - (time.perf_counter() - self.tick_start) * 1000 - 6)))
            if res is not None:
                self.path, self.path_goal, self.path_t = list(res), (goal[0], goal[1]), t
                if not res and self.mode.split(":")[0] in ("loot", "cube", "farm", "drop", "drift"):
                    self.give_up(goal)
                if self.path:  # start from the nearest waypoint, never walk back to the start cell
                    k = min(range(min(len(self.path), 12)), key=lambda i: hdist(p, self.path[i]))
                    self.path = self.path[k:]
        if self.path:
            # drop reached waypoints, and any waypoint that is behind us relative to the next one
            while len(self.path) > 1 and (hdist(p, self.path[0]) < 4.0 or
                                          hdist(p, self.path[1]) < hdist(self.path[0], self.path[1])):
                self.path.pop(0)
            # if we've wandered off the path, rejoin at the nearest waypoint
            if hdist(p, self.path[0]) > 20:
                k = min(range(min(len(self.path), 40)), key=lambda i: hdist(p, self.path[i]))
                self.path = self.path[k:]
            tgt = self.path[0]
            for wp in self.path[1:8]:
                dd = hdist(p, wp)
                y = yaw_to(p, wp)
                if self.world.probe(p[0], p[1], p[2], y, dd, self.step_up, h, self.max_slope, radius=self.rad) >= dd - 0.5:
                    tgt = wp
                else:
                    break
            return self.steer(p, yaw_to(p, tgt), me, look=min(8.0, hdist(p, tgt)))
        return self.steer(p, gy, me)

    def steer(self, p, goal_yaw: float, me, look: float = 9.0) -> float:
        """Pick the walkable direction closest to goal_yaw, using the map probe."""
        if self.world is None:
            return goal_yaw
        h = me["stats"].get("height", 9.0)
        look = max(3.0, look)
        for off in (0, 20, -20, 45, -45, 70, -70, 100, -100, 140, -140):
            y = goal_yaw + math.radians(off) * self.side
            if self.world.probe(p[0], p[1], p[2], y, look, self.step_up, h, self.max_slope, radius=self.rad) >= look - 0.1:
                return y
        return goal_yaw  # boxed in: push on and let unstick (jets) handle it

    def update_strafe(self, t):
        if t > self.next_flip:
            r = self.rng.random()
            self.strafe = -self.strafe if r < 0.75 else self.strafe
            self.radial_bias = self.rng.uniform(-0.4, 0.4)
            self.next_flip = t + self.rng.uniform(*TUNING["strafe_period"])

    def fight_move(self, obs, enemy: Track, retreat: bool = False):
        """Keep a weapon-appropriate range, juke sideways unpredictably, back off when losing."""
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        self.update_strafe(t)
        d = hdist(p, enemy.pos)
        to = yaw_to(p, enemy.pos)
        held = [w for w in me["weapons"] if w]
        # range: the best weapon that's ready decides
        want = 65.0
        for w in held:
            if w["kind"] == "lightning":
                want = min(want, TUNING["fight_range"]["lightning"])
            elif w["kind"] in ("laser", "mortar"):
                want = max(want, TUNING["fight_range"][w["kind"]]) if want > 30 else want
        if "lightning" in enemy.weapons and not any(w["kind"] == "lightning" for w in held):
            want = max(want, 60.0)  # stay out of arc range
        if retreat:
            radial = -1.0
        else:
            radial = max(-1.0, min(1.0, (d - want) / 25.0)) + self.radial_bias
            finishing = enemy.hp < TUNING["finish_hp_frac"] * max(1.0, enemy.max_hp)
            if not TUNING["chase"] and not finishing:
                radial = min(radial, 0.0)  # cautious: let them come to us; never walk into them
        strafe = 0.9 if not retreat else 0.5
        if TUNING["two_front"]:
            # Another enemy nearby: strafe the way that moves us away from it, so both stay in our front
            # 180 deg arc (hits from behind do x1.5) instead of letting it flank us.
            others = [tr for tr in self.threats(obs) if tr.id != enemy.id and t - tr.seen_t < 3.0]
            if others:
                o2 = min(others, key=lambda tr: hdist(p, tr.pos))
                side = math.cos(to + math.pi / 2) * (o2.pos[0] - p[0]) + math.sin(to + math.pi / 2) * (o2.pos[1] - p[1])
                self.strafe = -1.0 if side > 0 else 1.0
        mx = math.cos(to) * radial + math.cos(to + math.pi / 2) * strafe * self.strafe
        my = math.sin(to) * radial + math.sin(to + math.pi / 2) * strafe * self.strafe
        y = math.atan2(my, mx)
        y = self.steer(p, y, me, look=6.0)
        boost = ((retreat or d < 30) and me["energy"] > TUNING["boost_min_energy"]) or             me["energy"] > TUNING["fight_boost_energy"]
        return unit(y), boost

    def dodge(self, obs):
        """Sidestep a laser aimed at us or get out of a mortar shell's splash. -> (move, boost) or None."""
        me = obs["self"]
        p = me["pos"]
        # mortar shells: predict where they land
        g = self.cfg["weapons"]["mortar"]["gravity"]
        R = self.cfg["weapons"]["mortar"]["splash_radius"] + me["stats"].get("radius", 2.5) + 2.5
        for pr in obs.get("projectiles", []):
            if pr["kind"] != "mortar":
                continue
            x, y, z = pr["pos"]
            vx, vy, vz = pr["vel"]
            # solve z(t) = my feet height
            a, b, c = -0.5 * g, vz, z - p[2]
            disc = b * b - 4 * a * c
            if disc < 0:
                continue
            tl = (-b - math.sqrt(disc)) / (2 * a)
            if tl < 0 or tl > 4:
                continue
            lx, ly = x + vx * tl, y + vy * tl
            if math.hypot(lx - p[0], ly - p[1]) < R:
                away = math.atan2(p[1] - ly, p[0] - lx)
                return list(unit(self.steer(p, away, me, look=6.0))), me["energy"] > 10
        # lasers charging at us
        for m in self.visible_enemies(obs):
            tr = self.tracks[m["id"]]
            if tr.charging and self.aimed_at_me(tr, me) and hdist(tr.pos, p) < 290:
                side = yaw_to(tr.pos, p) + math.pi / 2 * self.strafe
                return list(unit(self.steer(p, side, me, look=6.0))), me["energy"] > 15
        return None

    # ================================================================== combat
    def combat(self, obs, enemy: Optional[Track], farm: Optional[dict]):
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        st = me["stats"]
        muzzle = (p[0], p[1], p[2] + st.get("muzzle_height", 7.0))
        weapons = me["weapons"]
        fire = [False, False]
        reload = False
        cur_yaw, cur_pitch = me["aim"]

        if enemy is not None:
            tp, tvel = enemy.pos, enemy.vel
            h = 9.0 * (1 + self.cfg["cubes"].get("size_per_cube", 0.05) * enemy.cubes)
            r = 2.5 * h / 9.0
            kind = "mech"
        elif farm is not None:
            tp, tvel = farm["pos"], farm.get("vel", (0, 0, 0))
            kind = farm.get("kind")
            if kind == "npc":
                h, r = self.cfg["npc"].get("height", 3.5), self.cfg["npc"].get("radius", 1.8)
            else:
                h, r = self.cfg["crates"].get("height", 2.4), self.cfg["crates"].get("radius", 1.4)
        else:
            aim = self.idle_aim(obs)
            # Start every fight with a full magazine: top up whenever nobody has been seen for a moment.
            quiet = all(t - tr.seen_t > TUNING["reload_quiet_s"] for tr in self.tracks.values())
            for i, w in enumerate(weapons):
                if w and w["kind"] == "gun" and not w.get("reload") and \
                        w.get("ammo", 99) < w["stats"].get("mag", 12) * (1.0 if quiet else 0.6):
                    reload = True
            return [aim[0], aim[1]], fire, reload

        center = (tp[0], tp[1], tp[2] + h * 0.5)
        dist = math.dist(muzzle, center)
        clear = self.world.los(muzzle, center) if self.world else True
        if not clear and self.world is not None and kind == "mech":
            top = (tp[0], tp[1], tp[2] + h * 0.85)
            if self.world.los(muzzle, top):
                center, clear = top, True

        aim = list(aim_at(muzzle, center))
        gun_slot = next((i for i, w in enumerate(weapons) if w and w["kind"] == "gun"), None)
        if gun_slot is not None:
            g = weapons[gun_slot]
            pt, _ = lead_point(muzzle, center, tvel, g["stats"]["speed"])
            aim = list(aim_at(muzzle, pt))
        laser = next((w for w in weapons if w and w["kind"] == "laser"), None)
        if laser is not None and (laser.get("charge", 0) > 0 or gun_slot is None):
            # hitscan: aim where it'll be when we release (~1 decision)
            pt = (center[0] + tvel[0] * 0.1, center[1] + tvel[1] * 0.1, center[2])
            aim = list(aim_at(muzzle, pt))

        # mortar takes over the aim when it's ready and it's a mech (lob over cover too)
        for i, w in enumerate(weapons):
            if not w or w["kind"] != "mortar" or kind != "mech":
                continue
            ws = w["stats"]
            hd = hdist(p, tp)
            gun_busy = gun_slot is None or weapons[gun_slot].get("reload", 0) > 0 or not clear
            if w["cooldown"] <= 0.1 and TUNING["mortar_min_range"] < hd < 0.95 * ws["speed"] ** 2 / ws["gravity"] \
                    and (gun_busy or hd > 60 or enemy is None or enemy.vel[0] ** 2 + enemy.vel[1] ** 2 < 4):
                tube = p[2] + st.get("height", 9.0) + 0.5
                aim_pt, pitch = list(tp), None
                for _ in range(3):  # lead by flight time
                    pitch = ballistic_pitch(hdist(p, aim_pt), aim_pt[2] + 1.0 - tube, ws["speed"], ws["gravity"],
                                            high=not clear)
                    if pitch is None:
                        break
                    ft = flight_time(hdist(p, aim_pt), pitch, ws["speed"])
                    aim_pt = [tp[0] + tvel[0] * ft, tp[1] + tvel[1] * ft, tp[2]]
                if pitch is not None and pitch <= st.get("max_pitch", 1.3):
                    aim = [yaw_to(p, aim_pt), pitch]
                    fire[i] = w["cooldown"] <= 0 and self.aligned(me, aim, 0.03, 0.025)

        yaw_err = abs(wrap(aim[0] - cur_yaw))
        pitch_err = abs(aim[1] - cur_pitch)
        tol_yaw = math.atan2(r * 0.85, max(dist, 1.0)) + 0.008
        tol_pitch = math.atan2(h * 0.4, max(dist, 1.0)) + 0.008
        on_target = yaw_err < tol_yaw and pitch_err < tol_pitch

        for i, w in enumerate(weapons):
            if not w or fire[i]:
                continue
            k, ws = w["kind"], w["stats"]
            if k == "gun":
                fire[i] = clear and on_target and dist < ws["range"] and w.get("reload", 0) <= 0
                if w.get("ammo", 1) == 0 and not w.get("reload"):
                    reload = True
            elif k == "laser":
                in_range = clear and dist < ws["range"]
                charge = w.get("charge", 0)
                if kind != "mech" and dist > 60:
                    in_range = False  # don't waste the laser on far crates
                if in_range and w["cooldown"] <= 0:
                    full = charge >= ws["charge"] - 0.05
                    fire[i] = not (full and on_target)
                else:
                    fire[i] = False  # release (fires if charged) / idle
            elif k == "lightning":
                rng_ok = dist < ws["range"] * 0.95
                cone = math.radians(ws["cone_deg"]) * 0.8
                fire[i] = rng_ok and abs(wrap(yaw_to(p, tp) - cur_yaw)) < cone and clear and \
                    me["energy"] > TUNING["lightning_min_energy"] and w["cooldown"] <= 0
        return aim, fire, reload

    def aligned(self, me, aim, ty, tp) -> bool:
        return abs(wrap(aim[0] - me["aim"][0])) < ty and abs(aim[1] - me["aim"][1]) < tp

    def idle_aim(self, obs) -> Tuple[float, float]:
        """No target in view. Face (in order): the last attacker, a remembered enemy, a recent sound;
        else sweep around the walking direction to widen the 120 deg vision wedge."""
        me, t = obs["self"], obs["t"]
        p = me["pos"]
        if self.threat and t - self.threat[1] < 3.0:
            return self.threat[0], 0.0
        recent = [tr for tr in self.tracks.values() if t - tr.t < 10.0 and hdist(tr.pos, p) < 250]
        if recent:
            tr = min(recent, key=lambda tr: hdist(tr.pos, p))
            return yaw_to(p, tr.pos), 0.0
        if self.sound_contacts:
            return self.sound_contacts[-1][0], 0.0
        v = me["vel"]
        base = math.atan2(v[1], v[0]) if math.hypot(v[0], v[1]) > 1 else me["aim"][0]
        return wrap(base + 1.0 * math.sin(t * 1.4)), 0.0
