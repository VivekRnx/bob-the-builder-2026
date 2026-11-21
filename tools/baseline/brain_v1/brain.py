"""The bot's decision loop. One Brain per match; act() is called ~10x per match-second.

Pipeline each tick:
    perceive()  -> update memory (threats from damage events, failed pickups, stuck detection)
    pick_enemy()-> who to fight (visible mechs), or a farm target (crate / NPC drone)
    choose_goal()-> where the legs go: zone safety > loot > cubes > supply drop > crates > zone drift
    navigate()  -> turn a goal into a move vector, avoiding walls with world.probe, unsticking with jets
    combat()    -> aim + fire for each slot (gun lead, mortar ballistics, laser charge/release, lightning)

Every number worth tuning is in TUNING below. Read live rule numbers from hello.config, never hard-code.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import mechfight as mf  # noqa

from .geom import aim_at, ballistic_pitch, flight_time, hdist, lead_point, unit, wrap, yaw_to
from .world import World, load_world

TUNING = {
    # weapons: higher = better to hold in slot 1 (slot 0 keeps the gun)
    "weapon_rank": {"laser": 3, "mortar": 2, "lightning": 1, "gun": 0},
    # upgrade ids in order of preference; unknown ids fall to the end (protocol.md section 10)
    # Max HP first: the zone ignores armor, so in the final burn the biggest HP pool lives longest.
    "upgrade_priority": ["plating", "armor", "nanites", "gun_rate", "gun_damage", "laser_damage",
                         "laser_charge", "gyros", "mortar_reload", "lightning_efficiency", "gun_mag",
                         "gun_velocity", "sensors", "servos", "reactor", "capacitor", "jets",
                         "mortar_splash", "mortar_velocity", "lightning_range", "lightning_chain",
                         "laser_pierce"],
    "loot_radius": 70.0,          # detour this far for a better weapon
    "cube_radius": 80.0,          # detour this far for a power cube
    "crate_radius": 110.0,        # shoot crates within this range when nothing is hunting us
    "npc_farm_min_hp_frac": 0.6,  # only farm NPC drones when healthy
    "drop_max_dist": 180.0,       # consider supply drops this close
    "retreat_hp_frac": 0.35,      # back off below this HP fraction if the enemy is healthier
    "zone_inner_frac": 0.6,       # aim for this fraction of the next circle's radius
    "zone_slack_s": 12.0,         # start moving for the next circle this many seconds early
    "boost_min_energy": 35.0,     # keep this much energy in reserve (for jets / escapes)
    "lightning_min_energy": 20.0,
    "strafe_period": (0.8, 2.0),  # seconds between strafe-direction flips
    "fight_range": {"lightning": 28.0, "laser": 110.0, "mortar": 120.0, "gun": 70.0},
    "mortar_min_range": 28.0,     # don't splash ourselves
    "debug": True,
}


class Brain:
    name = "baseline-v1"

    def __init__(self, hello: mf.Hello):
        self.cfg = hello["config"]
        self.me_id = int(hello["slot"])
        self.rng = random.Random(hello["seed"])
        t0 = time.perf_counter()
        self.world: Optional[World] = load_world(hello.get("map", {}).get("dir"))
        mf.log(f"brain: slot {self.me_id}, map {'ok' if self.world else 'MISSING'} "
               f"({(time.perf_counter() - t0) * 1000:.0f} ms)")
        mech = self.cfg["mech"]
        self.step_up = mech.get("step_up", 1.2)
        self.max_slope = mech.get("max_slope_deg", 50)
        # memory
        self.threat: Optional[Tuple[float, float, Optional[int]]] = None  # (bearing, t, attacker id)
        self.last_attacker: Optional[int] = None
        self.bad_pickups: Dict[int, float] = {}  # pickup id -> time we gave up on it
        self.strafe = 1.0
        self.next_flip = 0.0
        self.stuck_since: Optional[float] = None
        self.unstick_until = 0.0
        self.side = 1.0
        self.mode = "start"

    # ================================================================== main
    def act(self, obs: mf.Observation) -> mf.Action:
        t = obs["t"]
        me = obs["self"]
        self.perceive(obs)
        act: Dict[str, Any] = {}

        if obs.get("offer"):
            act["pick"] = self.choose_upgrade(obs["offer"])

        enemy = self.pick_enemy(obs)
        farm = None if enemy else self.pick_farm_target(obs)
        take = self.loot_action(obs)
        act.update(take)

        goal, urgent, mode = self.choose_goal(obs, enemy, farm)
        self.mode = mode
        move, jump, boost = self.navigate(obs, goal, urgent, enemy)
        aim, fire, reload = self.combat(obs, enemy or farm, enemy is not None)

        act.update(move=move, jump=jump, boost=boost, aim=aim, fire=fire)
        if reload:
            act["reload"] = True

        if TUNING["debug"]:
            p = me["pos"]
            if goal:
                mf.draw.point([goal[0], goal[1], p[2] + 1], "#40c0ff", 6)
            tgt = enemy or farm
            if tgt:
                mf.draw.line([p[0], p[1], p[2] + 7], [tgt["pos"][0], tgt["pos"][1], tgt["pos"][2] + 3], "#ff5040")
            mf.draw.label(f"{mode} hp{me['hp']:.0f} e{me['energy']:.0f}")
        return act

    # ================================================================== perception
    def perceive(self, obs: mf.Observation) -> None:
        t = obs["t"]
        for ev in obs.get("events", []):
            et = ev.get("type")
            if et == "damaged" and ev.get("bearing") is not None:
                self.threat = (ev["bearing"], t, ev.get("by"))
                if ev.get("by") is not None:
                    self.last_attacker = int(ev["by"])
            elif et == "take_failed":
                self.bad_pickups[int(ev["id"])] = t
            elif et == "kill":
                mf.log(f"t={t:.1f} killed {ev['target']}")
        # stuck detection: we asked to move but are barely moving on the ground
        me = obs["self"]
        speed = math.hypot(me["vel"][0], me["vel"][1])
        wants = self.mode not in ("hold", "capture")
        if wants and me["grounded"] and speed < 1.5:
            if self.stuck_since is None:
                self.stuck_since = t
            elif t - self.stuck_since > 0.6 and t > self.unstick_until:
                self.unstick_until = t + 1.2
                self.side = -self.side
                self.stuck_since = None
        else:
            self.stuck_since = None

    # ================================================================== targets
    def pick_enemy(self, obs: mf.Observation) -> Optional[dict]:
        me = obs["self"]
        best, best_score = None, 1e9
        for m in obs["mechs"]:
            if m["kind"] != "mech" or m["id"] == me["id"]:
                continue
            d = math.dist(m["pos"], me["pos"])
            hp_frac = m.get("hp", 1) / max(1.0, m.get("max_hp", 1))
            score = d + 80 * hp_frac  # close and weak first
            if m["id"] == self.last_attacker:
                score -= 60
            if m.get("charging"):
                score -= 30  # a charging laser is about to hit someone: maybe us
            if score < best_score:
                best, best_score = m, score
        return best

    def pick_farm_target(self, obs: mf.Observation) -> Optional[dict]:
        me = obs["self"]
        p = me["pos"]
        best, bd = None, TUNING["crate_radius"]
        for c in obs["crates"]:
            d = hdist(p, c["pos"])
            if d < bd and abs(c["pos"][2] - p[2]) < 25:
                best, bd = dict(c, kind="crate"), d
        if me["hp"] / me["max_hp"] >= TUNING["npc_farm_min_hp_frac"]:
            for n in obs["mechs"]:
                if n["kind"] == "npc":
                    d = hdist(p, n["pos"])
                    if d < min(bd, 90):
                        best, bd = n, d
        return best

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
        # both full: swap out the worst non-gun weapon if this one is better
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
            if int(pk["id"]) in self.bad_pickups:
                continue
            if hdist(me["pos"], pk["pos"]) < reach and abs(pk["pos"][2] - me["pos"][2]) < 4:
                gain, drop = self.pickup_value(me, pk["kind"])
                if gain > 0:
                    out = {"take": pk["id"]}
                    if drop is not None:
                        out["drop"] = drop
                    return out
        return {}

    # ================================================================== goals
    def zone_status(self, obs: mf.Observation) -> Tuple[Optional[List[float]], bool]:
        """(point to head for or None, urgent?)"""
        me, z, t = obs["self"], obs["zone"], obs["t"]
        p = me["pos"]
        walk = me["stats"].get("walk_speed", 11.7)
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
        yaw = yaw_to(nc, p)  # point on our side of the next circle
        return [nc[0] + math.cos(yaw) * inner, nc[1] + math.sin(yaw) * inner], urgent

    def in_safe_zone(self, obs: mf.Observation, pt) -> bool:
        z = obs["zone"]
        return hdist(pt, z["next_center"]) < z["next_radius"] or obs["t"] < z["shrink_start"] - 20 and \
            hdist(pt, z["center"]) < z["radius"] - 10

    def choose_goal(self, obs, enemy, farm):
        me = obs["self"]
        p = me["pos"]
        zone_pt, urgent = self.zone_status(obs)
        if urgent and zone_pt:
            return zone_pt, True, "zone!"

        # fighting: movement is decided relative to the enemy in navigate()
        if enemy:
            return None, False, "fight"

        # better weapon nearby
        best, bd = None, TUNING["loot_radius"]
        for pk in obs["pickups"]:
            if int(pk["id"]) in self.bad_pickups or abs(pk["pos"][2] - p[2]) > 6:
                continue
            d = hdist(p, pk["pos"])
            if d < bd and self.pickup_value(me, pk["kind"])[0] > 0:
                best, bd = pk, d
        if best:
            return best["pos"], False, f"loot:{best['kind']}"

        # power cubes
        best, bd = None, TUNING["cube_radius"]
        for c in obs["cubes"]:
            d = hdist(p, c["pos"])
            if d < bd and abs(c["pos"][2] - p[2]) < 6:
                best, bd = c, d
        if best:
            return best["pos"], False, "cube"

        # supply drop: 3 cubes + full heal for standing alone in the ring
        for dr in obs.get("drops", []):
            d = hdist(p, dr["pos"])
            if d < TUNING["drop_max_dist"] and dr["lands_in"] < d / me["stats"].get("walk_speed", 11.7) + 8:
                if d < dr["radius"] * 0.6:
                    return dr["pos"], False, "capture"
                return dr["pos"], False, "drop"

        # crate / drone: stand off at ~20 m and shoot (combat handles firing)
        if farm:
            d = hdist(p, farm["pos"])
            if d > 25:
                return farm["pos"], False, "farm"
            return None, False, "hold"

        if zone_pt:
            return zone_pt, False, "zone"
        return None, False, "hold"

    # ================================================================== movement
    def navigate(self, obs, goal, urgent: bool, enemy):
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        energy = me["energy"]
        boost = jump = False
        mvec = (0.0, 0.0)

        if enemy is not None and not urgent:
            mvec, boost = self.fight_move(obs, enemy)
        elif goal is not None:
            d = hdist(p, goal)
            if d > 1.5:
                gy = yaw_to(p, goal)
                y = self.steer(p, gy, me)
                mvec = unit(y)
                boost = (urgent or not me["in_zone"]) and energy > TUNING["boost_min_energy"] * 0.5
            if urgent and enemy is not None:
                # still strafe a little while running for the zone
                s_yaw = yaw_to(p, enemy["pos"]) + math.pi / 2 * self.strafe
                mvec = (mvec[0] * 0.85 + math.cos(s_yaw) * 0.3, mvec[1] * 0.85 + math.sin(s_yaw) * 0.3)

        # unstick: jump and jet straight up while pushing toward the goal, then try the other side
        if t < self.unstick_until:
            jump = me["grounded"]
            boost = energy > 8
            if mvec == (0.0, 0.0) and goal is not None:
                mvec = unit(yaw_to(p, goal))
        n = math.hypot(*mvec)
        if n > 1:
            mvec = (mvec[0] / n, mvec[1] / n)
        return [round(mvec[0], 3), round(mvec[1], 3)], jump, boost

    def steer(self, p, goal_yaw: float, me) -> float:
        """Pick the walkable direction closest to goal_yaw, using the map probe."""
        if self.world is None:
            return goal_yaw
        look = 10.0
        h = me["stats"].get("height", 9.0)
        for off in (0, 25, -25, 50, -50, 75, -75, 105, -105, 140, -140):
            y = goal_yaw + math.radians(off) * self.side
            if self.world.probe(p[0], p[1], p[2], y, look, self.step_up, h, self.max_slope) >= look:
                return y
        return goal_yaw  # boxed in: push on and let unstick (jets) handle it

    def fight_move(self, obs, enemy):
        """Keep a weapon-appropriate range, strafe, and back off when losing."""
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        if t > self.next_flip:
            self.strafe = -self.strafe
            self.next_flip = t + self.rng.uniform(*TUNING["strafe_period"])
        d = hdist(p, enemy["pos"])
        to = yaw_to(p, enemy["pos"])
        held = [k for k in self.held(me) if k]
        want = min(TUNING["fight_range"].get(k, 70.0) for k in held) if held else 70.0
        my_frac = me["hp"] / me["max_hp"]
        en_frac = enemy.get("hp", 1) / max(1.0, enemy.get("max_hp", 1))
        retreat = my_frac < TUNING["retreat_hp_frac"] and en_frac > my_frac
        if retreat:
            radial = -1.0
        else:
            radial = max(-1.0, min(1.0, (d - want) / 25.0))
        strafe = 0.8 if not retreat else 0.4
        mx = math.cos(to) * radial + math.cos(to + math.pi / 2) * strafe * self.strafe
        my = math.sin(to) * radial + math.sin(to + math.pi / 2) * strafe * self.strafe
        y = math.atan2(my, mx)
        if self.world is not None:
            y = self.steer(p, y, me)
        boost = retreat and me["energy"] > TUNING["boost_min_energy"]
        return unit(y), boost

    # ================================================================== combat
    def combat(self, obs, target, is_mech: bool):
        me = obs["self"]
        p, t = me["pos"], obs["t"]
        st = me["stats"]
        muzzle = (p[0], p[1], p[2] + st.get("muzzle_height", 7.0))
        weapons = me["weapons"]
        fire = [False, False]
        reload = False
        cur_yaw, cur_pitch = me["aim"]

        if target is None:
            aim = self.idle_aim(obs)
            for i, w in enumerate(weapons):
                if w and w["kind"] == "laser" and w.get("charge", 0) > 0:
                    fire[i] = False  # release; nothing to hold for
                if w and w["kind"] == "gun" and w.get("ammo", 99) < w["stats"].get("mag", 12) * 0.5 \
                        and not w.get("reload"):
                    reload = True
            return [aim[0], aim[1]], fire, reload

        tp = target["pos"]
        tvel = target.get("vel", (0, 0, 0))
        if target.get("kind") == "mech":
            h = 9.0 * (1 + self.cfg["cubes"].get("size_per_cube", 0.05) * target.get("cubes", 0))
            r = 2.5 * h / 9.0
        elif target.get("kind") == "npc":
            h, r = self.cfg["npc"].get("height", 3.5), self.cfg["npc"].get("radius", 1.8)
        else:
            h, r = self.cfg["crates"].get("height", 2.4), self.cfg["crates"].get("radius", 1.4)
        center = (tp[0], tp[1], tp[2] + h * 0.5)
        dist = math.dist(muzzle, center)
        clear = self.world.los(muzzle, center) if self.world else True

        # default aim: gun lead (or straight at it)
        aim = list(aim_at(muzzle, center))
        gun_slot = next((i for i, w in enumerate(weapons) if w and w["kind"] == "gun"), None)
        if gun_slot is not None:
            g = weapons[gun_slot]
            pt, _ = lead_point(muzzle, center, tvel, g["stats"]["speed"])
            aim = list(aim_at(muzzle, pt))

        # mortar takes over the aim when it's ready and it's a mech (lob over cover too)
        for i, w in enumerate(weapons):
            if not w or w["kind"] != "mortar" or not is_mech:
                continue
            ws = w["stats"]
            hd = hdist(p, tp)
            if w["cooldown"] <= 0 and TUNING["mortar_min_range"] < hd < 0.95 * ws["speed"] ** 2 / ws["gravity"]:
                tube = p[2] + st.get("height", 9.0) + 0.5
                aim_pt, pitch = list(tp), None
                for _ in range(2):  # lead by flight time
                    pitch = ballistic_pitch(hdist(p, aim_pt), aim_pt[2] + 1.0 - tube, ws["speed"], ws["gravity"],
                                            high=not clear)
                    if pitch is None:
                        break
                    ft = flight_time(hdist(p, aim_pt), pitch, ws["speed"])
                    aim_pt = [tp[0] + tvel[0] * ft, tp[1] + tvel[1] * ft, tp[2]]
                if pitch is not None and pitch <= st.get("max_pitch", 1.3):
                    aim = [yaw_to(p, aim_pt), pitch]
                    fire[i] = self.aligned(me, aim, 0.03, 0.03)

        yaw_err = abs(wrap(aim[0] - cur_yaw))
        pitch_err = abs(aim[1] - cur_pitch)
        tol_yaw = math.atan2(r * 0.8, max(dist, 1.0)) + 0.01
        tol_pitch = math.atan2(h * 0.4, max(dist, 1.0)) + 0.01
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
                # Hold to charge, release (fire=False) when full and on target. Releasing fires.
                in_range = clear and dist < ws["range"]
                charge = w.get("charge", 0)
                if in_range and w["cooldown"] <= 0:
                    full = charge >= ws["charge"] - 0.05
                    fire[i] = not (full and on_target)
                else:
                    fire[i] = False
            elif k == "lightning":
                rng_ok = dist < ws["range"] * 0.95
                cone = math.radians(ws["cone_deg"]) * 0.8
                fire[i] = rng_ok and abs(wrap(yaw_to(p, tp) - cur_yaw)) < cone and clear and \
                    me["energy"] > TUNING["lightning_min_energy"] and w["cooldown"] <= 0
        return aim, fire, reload

    def aligned(self, me, aim, ty, tp) -> bool:
        return abs(wrap(aim[0] - me["aim"][0])) < ty and abs(aim[1] - me["aim"][1]) < tp

    def idle_aim(self, obs) -> Tuple[float, float]:
        """No target: face the last attacker (front armor + vision), else a fresh sound, else where we walk
        with a slow sweep to widen the 120 deg vision wedge."""
        me, t = obs["self"], obs["t"]
        if self.threat and t - self.threat[1] < 3.0:
            return self.threat[0], 0.0
        loud = [s for s in obs.get("sounds", []) if s["kind"] in ("gun", "laser", "laser_charge", "mortar",
                                                                   "lightning", "step", "boost", "jet")
                and s["dist"] != "far"]
        if loud:
            return loud[0]["bearing"], 0.0
        v = me["vel"]
        base = math.atan2(v[1], v[0]) if math.hypot(v[0], v[1]) > 1 else me["aim"][0]
        return wrap(base + 0.7 * math.sin(t * 0.9)), 0.0
