"""Independent learned decision policies around the existing bot's mechanics.

Each policy controls ONE module. Other hooks delegate unchanged to CurrentBrain.
No network or third-party dependencies are needed for training or inference.
"""
from __future__ import annotations
import copy
import json
import math
from .brain import Brain as CurrentBrain
from .geom import hdist, unit, wrap, yaw_to

VERSION = 1
MODULES = {
    "perception": ["baseline", "short_memory", "long_memory", "dead_reckon", "cautious_estimates"],
    "targeting": ["baseline", "nearest", "weakest", "attacker", "clear_shot"],
    "strategy": ["baseline", "seek_cover", "retreat", "hold_range", "recover"],
    "navigation": ["baseline", "conserve_boost", "travel_boost", "early_unstick", "wide_steering"],
    "dodging": ["baseline", "early_boost", "left", "right", "conserve_energy"],
    "combat": ["baseline", "strict_aim", "wide_aim", "short_lead", "long_lead"],
    "economy": ["baseline", "health_first", "weapon_first", "cubes_first", "supplies_first"],
    "zone": ["baseline", "early_center", "early_edge", "conserve_energy", "late_center"],
}
FEATURES = ["bias", "hp", "energy", "enemy_distance", "enemy_hp", "enemy_count",
            "recent_damage", "laser_charging", "mortar_count", "in_zone",
            "next_zone_distance", "shrink_time", "gun_ammo", "reloading",
            "second_weapon", "cubes", "nearest_resource", "resource_count",
            "oldest_memory", "speed", "zone_margin", "enemy_closing",
            "enemy_bearing", "upgrade_offer"]


def initial_weights():
    w = [[0.0]*len(FEATURES) for _ in range(5)]
    w[0][0] = 0.05
    return w


def validate_weights(w):
    if not isinstance(w, list) or len(w) != 5 or any(not isinstance(row, list) or len(row) != len(FEATURES) for row in w):
        raise ValueError("Wrong specialist policy dimensions")
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or abs(v) > 100 for row in w for v in row):
        raise ValueError("Invalid specialist policy weight")
    return w


def policy_document(module, weights):
    if module not in MODULES:
        raise ValueError("Unknown specialist")
    return dict(version=VERSION, module=module, actions=MODULES[module], features=FEATURES,
                weights=validate_weights(weights))


def load_policy(path, module):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("version") != VERSION or d.get("module") != module or d.get("actions") != MODULES[module] or d.get("features") != FEATURES:
        raise ValueError("Policy module or schema does not match")
    return validate_weights(d["weights"])


def probabilities(w, x, allowed):
    if not allowed:
        raise ValueError("No allowed actions")
    logits = [sum(a*b for a, b in zip(row, x)) for row in w]
    maximum = max(logits[a] for a in allowed)
    p = [math.exp(max(-60.0, v-maximum)) if i in allowed else 0.0 for i, v in enumerate(logits)]
    total = sum(p)
    return [v/total for v in p]


def features(obs, tracks):
    me, p, z = obs["self"], obs["self"]["pos"], obs["zone"]
    enemies = [m for m in obs["mechs"] if m["kind"] == "mech" and m["id"] != me["id"]]
    enemy = min(enemies, key=lambda e: hdist(p, e["pos"])) if enemies else None
    d = hdist(p, enemy["pos"]) if enemy else 300.0
    resources = obs["cubes"]+obs["pickups"]+obs["crates"]+obs.get("drops", [])
    resource_d = min((hdist(p, r["pos"]) for r in resources), default=300)
    gun = next((w for w in me["weapons"] if w and w["kind"] == "gun"), {})
    closing = 0.0
    if enemy and d > 1:
        closing = sum((enemy["vel"][k]-me["vel"][k])*(p[k]-enemy["pos"][k])/d for k in (0, 1))/30
    damaged = any(e.get("type") == "damaged" for e in obs.get("events", []))
    x = [1.0, me["hp"]/max(1, me["max_hp"]), me["energy"]/max(1, me["max_energy"]),
         min(1, d/250), enemy["hp"]/max(1, enemy["max_hp"]) if enemy else 1.0,
         min(1, len(enemies)/4), float(damaged),
         float(any(m.get("charging") for m in enemies)),
         min(1, sum(p["kind"] == "mortar" for p in obs.get("projectiles", []))/3),
         float(me["in_zone"]), min(1, hdist(p, z["next_center"])/300),
         max(0, min(1, (z["shrink_end"]-obs["t"])/40)),
         gun.get("ammo", 0)/max(1, gun.get("stats", {}).get("mag", 12)),
         float(gun.get("reload", 0) > 0), float(me["weapons"][1] is not None),
         min(1, me["cubes"]/10), min(1, resource_d/250), min(1, len(resources)/10),
         min(1, max((obs["t"]-tr.seen_t for tr in tracks.values()), default=0)/30),
         min(1, math.hypot(*me["vel"][:2])/25),
         max(-1, min(1, (z["radius"]-hdist(p, z["center"]))/max(1, z["radius"]))),
         max(-1, min(1, closing)), wrap(yaw_to(p, enemy["pos"])-me["aim"][0])/math.pi if enemy else 0.0,
         float(bool(obs.get("offer")))]
    return [float(v) for v in x]


class Brain(CurrentBrain):
    name = "specialist-experiment"
    module = "strategy"
    weights = None
    training = False
    params = {"debug": False}

    def __init__(self, hello):
        if self.module not in MODULES:
            raise ValueError("Unknown module")
        super().__init__(hello)
        self.T = copy.deepcopy(self.T)
        self.original_tuning = copy.deepcopy(self.T)
        self.policy_weights = copy.deepcopy(self.weights)
        if self.policy_weights is not None:
            validate_weights(self.policy_weights)
        self.choice = 0
        self.next_choice = -1.0
        self.active_obs = None
        self.trajectory = []
        self.action_counts = {a: 0 for a in MODULES[self.module]}
        self.errors = 0
        self.last_perception_t = 0.0

    def allowed(self, obs):
        # No need to explore target choices without multiple enemies, or upgrades with no resources.
        enemies = [m for m in obs["mechs"] if m["kind"] == "mech"]
        if self.module == "targeting" and len(enemies) < 2:
            return [0]
        if self.module == "dodging" and not obs.get("projectiles") and not any(m.get("charging") for m in enemies):
            return [0]
        if self.module == "combat" and not enemies and not obs["crates"]:
            return [0]
        return list(range(5))

    def configure(self):
        self.T = copy.deepcopy(self.original_tuning)
        a, mod = self.choice, self.module
        if mod == "perception":
            self.T["track_keep_s"] = {1: 6.0, 2: 35.0}.get(a, self.T["track_keep_s"])
        elif mod == "strategy":
            if a == 3:
                self.T["chase"] = False
                self.T["fight_range"] = {k: v*1.25 for k, v in self.T["fight_range"].items()}
            elif a == 4:
                self.T["heal_below_frac"] = 0.85
                self.T["cover_hp_frac"] = 0.7
                self.T["reengage_hp_frac"] = 0.9
        elif mod == "economy":
            if a == 1:
                self.T["upgrade_priority"] = ["plating", "nanites", "armor"]+self.T["upgrade_priority"]
            elif a == 2:
                self.T["upgrade_priority"] = ["gun_damage", "gun_rate", "laser_damage"]+self.T["upgrade_priority"]
        elif mod == "zone":
            settings = {1: (0.35, 25), 2: (0.85, 25), 3: (0.65, 15), 4: (0.35, 2)}
            if a in settings:
                self.T["zone_inner_frac"], self.T["zone_slack_s"] = settings[a]

    def act(self, obs):
        self.active_obs = obs
        try:
            if self.policy_weights is not None and obs["t"] >= self.next_choice:
                x, allowed = features(obs, self.tracks), self.allowed(obs)
                p = probabilities(self.policy_weights, x, allowed)
                if self.training:
                    threshold, cumulative = self.rng.random(), 0.0
                    self.choice = allowed[-1]
                    for i, prob in enumerate(p):
                        cumulative += prob
                        if threshold < cumulative:
                            self.choice = i
                            break
                    self.trajectory.append(dict(x=x, p=p, action=self.choice, reward=0.0))
                else:
                    self.choice = max(allowed, key=lambda i: p[i])
                self.next_choice = obs["t"]+0.4
                self.configure()
            self.action_counts[MODULES[self.module][self.choice]] += 1
            return super().act(obs)
        except Exception:
            self.errors += 1
            raise

    def perceive(self, obs):
        super().perceive(obs)
        if self.module == "perception" and self.choice:
            reported = {m["id"] for m in obs["mechs"]}
            dt = max(0, min(0.2, obs["t"]-self.last_perception_t))
            if self.choice == 3:
                for tr in self.tracks.values():
                    if tr.id not in reported and 0 < obs["t"]-tr.seen_t < 3 and not tr.est:
                        q = [tr.pos[k]+tr.vel[k]*dt for k in range(3)]
                        if self.world is None or self.world.in_bounds(q[0], q[1], 4):
                            tr.pos = q
            elif self.choice == 4:
                self.tracks = {k: tr for k, tr in self.tracks.items() if not tr.est or obs["t"]-tr.seen_t < 2}
        self.last_perception_t = obs["t"]

    def pick_enemy(self, obs):
        original = super().pick_enemy(obs)
        if self.module != "targeting" or self.choice == 0:
            return original
        enemies = [self.tracks[m["id"]] for m in self.visible_enemies(obs) if m["id"] in self.tracks]
        if not enemies:
            return original
        p, me = obs["self"]["pos"], obs["self"]
        if self.choice == 1:
            result = min(enemies, key=lambda tr: hdist(p, tr.pos))
        elif self.choice == 2:
            result = min(enemies, key=lambda tr: tr.hp/max(1, tr.max_hp)+hdist(p, tr.pos)/1000)
        elif self.choice == 3:
            attacker = self.threat[2] if self.threat and obs["t"]-self.threat[1] < 4 else None
            result = next((tr for tr in enemies if tr.id == attacker), original)
        else:
            result = min(enemies, key=lambda tr: (not self.has_los_to(me, tr), hdist(p, tr.pos)))
        if result is not None:
            self.target_id = result.id
        return result

    def decide(self, obs, enemy, farm, safe):
        result = super().decide(obs, enemy, farm, safe)
        if result[1]:  # preserve urgent zone actions
            return result
        if self.module == "strategy":
            if self.choice == 1 and self.threats(obs):
                cover = self.find_cover(obs, self.threats(obs))
                if cover is not None:
                    return cover, False, "cover", False
            if self.choice == 2 and enemy is not None:
                return None, False, "retreat", True
        if self.module == "economy" and safe and self.choice in (2, 3, 4):
            me = obs["self"]
            items = obs["pickups"] if self.choice == 2 else obs["cubes"] if self.choice == 3 else [
                d for d in obs.get("drops", []) if d.get("holder") in (None, me["id"])]
            valid = [r for r in items if hdist(me["pos"], r["pos"]) < 130
                     and abs(me["pos"][2]-r["pos"][2]) < 6 and self.goal_ok(r["pos"])
                     and (self.choice != 2 or self.pickup_value(me, r["kind"])[0] > 0)]
            if valid:
                selected = min(valid, key=lambda r: hdist(me["pos"], r["pos"]))
                return selected["pos"], False, "cube" if self.choice == 3 else "drop" if self.choice == 4 else "loot", False
        return result

    def navigate(self, obs, goal, urgent, enemy, fight_move):
        result = super().navigate(obs, goal, urgent, enemy, fight_move)
        if self.module == "navigation":
            move, jump, boost = result
            if self.choice == 1 and not urgent:
                boost = False
            elif self.choice == 2 and goal is not None and obs["self"]["grounded"] and obs["self"]["energy"] > 30:
                boost = True
            elif self.choice == 3 and self.stuck_since is not None and obs["t"]-self.stuck_since > 0.2:
                jump, boost = obs["self"]["grounded"], obs["self"]["energy"] > 15
            return move, jump, boost
        if self.module == "zone" and self.choice == 3 and not urgent:
            return result[0], result[1], False
        return result

    def steer(self, p, goal_yaw, me, look=9.0):
        if self.module == "navigation" and self.choice == 4:
            look = max(look, 14.0)
        return super().steer(p, goal_yaw, me, look)

    def dodge(self, obs):
        result = super().dodge(obs)
        if self.module != "dodging" or result is None:
            return result
        move, boost = result
        me = obs["self"]
        if self.choice == 1:
            boost = me["energy"] > 8 and me["grounded"]
        elif self.choice in (2, 3):
            enemies = self.visible_enemies(obs)
            # Side relative to the incoming attack; preserve baseline direction for shells with no contact.
            if enemies:
                nearest = min(enemies, key=lambda e: hdist(me["pos"], e["pos"]))
                angle = yaw_to(me["pos"], nearest["pos"])+(math.pi/2 if self.choice == 2 else -math.pi/2)
                move = list(unit(self.steer(me["pos"], angle, me, look=8)))
        elif self.choice == 4:
            boost = boost and me["energy"] > 50
        return move, boost

    def combat(self, obs, enemy, farm):
        if self.module != "combat" or self.choice == 0:
            return super().combat(obs, enemy, farm)
        old = self.T["aim_tol_mult"]
        try:
            if self.choice in (1, 2):
                self.T["aim_tol_mult"] = old*(0.65 if self.choice == 1 else 1.35)
            elif enemy is not None:
                enemy = copy.copy(enemy)
                enemy.vel = [v*(0.65 if self.choice == 3 else 1.35) for v in enemy.vel]
            return super().combat(obs, enemy, farm)
        finally:
            self.T["aim_tol_mult"] = old


def specialist_class(module, weights=None, training=False):
    if module not in MODULES:
        raise ValueError("Unknown module")
    return type("Specialist_"+module, (Brain,), dict(module=module, weights=weights,
                training=training, name=("learned-"+module)[:32]))
