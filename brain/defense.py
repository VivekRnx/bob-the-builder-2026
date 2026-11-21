"""Experimental learned defensive controller; production bot.py remains unchanged.

Load with DEFENSE_POLICY=/path/to/policy.json and import brain.defense:Brain.
Training uses a small softmax policy with REINFORCE and a learned value baseline.
Only observations available to the submitted bot are used as policy inputs.
"""
from __future__ import annotations
import json
import math
import os
from .brain import Brain as BaseBrain
from .geom import hdist, unit, yaw_to

ACTIONS = ["baseline", "hide", "escape", "dodge_left", "dodge_right"]
FEATURES = ["bias", "hp", "energy", "enemy_distance", "threat_count",
            "recent_damage", "laser_aimed", "mortar_danger", "mortar_imminence",
            "cover_available", "cover_distance", "left_clear", "right_clear",
            "in_zone", "zone_margin", "closing_speed", "enemy_hp",
            "regen_ready", "mortar_side", "enemy_los"]
VERSION = 1


def initial_weights():
    w = [[0.0] * len(FEATURES) for _ in ACTIONS]
    w[0][0] = 0.1  # explore, while preferring the existing behavior initially
    return w


def validate_weights(w):
    if not isinstance(w, list) or len(w) != len(ACTIONS):
        raise ValueError("Wrong number of policy actions")
    if any(not isinstance(row, list) or len(row) != len(FEATURES) for row in w):
        raise ValueError("Wrong number of policy features")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for row in w for v in row):
        raise ValueError("Policy weights must be finite numbers")
    return w


def load_policy(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d.get("version") != VERSION or d.get("features") != FEATURES or d.get("actions") != ACTIONS:
        raise ValueError("Incompatible defense policy")
    return validate_weights(d["weights"])


def probabilities(weights, x, allowed):
    logits = [sum(a*b for a, b in zip(row, x)) for row in weights]
    top = max(logits[i] for i in allowed)
    p = [math.exp(max(-60.0, logits[i]-top)) if i in allowed else 0.0 for i in range(len(ACTIONS))]
    total = sum(p)
    return [v/total for v in p]


class Brain(BaseBrain):
    name = "defense-experimental"
    weights = None
    training = False

    def __init__(self, hello):
        super().__init__(hello)
        self.policy_weights = self.weights
        if self.policy_weights is None and os.environ.get("DEFENSE_POLICY"):
            self.policy_weights = load_policy(os.environ["DEFENSE_POLICY"])
        if self.policy_weights is not None:
            validate_weights(self.policy_weights)
        self.trajectory = []
        self.action_counts = {a: 0 for a in ACTIONS}
        self.next_choice = -1.0
        self.choice = 0
        self.last_state = None
        self.errors = 0
        self.last_cover = None

    def dodge(self, obs):
        # The policy chooses a maneuver after the underlying brain has updated its state.
        return None

    def reward_until(self, t, hp, max_hp, in_zone):
        if self.last_state is not None and self.trajectory:
            old_t, old_hp = self.last_state
            dt = max(0.0, t-old_t)
            # Net health change: getting hurt then healing does not create free net HP reward.
            reward = 3.0*(hp-old_hp)/max(1.0, max_hp) + 0.04*dt
            if not in_zone:
                reward -= 0.15*dt
            self.trajectory[-1]["reward"] += reward
        self.last_state = (t, hp)

    def features(self, obs, threats):
        me, p = obs["self"], obs["self"]["pos"]
        enemy = min(threats, key=lambda tr: hdist(p, tr.pos)) if threats else None
        heading = yaw_to(p, enemy.pos) if enemy else me["aim"][0]
        dist = hdist(p, enemy.pos) if enemy else 300.0
        cover = self.find_cover(obs, threats) if threats else None
        self.last_cover = cover
        left, right = 1.0, 1.0
        if self.world is not None:
            vals = []
            for sign in (1, -1):
                vals.append(min(1.0, self.world.probe(
                    p[0], p[1], p[2], heading+sign*math.pi/2, 8.0,
                    self.step_up, me["stats"].get("height", 9.0),
                    self.max_slope, radius=self.rad)/8.0))
            left, right = vals
        danger, imminent, mortar_side = 0.0, 0.0, 0.0
        g = self.cfg["weapons"]["mortar"]["gravity"]
        radius = self.cfg["weapons"]["mortar"]["splash_radius"] + self.rad + 2.5
        for pr in obs.get("projectiles", []):
            if pr["kind"] != "mortar":
                continue
            x, y, z = pr["pos"]
            vx, vy, vz = pr["vel"]
            disc = vz*vz + 2*g*(z-p[2])
            if disc < 0 or g <= 0:
                continue
            ttl = (vz+math.sqrt(disc))/g
            if not 0 <= ttl <= 4:
                continue
            lx, ly = x+vx*ttl, y+vy*ttl
            d = math.hypot(lx-p[0], ly-p[1])
            risk = max(0.0, 1-d/(2*radius))
            if risk > danger:
                danger, imminent = risk, 1-ttl/4
                mortar_side = max(-1.0, min(1.0,
                    ((lx-p[0])*(-math.sin(heading))+(ly-p[1])*math.cos(heading))/radius))
        z = obs["zone"]
        margin = (z["radius"]-hdist(p, z["center"]))/max(1.0, z["radius"])
        closing = 0.0
        if enemy and dist > 1:
            closing = sum((enemy.vel[k]-me["vel"][k])*(p[k]-enemy.pos[k])/dist for k in (0, 1))/30
        x = [1.0, me["hp"]/max(1, me["max_hp"]), me["energy"]/max(1, me["max_energy"]),
             min(1.0, dist/250), min(1.0, len(threats)/3),
             max(0.0, 1-(obs["t"]-self.last_hit_taken)/3),
             float(any(tr.charging and self.aimed_at_me(tr, me) for tr in threats)),
             danger, imminent, float(cover is not None),
             min(1.0, hdist(p, cover)/40) if cover else 1.0, left, right,
             float(me["in_zone"]), max(-1.0, min(1.0, margin)),
             max(-1.0, min(1.0, closing)), enemy.hp/max(1, enemy.max_hp) if enemy else 1.0,
             float(obs["t"]-max(self.last_hit_dealt, self.last_hit_taken) >= me["stats"].get("regen_delay", 5)),
             mortar_side, float(enemy is not None and self.has_los_to(me, enemy))]
        allowed = [0, 2, 3, 4] if threats or danger > 0 else [0]
        if cover is not None and threats:
            allowed.append(1)
        return x, allowed, heading

    def act(self, obs):
        try:
            return self._act(obs)
        except Exception:
            self.errors += 1
            raise

    def _act(self, obs):
        act = super().act(obs)
        me, t = obs["self"], obs["t"]
        self.reward_until(t, me["hp"], me["max_hp"], me["in_zone"])
        # Existing zone emergencies override learned maneuvers.
        urgent = self.mode.startswith("zone!")
        if self.policy_weights is None or urgent:
            dodge = BaseBrain.dodge(self, obs)
            if dodge is not None:
                act["move"], act["boost"] = dodge
            return act
        threats = self.threats(obs)
        if t >= self.next_choice:
            x, allowed, heading = self.features(obs, threats)
            probs = probabilities(self.policy_weights, x, allowed)
            if self.training:
                r, acc = self.rng.random(), 0.0
                self.choice = allowed[-1]
                for i, prob in enumerate(probs):
                    acc += prob
                    if r < acc:
                        self.choice = i
                        break
                self.trajectory.append(dict(x=x, p=probs, action=self.choice, reward=0.0))
            else:
                self.choice = max(allowed, key=lambda i: probs[i])
            self.next_choice = t+0.4
        self.action_counts[ACTIONS[self.choice]] += 1
        if self.choice == 0:
            dodge = BaseBrain.dodge(self, obs)
            if dodge is not None:
                act["move"], act["boost"] = dodge
            return act
        if not threats:
            return act
        enemy = min(threats, key=lambda tr: hdist(me["pos"], tr.pos))
        heading = yaw_to(me["pos"], enemy.pos)
        if self.choice == 1 and self.last_cover is not None:
            move, jump, boost = self.navigate(obs, self.last_cover, False, enemy, False)
            act.update(move=move, jump=jump, boost=boost, fire=[False, False])
        else:
            angle = heading + (math.pi if self.choice in (1, 2) else math.pi/2 if self.choice == 3 else -math.pi/2)
            angle = self.steer(me["pos"], angle, me, look=8)
            act["move"] = list(unit(angle))
            act["boost"] = bool(me["grounded"] and me["energy"] > 15)
        self.mode = "learned:"+ACTIONS[self.choice]
        return act

    def finish(self, t, hp, max_hp, alive, in_zone):
        self.reward_until(t, max(0.0, hp), max_hp, in_zone)
        if self.trajectory:
            self.trajectory[-1]["reward"] += (2.0 + max(0.0, hp)/max_hp) if alive else -3.0
