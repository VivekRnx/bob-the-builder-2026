"""A practice-board house bot for tools/sim.py, modelled on what other teams measured from real replays
(mh-anwar docs/knowledge/housebots.md, JuanQuenga docs/REVIEW.md, the track page text):

- 15 house bots: 4 easy, 4 medium, 4 hard, 3 expert. One squad (the simulator handles shared vision and
  the hunter schedule: one hunter early, more as they die, all of them late).
- Never retreat, even below 30% HP. Stay in the zone around 0.41 of its radius from the center.
- Gun: tap-fire every 0.30 s (not the 0.24 s cooldown). Expert aim error ~1.7 deg, exact lead,
  ~0.75 s reaction to a new target. Lower tiers aim worse and react slower.
- 86% grab a second weapon early (laser or lightning, rarely mortar).

    python tools/sim.py --opp tools.housebot          # our bot vs house-like squad
"""
from __future__ import annotations

import math
import os
import random

# The frozen v3 bot is the chassis (navigation, aim, laser/lightning use). Import it without the
# hero's MECH_TUNING overrides so A/B experiments on our bot never change the opponents.
_saved = os.environ.pop("MECH_TUNING", None)
try:
    from tools.baseline.brain_v3.brain import Brain as _Base
    from tools.baseline.brain_v3.geom import wrap
finally:
    if _saved is not None:
        os.environ["MECH_TUNING"] = _saved

# (aim error deg, reaction s, chance to want a 2nd weapon)
TIERS = {"easy": (5.0, 1.6, 0.6), "medium": (3.4, 1.2, 0.8), "hard": (2.4, 0.95, 0.9), "expert": (1.7, 0.75, 1.0)}


def tier_of(slot: int) -> str:
    # 15 house bots fill the slots that aren't the hero's; spread tiers over slot numbers
    k = slot % 15
    return "easy" if k < 4 else "medium" if k < 8 else "hard" if k < 12 else "expert"


class Brain(_Base):
    name = "house"
    params = {
        "disengage_ratio": 0.0, "cover_hp_frac": 0.0, "reengage_hp_frac": 0.0, "heal_below_frac": 0.0,
        "chase": True, "two_front": False, "finish_hp_frac": 1.0,
        "zone_inner_frac": 0.41, "zone_slack_s": 10.0,
        "weapon_rank": {"laser": 3, "lightning": 2, "mortar": -1, "gun": 0},
        "loot_radius": 120.0, "cube_radius": 60.0, "crate_radius": 0.0,
        "fight_range": {"gun": 60.0, "laser": 110.0, "mortar": 90.0, "lightning": 26.0},
        "strafe_period": (0.8, 2.0), "fight_boost_energy": 60.0,
        "debug": False,
    }

    def __init__(self, hello):
        super().__init__(hello)
        self.tier = tier_of(int(hello["slot"]))
        self.err_deg, self.react_s, p2 = TIERS[self.tier]
        self.hrng = random.Random(int(hello["seed"]) * 7 + 13)
        if self.hrng.random() > p2:
            self.T["weapon_rank"] = {"laser": -1, "lightning": -1, "mortar": -1, "gun": 0}
        self.aim_off = (0.0, 0.0)
        self.off_t = -99.0
        self.last_gun = -99.0
        self.first_seen = {}

    def combat(self, obs, enemy, farm):
        aim, fire, reload = super().combat(obs, enemy, farm)
        t = obs["t"]
        if enemy is None:
            return aim, fire, reload
        # reaction time to a new target
        fs = self.first_seen.setdefault(enemy.id, t)
        if t - self.tracks[enemy.id].seen_t > 3.0:
            self.first_seen[enemy.id] = fs = t
        if t - fs < self.react_s:
            return aim, [False, False], reload
        # persistent aim error, re-drawn every 0.3 s (one tap)
        if t - self.off_t >= 0.3:
            e = math.radians(self.err_deg)
            self.aim_off = (self.hrng.gauss(0, e), self.hrng.gauss(0, e * 0.5))
            self.off_t = t
        aim = [wrap(aim[0] + self.aim_off[0]), aim[1] + self.aim_off[1]]
        weapons = obs["self"]["weapons"]
        for i, w in enumerate(weapons):
            if w and w["kind"] == "gun" and fire[i]:
                if t - self.last_gun < 0.30:
                    fire[i] = False  # tap fire
                else:
                    self.last_gun = t
        return aim, fire, reload
