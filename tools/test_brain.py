"""Scenario tests for decisions the recorded stream rarely exercises. Run: python tools/test_brain.py

Each test takes a real observation from the kit stream, edits it into a situation, and checks the action.
Add one every time you fix a behavior bug so it stays fixed.
"""
import copy
import gzip
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KIT = os.path.join(ROOT, "mechfight-kit")

import mechfight as mf  # noqa: E402
from brain import Brain  # noqa: E402
from brain.geom import ballistic_pitch, intercept_time  # noqa: E402

with gzip.open(os.path.join(KIT, "check", "stream.jsonl.gz"), "rt") as f:
    MSGS = [json.loads(l) for l in f if l.strip()]
HELLO = MSGS[0]
HELLO["map"]["dir"] = os.path.join(KIT, "arena")
OBS = next(m for m in MSGS if m.get("type") == "obs")
CFG = HELLO["config"]


def fresh():
    mf.draw.clear()
    return Brain(copy.deepcopy(HELLO)), copy.deepcopy(OBS)


def weapon(kind, **kw):
    w = {"kind": kind, "cooldown": 0, "stats": dict(CFG["weapons"][kind])}
    w.update(kw)
    return w


def enemy_at(obs, dx, dy, **kw):
    p = obs["self"]["pos"]
    m = {"id": 9, "kind": "mech", "pos": [p[0] + dx, p[1] + dy, p[2]], "vel": [0, 0, 0], "aim": [0, 0],
         "hp": 1400, "max_hp": 1400, "cubes": 0, "weapons": ["gun"], "charging": False}
    m.update(kw)
    obs["mechs"] = [m]
    return m


def face(obs, act):
    obs["self"]["aim"] = list(act["aim"])  # pretend the torso finished slewing


def test_geometry():
    t = intercept_time((100, 0, 0), (0, 10, 0), 252)
    assert t and abs(math.hypot(100, 10 * t) - 252 * t) < 1e-6
    p = ballistic_pitch(100, 0, 72, 22)
    assert p and 0 < p < math.pi / 4


def test_gun_fires_when_aligned():
    b, o = fresh()
    enemy_at(o, 40, 0)
    a = b.act(o)
    face(o, a)
    a = b.act(o)
    assert a["fire"][0], a


def test_laser_charges_then_releases():
    b, o = fresh()
    enemy_at(o, 60, 0)
    o["self"]["weapons"][1] = weapon("laser", charge=0)
    a = b.act(o)
    face(o, a)
    a = b.act(o)
    assert a["fire"][1] is True, "should hold to charge"
    o["self"]["weapons"][1]["charge"] = CFG["weapons"]["laser"]["charge"]
    a = b.act(o)
    assert a["fire"][1] is False, "should release when full and on target"


def test_mortar_lobs():
    b, o = fresh()
    b.T["no_first_shot"] = False  # 120 m without a laser: we'd hold fire otherwise
    enemy_at(o, 120, 0)
    o["self"]["weapons"][1] = weapon("mortar")
    a = b.act(o)
    assert a["aim"][1] > 0.1, f"mortar should pitch up, got {a['aim']}"


def test_takes_better_weapon():
    b, o = fresh()
    p = o["self"]["pos"]
    o["pickups"] = [{"id": 8001, "kind": "laser", "pos": [p[0] + 1, p[1], p[2]]}]
    a = b.act(o)
    assert a.get("take") == 8001 and "drop" not in a, a
    o["self"]["weapons"][1] = weapon("lightning")
    a = b.act(o)
    assert a.get("take") == 8001 and a.get("drop") == 1, "should swap lightning for laser"


def test_upgrade_pick():
    b, o = fresh()
    o["offer"] = {"id": 1, "expires": 99, "choices": [{"id": "servos", "name": "", "desc": ""},
                                                      {"id": "jets", "name": "", "desc": ""},
                                                      {"id": "plating", "name": "", "desc": ""}]}
    assert b.act(o)["pick"] == 2


def test_runs_for_zone_when_outside():
    b, o = fresh()
    o["self"]["in_zone"] = False
    o["t"] = o["zone"]["shrink_end"] - 1
    a = b.act(o)
    assert b.mode == "zone!" and a["move"] != [0, 0], (b.mode, a)


def test_turns_to_unseen_attacker():
    """Shot in the back by someone we can't see: face them (front armor) instead of farming/other targets."""
    b, o = fresh()
    p = o["self"]["pos"]
    o["crates"] = [{"id": 1001, "pos": [p[0] + 30, p[1], p[2]], "hp": 300}]
    o["events"] = [{"type": "damaged", "amount": 67.5, "weapon": "gun", "bearing": 2.0, "by": 5, "rear": True}]
    a = b.act(o)
    assert abs(a["aim"][0] - 2.0) < 0.05 and a["fire"] == [False, False], a


def test_sticks_to_target():
    b, o = fresh()
    b.T["no_first_shot"] = False
    enemy_at(o, 60, 0)
    o["mechs"].append(dict(o["mechs"][0], id=10, pos=[o["self"]["pos"][0], o["self"]["pos"][1] + 62,
                                                      o["self"]["pos"][2]]))
    b.perceive(o)
    first = b.pick_enemy(o).id
    for _ in range(5):
        o["mechs"][0]["pos"][0] += 0.5  # tiny changes must not flip the target
        b.perceive(o)
        assert b.pick_enemy(o).id == first


def test_gives_up_unreachable_goal():
    b, o = fresh()
    goal = [o["self"]["pos"][0] + 50, o["self"]["pos"][1], 0]
    b.t = 0.0
    for k in range(80):
        b.t = k * 0.1
        b.watch_goal(o, goal, "cube")  # we never get closer
    assert not b.goal_ok(goal)


def test_climbs_out_of_courtyard():
    """Spawned in a walled-in courtyard (420, 30): wait for energy, then jump + jet toward open ground."""
    b, o = fresh()
    w = b.world
    o["self"]["pos"] = [420.0, 30.0, w.ground(420, 30)]
    o["self"]["vel"] = [0.0, 0.0, 0.0]
    o["mechs"] = []
    o["self"]["energy"] = 40
    o["self"]["in_zone"] = False
    a = b.act(o)
    assert "climb" in b.mode and a["move"] == [0, 0] and not a["jump"], (b.mode, a)  # charge energy first
    o["self"]["energy"] = 100
    a = b.act(o)
    assert "climb" in b.mode and a["jump"] and a["boost"] and a["move"] != [0, 0], (b.mode, a)


def test_no_first_shot_without_laser():
    b, o = fresh()
    b.T["no_first_shot"] = True  # on in free-for-all mode
    enemy_at(o, 100, 0)  # far, not attacking us, and we have no laser: don't start it
    b.perceive(o)
    assert b.pick_enemy(o) is None
    enemy_at(o, 40, 0, weapons=["gun", "laser"])  # a laser carrier: don't start it either
    b.perceive(o)
    assert b.pick_enemy(o) is None
    enemy_at(o, 40, 0)  # close gun-only mech: fair fight
    b.perceive(o)
    assert b.pick_enemy(o) is not None
    o["self"]["weapons"][1] = weapon("laser", charge=0)
    enemy_at(o, 100, 0, weapons=["gun", "laser"])  # with a laser we take it on
    b.perceive(o)
    assert b.pick_enemy(o) is not None


def test_lethal_laser_releases_early():
    b, o = fresh()
    enemy_at(o, 60, 0, hp=120)
    o["self"]["weapons"][1] = weapon("laser", charge=0)
    a = b.act(o)
    face(o, a)
    o["self"]["weapons"][1]["charge"] = 0.3  # partial charge already kills a 120 HP mech
    a = b.act(o)
    assert a["fire"][1] is False, "should release the partial charge"
    o["mechs"][0]["hp"] = 1400
    a = b.act(o)
    assert a["fire"][1] is True, "keep charging against a healthy target"


def test_never_picks_up_mortar():
    b, o = fresh()
    p = o["self"]["pos"]
    o["pickups"] = [{"id": 8002, "kind": "mortar", "pos": [p[0] + 1, p[1], p[2]]}]
    assert b.act(o).get("take") is None


def test_detects_free_for_all():
    b, o = fresh()
    o["mechs"] = []
    o["alive"] = 16
    b.act(o)
    assert not b.ffa
    o["alive"] = 14  # two mechs died and we killed nobody
    for _ in range(6):
        b.act(o)
    assert b.ffa and b.T["quiet_walk"]


def test_practice_kills_are_not_ffa():
    b, o = fresh()
    o["mechs"] = []
    o["alive"] = 16
    b.act(o)
    o["alive"] = 15
    o["events"] = [{"type": "kill", "target": 4}]
    b.act(o)
    o["events"] = []
    for _ in range(6):
        b.act(o)
    assert not b.ffa


def test_cube_cap():
    b, o = fresh()
    me = o["self"]
    for n, want in ((0, True), (9, True), (10, False), (12, False), (13, True), (14, True), (15, False)):
        me["cubes"] = n
        assert b.wants_cube(me) == want, n


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL  {name}: {e!r}")
    sys.exit(1 if failed else 0)
