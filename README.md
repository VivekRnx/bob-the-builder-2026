# bob-the-builder-2026: a Mech Royale bot

A Python bot for the **Mech Royale** track of the Builders Buildathon 2026: it pilots one mech in a
16-mech battle royale on the Ohio State campus. It speaks the `mechfight/1` protocol on stdin/stdout.
The full rules are in [mechfight-kit/protocol.md](mechfight-kit/protocol.md).

**Cutoff: Sun Nov 22, 2026, 11:00 AM ET.** Teams can submit 4 times an hour. Ties go to the earlier submission.

## Submitting

1. The repo is public with the `Dockerfile` at the top level, so it is ready as is.
2. Submit `https://github.com/VivekRnx/bob-the-builder-2026` on the Mech Royale track page.
3. Keep the repo public, with every submitted commit, until results are published. The final rebuilds your commit.

The **conformance** GitHub Action ([.github/workflows/check.yml](.github/workflows/check.yml)) runs on
every push. It builds the image on linux/amd64 and runs the kit's `check.py` under the scorer's
sandbox flags. A green check means the submission will pass the scorer's gate.

## Quick start

```sh
python tools/test_brain.py                                 # scenario tests
python tools/bench.py                                      # replay a recorded match in-process: latency + crashes
python mechfight-kit/check/check.py -- python -u bot.py    # official conformance check
python tools/sim.py -n 64 -j 16                            # simulator: our bot vs a 15-bot hunting squad
python tools/sim.py --mode ffa -n 16 -j 16                 # simulator: 8 of ours vs 8 baseline, free-for-all
python tools/sim.py --mode duel -n 40 -j 16                # simulator: 1v1 duels (combat skill only)
python tools/trace.py 7 --to 60                            # print what our bot saw and did in sim match seed 7
sh scripts/check.sh --docker                               # tests + bench + check, then the Docker sandbox check
```

## Watching simulated matches

```sh
python tools/replay.py 142 --open              # record sim match seed 142, open it in your browser
python tools/replay.py 7 --mode ffa --open     # free-for-all
python tools/replay.py 142 --hero tools.baseline.brain_v2 --open   # same seed, older bot, to compare
```

This writes a self-contained `replays/<mode>_<seed>.html` that you open by double-clicking. It shows:

- the campus top-down, with buildings shaded by height and the zone plus the dashed next circle
- every mech, colored by role (ours green, house tiers grey → red, red ring = hunting you), with aim
  line, HP bar, and a dot for its second weapon
- bullets, mortar shells, explosions, laser and lightning hits, and floating damage numbers
- crates, cubes, weapon pickups and supply drops
- **our bot's debug drawing**: its planned A* path, goal point and current mode, so you can see *why* it moved

Controls: Space play/pause · ←/→ ±2 s · `,`/`.` step · 1–6 speed · wheel zoom · drag pan ·
double-click reset. Click a mech in the side panel to follow it, or a kill in the feed to jump there;
kill ticks on the timeline are green for yours and white for your death. Use the same seed as a bad
result in `tools/sim.py --diag` to watch exactly what went wrong.

## Layout

| Path | What |
| --- | --- |
| [Dockerfile](Dockerfile) | Submission image. It copies only `bot.py`, `mechfight.py` and `brain/`. |
| [bot.py](bot.py) | Entrypoint: `mf.run(Brain)`. |
| [mechfight.py](mechfight.py) | The kit's protocol wrapper, unchanged: I/O loop, newest-obs-only, crash-safe coasting, debug drawing. |
| [brain/brain.py](brain/brain.py) | **All decisions.** The `TUNING` dict at the top holds the knobs. |
| [brain/nav.py](brain/nav.py) | 4 m navigation grid over the campus, built at startup (~0.3 s), plus time-budgeted, resumable A*. |
| [brain/world.py](brain/world.py) | Map model from `/arena`: terrain height, buildings, line of sight, walkability probe. |
| [brain/geom.py](brain/geom.py) | Angles, projectile lead (intercept), mortar ballistic pitch. |
| [tools/sim.py](tools/sim.py) | **Local match simulator** written from protocol.md: physics, walls, vision, all weapons, zone, loot, cubes, upgrades, drops, and a practice-like hunting squad. |
| [tools/trace.py](tools/trace.py) | Step-by-step text trace of our bot in one simulated match. |
| [tools/replay.py](tools/replay.py), [tools/viewer.html](tools/viewer.html) | Record a sim match and watch it in the browser. |
| [tools/baseline/brain_v1/](tools/baseline/brain_v1/) | The first version of the bot, kept as the opponent and A/B reference. |
| [tools/bench.py](tools/bench.py), [tools/test_brain.py](tools/test_brain.py) | Latency/crash replay and scenario tests. Add a test for every bug you fix. |
| [mechfight-kit/](mechfight-kit/) | The official kit, unmodified: protocol, schema, check, map, the TS wrapper. Not copied into the image. |

## How the bot plays

Each decision (10 per second) runs **perceive → pick target → decide → navigate → dodge → combat**.

- **Memory:** it tracks enemies for 20 s, beyond the server's 5 s. An attacker it can't see is placed
  along the damage bearing at a distance estimated from the weapon that hit. Weapon sounds are kept
  as contacts for 2 s.
- **Threat first:** when hit by someone it can't see (often in the back, ×1.5 damage), it drops
  everything and turns to face them. It keeps its current target unless another is clearly better,
  so it never flip-flops aim between two enemies.
- **Fight or hide:** it compares its time-to-die against everyone tracked near it with its time to kill
  the target. When losing, or hurt with threats around, it searches 48 nearby spots for one that **no**
  nearby threat can see. It hides there, holding fire so regen (5 s without dealing or taking damage)
  can start, and re-engages when healed.
- **Fighting:** it holds a range suited to its weapons (and stays out of lightning range when the enemy
  has lightning), jukes sideways at random intervals, and always faces the enemy (front armor).
- **Dodging:** it predicts where enemy mortar shells land and leaves the splash, and sidesteps when a
  visible enemy is charging a laser aimed at it.
- **Weapons:**
  - Gun: leads moving targets.
  - Mortar: solves the ballistic arc, lobs high over cover, and won't fire inside its own splash range.
  - Laser: charges, then releases when full and on target.
  - Lightning: fires only when the target is inside its range and cone.
- **Navigation:** it walks straight when the map probe says the way is clear, and otherwise follows an
  A* path on the nav grid. Optional goals that stop getting closer for 6 s, or that A* says are
  unreachable, are dropped for 30 s. When stuck, it jumps and jets.
- **Economy:** when safe, it picks up better weapons (laser > mortar > lightning), collects cubes, and
  shoots crates and drones from where it stands when it has line of sight. It grabs a kill's dropped
  cubes when no enemy is near them, and takes uncontested supply drops.
- **Zone:** it moves inside the *next* circle early enough to arrive before it closes, and boosts when late.
- **Upgrades:** it takes max HP first (`plating`), because the final zone ignores armor and the biggest
  HP pool lasts longest. Then `armor`, then `nanites` (regen).

## Results in the local simulator

The simulator is an approximation of the real server, not the real thing; trust practice scores over
it. Mean placement points (`16 − place`):

| Setup | Baseline v1 | Current |
| --- | --- | --- |
| Practice-like: alone vs a 15-bot squad, 64 matches | 0.55 | **1.06** |
| Free-for-all: 8 vs 8 baseline bots, 22 matches (7.5 = even) | 6.33 | **8.67** |
| 1v1 duels vs baseline, 40 matches (win rate) | 60% | **78%** |

## How you're actually scored

- Score = mean of `16 − place`. **Placement only.** Kills matter only because each dead opponent moves you up one place.
- **Practice:** your bot plays against 15 house bots that work as **one squad**. They share vision,
  never fight each other, and hunt *you*: one at the start, more as you thin them out, all of them
  late. Your place ≈ 16 − (house bots dead before you), so **win the early 1v1s** and heal between them.
- **Final:** a free-for-all among the top teams' bots. Staying alive and third-partying matter more
  there, so don't overfit to the squad.

## Next steps (highest expected value first)

1. **Submit now** to get a real practice score and an early tiebreak timestamp.
2. Read the practice results, then tune `TUNING`, changing one thing per submission:
   `disengage_ratio`, `cover_hp_frac`, `fight_range`, `weapon_rank`, `upgrade_priority`.
3. Endgame: pick a covered spot inside the final circles early, and stop fighting when ahead.
4. Rooftops: the nav grid is ground-only. Roof camping (jets climb ~60 m) is unexplored.
5. Make the simulator's squad smarter, or add your own opponent bots to `tools/`, so A/B tests stay meaningful.

Log with `mf.log(...)` (stderr). **Never `print()` to stdout**: it breaks the protocol.

## License

MIT (see [LICENSE](LICENSE)). The kit files in `mechfight-kit/` come from the organizers.


## Independent specialist learning (experimental)

The eight functional parts of the bot can now be trained separately. These are learned
selectors over behavior alternatives, rather than eight separate processes or language models.
Only one module changes in each experiment; the rest of the current bot and the opponents stay fixed.

| Module | Decisions learned | Additional task feedback |
| --- | --- | --- |
| perception | memory duration, bounded dead reckoning, trusting estimates | tracking quality against simulator labels |
| targeting | nearest, weakest, attacker, clear shot, current targeting | damage dealt and eliminations |
| strategy | cover, retreat, holding range, recovery, current strategy | health, survival, zone damage |
| navigation | boost use, early unsticking, steering clearance | progress, stuck time, reaching the destination |
| dodging | boost timing and cost, left/right evasions, current dodge | health and survival against lasers/mortars |
| combat | firing tolerance and target lead | damage dealt, health and survival |
| economy | health, weapons, cubes, supplies, current priorities | cubes, upgrades and equipment |
| zone | when and how deeply to rotate, boost conservation | zone exposure, progress and survival |

From the repository root:

~~~sh
# Train every module for 1,008 episodes each. Repeat to resume a partial run.
python -B tools/train_specialists.py --run runs/specialists-v1 --episodes 1008 --workers 4

# Inspect saved progress without starting training.
python -B tools/train_specialists.py --run runs/specialists-v1 --status

# Continue just combat to a TOTAL of 2,016 training episodes.
python -B tools/train_specialists.py --run runs/specialists-v1 --modules combat --episodes 2016 --workers 4

# Evaluate the selected models (or unpromoted candidates) on separate test seeds.
python -B tools/train_specialists.py --run runs/specialists-v1 --test --eval-seeds 48 --workers 4

# Run ONE experimental specialist on the game protocol; this does not change bot.py.
python -B tools/specialist_bot.py --module combat --policy runs/specialists-v1/combat/candidate.json

python -B tools/test_specialists.py
~~~

Each module has a directory under the run containing `state.json` (resume state),
`candidate.json`, `history.jsonl`, and validation/test reports. A `champion.json` is written
only when validation shows a positive approximate paired 95% lower bound, acceptable
decision latency, and no more than one episode's survival regression versus baseline.
Promotion means selection inside the experiment; it never activates the policy in `bot.py`.

Training uses a small linear softmax actor and learned value baseline (REINFORCE), running on
CPU with standard-library Python. A GPU is unnecessary for these 120-weight policies.
Seed sets are separate for each module and for training, validation, and final testing.
Changing source code requires a new run directory; increasing `--episodes` or changing
`--workers` does not. Keep batch and validation settings unchanged when resuming training.
Test sample count may differ from the validation sample count.

Rewards are specialized proxies, not the official placement score. The simulator approximates
the real game. Positive validation results need separate test confirmation, then full-match
and real practice evaluation. Independently useful policies may also conflict when combined;
this tool intentionally runs them one at a time. Evaluate a test set once after model selection;
further tuning should use training/validation and reserve new test scenarios for a later audit.
