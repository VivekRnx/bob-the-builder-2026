"""Meaningful checks for the experimental defensive learner. Run from project root."""
import copy
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brain.defense import ACTIONS, FEATURES, Brain, initial_weights, probabilities, load_policy
from tools.train_defense import learn, policy, episode, worker_init, paired_report


class PolicyTests(unittest.TestCase):
    def test_mask_and_numerical_stability(self):
        w = initial_weights()
        w[1][0] = 1e5
        p = probabilities(w, [1.0]*len(FEATURES), [0, 2, 3])
        self.assertEqual(p[1], 0)
        self.assertAlmostEqual(sum(p), 1)
        self.assertTrue(all(math.isfinite(x) for x in p))

    def test_positive_experience_increases_action_probability(self):
        w = initial_weights()
        x = [1.0]+[0.0]*(len(FEATURES)-1)
        p = probabilities(w, x, list(range(len(ACTIONS))))
        step = dict(x=x, p=p, action=2, reward=3.0)
        new, _, count = learn(w, [0.0]*len(FEATURES), [dict(trajectory=[step])])
        self.assertEqual(count, 1)
        self.assertGreater(probabilities(new, x, list(range(len(ACTIONS))))[2], p[2])

    def test_negative_experience_decreases_action_probability(self):
        w = initial_weights()
        x = [1.0]+[0.0]*(len(FEATURES)-1)
        p = probabilities(w, x, list(range(len(ACTIONS))))
        step = dict(x=x, p=p, action=3, reward=-3.0)
        new, _, _ = learn(w, [0.0]*len(FEATURES), [dict(trajectory=[step])])
        self.assertLess(probabilities(new, x, list(range(len(ACTIONS))))[3], p[3])

    def test_saved_policy_roundtrip_and_reject_bad_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"policy.json"
            path.write_text(json.dumps(policy(initial_weights())))
            self.assertEqual(load_policy(path), initial_weights())
            path.write_text(json.dumps(dict(version=-1)))
            with self.assertRaises(ValueError):
                load_policy(path)

    def test_reward_cannot_profit_from_damage_heal_cycle(self):
        b = object.__new__(Brain)
        b.last_state = (0.0, 100.0)
        b.trajectory = [dict(reward=0.0)]
        b.reward_until(1.0, 50, 100, True)
        b.reward_until(2.0, 100, 100, True)
        self.assertAlmostEqual(b.trajectory[0]["reward"], 0.08)

    def test_mortar_episode_produces_valid_training_data(self):
        worker_init()
        row = episode((initial_weights(), 100005, 1.5, True))  # mortar scenario
        self.assertEqual(row["scenario"], "mortar")
        self.assertGreater(row["decisions"], 0)
        self.assertTrue(row["trajectory"])
        self.assertTrue(any(s["x"][7] > 0 for s in row["trajectory"]))
        for step in row["trajectory"]:
            self.assertEqual(len(step["x"]), len(FEATURES))
            self.assertTrue(math.isfinite(step["reward"]))
            self.assertAlmostEqual(sum(step["p"]), 1)

    def test_cover_curriculum_offers_hiding(self):
        from tools.train_defense import scenario, Learner
        Learner.weights, Learner.training = initial_weights(), True
        found = False
        for seed in range(100002, 100032, 6):  # gun scenarios beside building corners
            s, _, _ = scenario(seed, Learner, 2)
            s.run(max_t=2)
            if any(step["x"][9] > 0 for step in s.mechs[0].bot.trajectory):
                found = True
                break
        self.assertTrue(found, "Curriculum must expose an actual reachable hiding spot")

    def test_identical_evaluation_has_zero_difference(self):
        row = dict(score=1.0, alive=True, hp=0.8, damage=0, survival_s=2,
                   zone_s=0, late=0, decisions=20, max_ms=1, scenario="gun")
        report = paired_report([row, row], [row, row])
        self.assertEqual(report["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
