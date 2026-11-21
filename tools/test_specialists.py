"""Checks for independently trained specialists; no external dependencies."""
import copy
import gzip
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from brain.brain import Brain as Current, TUNING
from brain.specialists import MODULES, FEATURES, features, initial_weights, probabilities, specialist_class, policy_document, load_policy
from tools.train_specialists import learn, episode, seed_set, paired, reward_delta
from tools.train_defense import worker_init

with gzip.open(ROOT/"mechfight-kit"/"check"/"stream.jsonl.gz","rt") as f:
    messages=[json.loads(line) for line in f]
HELLO=messages[0]
HELLO["map"]["dir"]=str(ROOT/"mechfight-kit"/"arena")
OBS=next(m for m in messages if m.get("type")=="obs")


class SpecialistTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker_init()

    def test_baseline_equivalence_for_all_modules(self):
        reference=Current(copy.deepcopy(HELLO))
        reference.T["debug"]=False
        wanted=reference.act(copy.deepcopy(OBS))
        for module in MODULES:
            with self.subTest(module=module):
                b=specialist_class(module)(copy.deepcopy(HELLO))
                self.assertEqual(b.act(copy.deepcopy(OBS)),wanted)

    def test_profile_configuration_does_not_mutate_global_tuning(self):
        before=copy.deepcopy(TUNING)
        for module in MODULES:
            b=specialist_class(module,initial_weights())(copy.deepcopy(HELLO))
            for choice in range(5):
                b.choice=choice
                b.configure()
        self.assertEqual(TUNING,before)

    def test_features_use_only_observation_and_public_memory(self):
        x=features(copy.deepcopy(OBS),{})
        self.assertEqual(len(x),len(FEATURES))
        self.assertTrue(all(math.isfinite(v) for v in x))

    def test_each_module_gets_independent_policy_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"policy.json"
            for module in MODULES:
                path.write_text(json.dumps(policy_document(module,initial_weights())))
                self.assertEqual(load_policy(path,module),initial_weights())
                other=next(m for m in MODULES if m!=module)
                with self.assertRaises(ValueError):
                    load_policy(path,other)

    def test_seed_streams_do_not_overlap(self):
        sets=[set(seed_set(m,s,24)) for m in MODULES for s in ("train","validation","test")]
        for i, a in enumerate(sets):
            for b in sets[i+1:]:
                self.assertFalse(a&b)

    def test_learning_favors_rewarded_action_and_respects_mask(self):
        w=initial_weights()
        x=[1.0]+[0.0]*(len(FEATURES)-1)
        p=probabilities(w,x,[0,2])
        self.assertEqual(p[1],0)
        rows=[dict(trajectory=[dict(x=x,p=p,action=2,reward=4)])]
        trained,_,samples,choices=learn(w,[0.0]*len(FEATURES),rows)
        self.assertGreater(probabilities(trained,x,[0,2])[2],p[2])
        self.assertEqual((samples,choices),(1,1))

    def test_navigation_potential_cannot_reward_round_trip(self):
        start=dict(t=0,hp=100,max_hp=100,out=0,stuck=0,dist=100)
        near=dict(start,t=1,dist=50)
        back=dict(start,t=2)
        total=reward_delta("navigation",start,near,0)+reward_delta("navigation",near,back,0)
        self.assertAlmostEqual(total,0.08)

    def test_paired_evaluation_rejects_different_seeds(self):
        with self.assertRaises(ValueError):
            paired([dict(seed=1)],[dict(seed=2)])

    def test_strategy_retains_zone_emergency(self):
        b=specialist_class("strategy",initial_weights())(copy.deepcopy(HELLO))
        b.choice=2
        b.configure()
        obs=copy.deepcopy(OBS)
        obs["self"]["in_zone"]=False
        b.perceive(obs)
        result=b.decide(obs,b.pick_enemy(obs),None,False)
        self.assertTrue(result[1])
        self.assertEqual(result[2],"zone!")

    def test_each_module_produces_finite_training_transitions(self):
        for i,module in enumerate(MODULES):
            with self.subTest(module=module):
                row=episode((module,initial_weights(),100000+i*100000,1.5,True))
                self.assertGreater(row["decisions"],0)
                self.assertTrue(row["trajectory"])
                self.assertTrue(math.isfinite(row["score"]))
                for step in row["trajectory"]:
                    self.assertAlmostEqual(sum(step["p"]),1)
                    self.assertTrue(math.isfinite(step["reward"]))


if __name__=="__main__":
    unittest.main()
