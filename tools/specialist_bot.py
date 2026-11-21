"""Run ONE experimental specialist on the real mechfight protocol.

  python tools/specialist_bot.py --module combat --policy runs/specialists-v1/combat/candidate.json

This does not change bot.py or activate other learned policies.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mechfight
from brain.specialists import MODULES, load_policy, specialist_class

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module",choices=list(MODULES),required=True)
    parser.add_argument("--policy",required=True)
    args=parser.parse_args()
    weights=load_policy(args.policy,args.module)
    mechfight.run(specialist_class(args.module,weights))

if __name__=="__main__":
    main()
