"""Entrypoint: `python3 -u bot.py`. The Dockerfile runs this.

mechfight.run() handles the protocol loop (hello/ready, newest-obs-only, tick echo, crash-safe coasting).
All decisions live in brain/brain.py.
"""
import mechfight as mf
from brain import Brain

if __name__ == "__main__":
    mf.run(Brain)
