"""Small math helpers. Frame: x east, y north, z up. Yaw 0 = east, CCW positive. Pitch up-positive."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

TAU = 2.0 * math.pi


def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % TAU - math.pi


def hdist(a: Sequence[float], b: Sequence[float]) -> float:
    """Horizontal (xy) distance."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def yaw_to(a: Sequence[float], b: Sequence[float]) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def aim_at(src: Sequence[float], dst: Sequence[float]) -> Tuple[float, float]:
    """[yaw, pitch] pointing from 3D point src to 3D point dst."""
    dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
    return math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))


def unit(yaw: float) -> Tuple[float, float]:
    return math.cos(yaw), math.sin(yaw)


def intercept_time(rel: Sequence[float], vel: Sequence[float], speed: float) -> Optional[float]:
    """Time t > 0 for a projectile at `speed` from the origin to meet a target at `rel` moving at `vel`.
    Solves |rel + vel*t| = speed*t. None if it can't catch the target."""
    a = vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2 - speed * speed
    b = 2.0 * (rel[0] * vel[0] + rel[1] * vel[1] + rel[2] * vel[2])
    c = rel[0] ** 2 + rel[1] ** 2 + rel[2] ** 2
    if abs(a) < 1e-9:
        return -c / b if b < 0 else None
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    r = math.sqrt(disc)
    ts = [t for t in ((-b - r) / (2 * a), (-b + r) / (2 * a)) if t > 0]
    return min(ts) if ts else None


def lead_point(src: Sequence[float], tgt: Sequence[float], vel: Sequence[float], speed: float,
               grounded_guess: bool = True) -> Tuple[Sequence[float], float]:
    """Where to aim a straight-flying projectile so it meets a constant-velocity target.
    Vertical velocity is ignored when the target looks grounded (it will not keep rising)."""
    v = (vel[0], vel[1], 0.0 if grounded_guess and abs(vel[2]) < 0.5 else vel[2])
    rel = (tgt[0] - src[0], tgt[1] - src[1], tgt[2] - src[2])
    t = intercept_time(rel, v, speed)
    if t is None:
        return tgt, math.dist(src, tgt) / speed
    return (tgt[0] + v[0] * t, tgt[1] + v[1] * t, tgt[2] + v[2] * t), t


def ballistic_pitch(horiz: float, dz: float, speed: float, g: float, high: bool = False) -> Optional[float]:
    """Launch pitch to hit a point `horiz` m away and `dz` m higher with speed `speed` under gravity g.
    Low arc by default, high arc (lob over buildings) with high=True. None if out of range."""
    if horiz < 1e-3:
        return math.pi / 2 if dz > 0 else -math.pi / 2
    v2 = speed * speed
    disc = v2 * v2 - g * (g * horiz * horiz + 2 * dz * v2)
    if disc < 0:
        return None
    r = math.sqrt(disc)
    return math.atan((v2 + r) / (g * horiz)) if high else math.atan((v2 - r) / (g * horiz))


def flight_time(horiz: float, pitch: float, speed: float) -> float:
    c = math.cos(pitch)
    return horiz / (speed * c) if c > 1e-3 else 99.0
