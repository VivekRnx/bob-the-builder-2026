"""Static map model from /arena (protocol.md section 11): terrain heights, solid prisms, line of sight,
and a cheap "can I walk this way" probe.

Everything here is pure Python and sized for the 50 ms budget: a LOS check over 200 m costs ~1-2 ms.
Call the expensive methods (los, probe) a handful of times per tick, not hundreds.
"""
from __future__ import annotations

import json
import math
import os
from array import array
from typing import Dict, List, Optional, Sequence, Tuple

CELL = 16.0  # spatial-hash cell size (m) for prism lookup


def _pip(x: float, y: float, xs: List[float], ys: List[float]) -> bool:
    """Even-odd point-in-polygon."""
    inside = False
    j = len(xs) - 1
    for i in range(len(xs)):
        yi, yj = ys[i], ys[j]
        if (yi > y) != (yj > y) and x < (xs[j] - xs[i]) * (y - yi) / (yj - yi) + xs[i]:
            inside = not inside
        j = i
    return inside


class Prism:
    __slots__ = ("kind", "base", "top", "xs", "ys", "holes", "minx", "miny", "maxx", "maxy")

    def __init__(self, d: dict):
        self.kind: str = d["kind"]
        self.base: float = d["base"]
        self.top: float = d["top"]
        r = d["ring"]
        self.xs, self.ys = r[0::2], r[1::2]
        self.holes = [(h[0::2], h[1::2]) for h in d.get("holes", []) if h]
        self.minx, self.maxx = min(self.xs), max(self.xs)
        self.miny, self.maxy = min(self.ys), max(self.ys)

    def covers(self, x: float, y: float) -> bool:
        """Footprint contains (x, y) (ignores height)."""
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy:
            return False
        if not _pip(x, y, self.xs, self.ys):
            return False
        return not any(_pip(x, y, hx, hy) for hx, hy in self.holes)


class World:
    def __init__(self, map_dir: str):
        with open(os.path.join(map_dir, "v2", "world.json")) as f:
            w = json.load(f)
        b = w["bounds"]
        self.minX, self.maxX, self.minY, self.maxY = b["minX"], b["maxX"], b["minY"], b["maxY"]
        t = w["terrain"]
        self.tx0, self.ty0, self.tstep = t["minX"], t["minY"], t["step"]
        self.nx, self.ny = t["nx"], t["ny"]
        self.tscale = t.get("scale", 0.01)
        self.h = array("h")
        with open(os.path.join(map_dir, "v2", t.get("file", "terrain.i16")), "rb") as f:
            self.h.frombytes(f.read())
        if self.h.itemsize != 2:
            raise RuntimeError("int16 array unsupported")
        import sys
        if sys.byteorder != "little":
            self.h.byteswap()

        self.prisms: List[Prism] = [Prism(p) for p in w["prisms"]]
        self.grid: Dict[Tuple[int, int], List[Prism]] = {}
        for p in self.prisms:
            for ci in range(int(math.floor(p.minx / CELL)), int(math.floor(p.maxx / CELL)) + 1):
                for cj in range(int(math.floor(p.miny / CELL)), int(math.floor(p.maxy / CELL)) + 1):
                    self.grid.setdefault((ci, cj), []).append(p)

    # ------------------------------------------------------------------ queries
    def ground(self, x: float, y: float) -> float:
        """Bilinear terrain height (m) at (x, y); clamps to the grid."""
        fx = (x - self.tx0) / self.tstep
        fy = (y - self.ty0) / self.tstep
        fx = min(max(fx, 0.0), self.nx - 1.001)
        fy = min(max(fy, 0.0), self.ny - 1.001)
        i, j = int(fx), int(fy)
        ax, ay = fx - i, fy - j
        k = j * self.nx + i
        h = self.h
        h00, h10, h01, h11 = h[k], h[k + 1], h[k + self.nx], h[k + self.nx + 1]
        return ((h00 * (1 - ax) + h10 * ax) * (1 - ay) + (h01 * (1 - ax) + h11 * ax) * ay) * self.tscale

    def prisms_at(self, x: float, y: float) -> List[Prism]:
        return self.grid.get((int(math.floor(x / CELL)), int(math.floor(y / CELL))), [])

    def solid(self, x: float, y: float, z: float) -> bool:
        """Is the point inside terrain or a building/stadium/bridge prism?"""
        if z < self.ground(x, y):
            return True
        for p in self.prisms_at(x, y):
            if p.base <= z <= p.top and p.covers(x, y):
                return True
        return False

    def floor_under(self, x: float, y: float, z: float, step_up: float = 1.2) -> float:
        """Height you would stand at in column (x, y) arriving from feet height z: the highest surface
        (terrain or prism top) at or below z + step_up."""
        best = self.ground(x, y)
        for p in self.prisms_at(x, y):
            if best < p.top <= z + step_up and p.covers(x, y):
                best = p.top
        return best

    def in_bounds(self, x: float, y: float, margin: float = 3.0) -> bool:
        return self.minX + margin < x < self.maxX - margin and self.minY + margin < y < self.maxY - margin

    def los(self, a: Sequence[float], b: Sequence[float], step: float = 2.0, skip_start: float = 3.0) -> bool:
        """True if nothing solid lies on the segment a->b (trees ignored: they never stop shots).
        Samples every `step` m, skipping the first `skip_start` m (shots leave 3 m out)."""
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < 1e-6:
            return True
        n = int(d / step)
        for k in range(1, n):
            s = k * step
            if s < skip_start:
                continue
            f = s / d
            if self.solid(a[0] + dx * f, a[1] + dy * f, a[2] + dz * f):
                return False
        return True

    def probe(self, x: float, y: float, z: float, yaw: float, dist: float,
              step_up: float = 1.2, height: float = 9.0, max_slope_deg: float = 50.0, step: float = 1.5) -> float:
        """Walk a straight line from feet (x, y, z) along `yaw` and return how far you get (<= dist)
        before a wall, a too-steep slope or the map edge. Heuristic: ignores your radius except via
        the caller's margin, and lets you drop off ledges."""
        c, s = math.cos(yaw), math.sin(yaw)
        slope = math.tan(math.radians(max_slope_deg)) * step
        feet = z
        k = 1
        while k * step <= dist:
            px, py = x + c * k * step, y + s * k * step
            if not self.in_bounds(px, py):
                return (k - 1) * step
            g = self.ground(px, py)
            if g > feet + max(step_up, slope):
                return (k - 1) * step
            for p in self.prisms_at(px, py):
                # A prism blocks if it overlaps our body height and we can't step onto it.
                if p.base < feet + height and p.top > feet + step_up and p.covers(px, py):
                    return (k - 1) * step
            feet = self.floor_under(px, py, feet, step_up)
            k += 1
        return dist


def load_world(map_dir: Optional[str]) -> Optional[World]:
    """Load the map, or None (bot keeps working without map awareness)."""
    if not map_dir:
        return None
    try:
        return World(map_dir)
    except Exception as e:  # never let the map kill the bot
        import sys
        print(f"world: failed to load map from {map_dir}: {e!r}", file=sys.stderr, flush=True)
        return None
