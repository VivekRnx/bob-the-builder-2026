"""Ground-level navigation grid + time-budgeted A*.

The map is fixed, so the grid (4 m cells, ~48k of them) is built once in __init__ from World (~1 s,
well inside the 20 s hello budget). Cells are blocked by buildings/stadium a mech can't stand beside
(footprint dilated by the mech radius), by steep terrain, and by the map edge. Bridges high enough to
walk under don't block. Roofs aren't modeled: the planner keeps you on the ground; jets are for unsticking.

A* runs incrementally: plan() spends at most `budget_ms` per call and resumes next tick, so a long
route never blows the 50 ms answer budget.
"""
from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Tuple

from .world import World

SQ2 = math.sqrt(2.0)
NEIGH = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, SQ2), (1, -1, SQ2), (-1, 1, SQ2), (-1, -1, SQ2)]


class NavGrid:
    def __init__(self, world: World, cell: float = 4.0, radius: float = 2.5, height: float = 9.0,
                 step_up: float = 1.2, max_slope_deg: float = 50.0):
        self.w = world
        self.cell = cell
        self.x0, self.y0 = world.minX, world.minY
        self.nx = int((world.maxX - world.minX) / cell) + 1
        self.ny = int((world.maxY - world.minY) / cell) + 1
        n = self.nx * self.ny
        self.ground = [0.0] * n
        self.free = bytearray(n)
        slope = math.tan(math.radians(max_slope_deg))
        margin = radius + 0.5
        # 1) terrain height and slope per cell
        for j in range(self.ny):
            for i in range(self.nx):
                x, y = self.x0 + (i + 0.5) * cell, self.y0 + (j + 0.5) * cell
                g = world.ground(x, y)
                self.ground[j * self.nx + i] = g
                if not world.in_bounds(x, y, margin + 1):
                    continue
                h = cell * 0.5
                gx = abs(world.ground(x + h, y) - world.ground(x - h, y)) / cell
                gy = abs(world.ground(x, y + h) - world.ground(x, y - h)) / cell
                if gx < slope * 0.9 and gy < slope * 0.9:
                    self.free[j * self.nx + i] = 1
        # 2) rasterize prisms (dilated by the mech radius) onto the grid
        offs = [(0.0, 0.0), (margin, 0.0), (-margin, 0.0), (0.0, margin), (0.0, -margin),
                (margin * 0.7, margin * 0.7), (-margin * 0.7, margin * 0.7),
                (margin * 0.7, -margin * 0.7), (-margin * 0.7, -margin * 0.7)]
        for p in world.prisms:
            i0 = max(0, int((p.minx - margin - self.x0) / cell))
            i1 = min(self.nx - 1, int((p.maxx + margin - self.x0) / cell))
            j0 = max(0, int((p.miny - margin - self.y0) / cell))
            j1 = min(self.ny - 1, int((p.maxy + margin - self.y0) / cell))
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    k = j * self.nx + i
                    if not self.free[k]:
                        continue
                    g = self.ground[k]
                    if p.base >= g + height or p.top <= g + step_up:
                        continue  # walk under (bridge) or step over
                    x, y = self.x0 + (i + 0.5) * cell, self.y0 + (j + 0.5) * cell
                    for ox, oy in offs:
                        if p.covers(x + ox, y + oy):
                            self.free[k] = 0
                            break
        self.max_dz = max(step_up, slope * cell * 0.9)
        self._search: Optional[dict] = None
        self._components()

    def _components(self) -> None:
        """Label walkable regions (4-connected). Small regions are courtyards/pockets you can only leave
        by jetting over a wall; `main` is the big connected campus."""
        from collections import deque
        nx, ny, free, ground, max_dz = self.nx, self.ny, self.free, self.ground, self.max_dz
        comp = [-1] * (nx * ny)
        sizes: List[int] = []
        for k in range(nx * ny):
            if not free[k] or comp[k] >= 0:
                continue
            c = len(sizes)
            comp[k] = c
            q = deque([k])
            n = 0
            while q:
                u = q.popleft()
                n += 1
                i, j = u % nx, u // nx
                for v, ok in ((u + 1, i + 1 < nx), (u - 1, i > 0), (u + nx, j + 1 < ny), (u - nx, j > 0)):
                    if ok and free[v] and comp[v] < 0 and abs(ground[v] - ground[u]) <= max_dz:
                        comp[v] = c
                        q.append(v)
            sizes.append(n)
        self.comp = comp
        self.comp_size = sizes
        self.main = max(range(len(sizes)), key=lambda c: sizes[c]) if sizes else -1

    def component_at(self, x: float, y: float) -> int:
        c = self.nearest_free(*self.cell_of(x, y), 3)
        return self.comp[c[1] * self.nx + c[0]] if c else -1

    def nearest_in_component(self, x: float, y: float, comp: int, max_r: int = 40) -> Optional[Tuple[float, float]]:
        """Closest cell center (ignoring walls) that belongs to region `comp`."""
        i, j = self.cell_of(x, y)
        for r in range(1, max_r + 1):
            best = None
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny and self.comp[nj * self.nx + ni] == comp:
                        d = di * di + dj * dj
                        if best is None or d < best[0]:
                            best = (d, ni, nj)
            if best:
                return self.center(best[1], best[2])
        return None

    # ------------------------------------------------------------------ helpers
    def cell_of(self, x: float, y: float) -> Tuple[int, int]:
        i = min(self.nx - 1, max(0, int((x - self.x0) / self.cell)))
        j = min(self.ny - 1, max(0, int((y - self.y0) / self.cell)))
        return i, j

    def center(self, i: int, j: int) -> Tuple[float, float]:
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def is_free(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny and self.free[j * self.nx + i] == 1

    def nearest_free(self, i: int, j: int, max_r: int = 12) -> Optional[Tuple[int, int]]:
        if self.is_free(i, j):
            return i, j
        for r in range(1, max_r + 1):
            best = None
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    if self.is_free(i + di, j + dj):
                        d = di * di + dj * dj
                        if best is None or d < best[0]:
                            best = (d, i + di, j + dj)
            if best:
                return best[1], best[2]
        return None

    # ------------------------------------------------------------------ A*
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], budget_ms: float = 8.0,
             max_nodes: int = 60000) -> Optional[List[Tuple[float, float]]]:
        """Returns a list of waypoints (world xy) when done, [] if unreachable, None if still searching
        (call again next tick with the same goal; the search resumes)."""
        s = self.nearest_free(*self.cell_of(*start))
        g = self.nearest_free(*self.cell_of(*goal))
        if s is None or g is None:
            return []
        key = (g, )
        st = self._search
        if st is None or st["goal"] != g:
            st = self._search = {"goal": g, "start": s, "open": [(0.0, 0.0, s)], "came": {s: None},
                                 "cost": {s: 0.0}, "n": 0}
        deadline = time.perf_counter() + budget_ms / 1000.0
        nx, free, ground, max_dz = self.nx, self.free, self.ground, self.max_dz
        gi, gj = g
        open_, came, cost = st["open"], st["came"], st["cost"]
        steps = 0
        while open_:
            steps += 1
            if steps % 256 == 0 and time.perf_counter() > deadline:
                return None
            _, c, cur = heapq.heappop(open_)
            if cur == g:
                self._search = None
                return self._reconstruct(came, cur)
            if c > cost.get(cur, 1e18):
                continue
            ci, cj = cur
            hk = ground[cj * nx + ci]
            for di, dj, w in NEIGH:
                ni, nj = ci + di, cj + dj
                if ni < 0 or nj < 0 or ni >= nx or nj >= self.ny:
                    continue
                k = nj * nx + ni
                if not free[k] or abs(ground[k] - hk) > max_dz:
                    continue
                if di and dj and not (free[cj * nx + ni] and free[nj * nx + ci]):
                    continue  # no corner cutting
                nc = c + w
                if nc < cost.get((ni, nj), 1e18):
                    cost[(ni, nj)] = nc
                    came[(ni, nj)] = cur
                    dx, dy = abs(ni - gi), abs(nj - gj)
                    h = (dx + dy) + (SQ2 - 2) * min(dx, dy)
                    heapq.heappush(open_, (nc + h, nc, (ni, nj)))
            st["n"] += 1
            if st["n"] > max_nodes:
                break
        self._search = None
        return []

    def _reconstruct(self, came, cur) -> List[Tuple[float, float]]:
        out = []
        while cur is not None:
            out.append(self.center(*cur))
            cur = came[cur]
        out.reverse()
        return out
