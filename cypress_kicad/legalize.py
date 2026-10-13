"""Remove overlaps left by global placement.

Cypress's tuned flows stop after global placement, so parts can still overlap.
This pass keeps each part as close as possible to where the placer put it:
largest parts first, each moved to the nearest grid position (Euclidean) that
does not overlap anything already placed on the same side. What a part
occupies on the other side (its other-side courtyard, holes, pads) blocks
that side too. Pure Python so it runs inside KiCad's
interpreter.
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from .model import Box, Placement, Problem, Solution

EPS = 1e-6


class _Occupancy:
    def __init__(self, bucket: float):
        self.bucket = bucket
        self.cells: Dict[Tuple[int, int, int], List[Box]] = defaultdict(list)

    def _keys(self, box: Box, side: int):
        b = self.bucket
        for ix in range(math.floor(box[0] / b), math.floor(box[2] / b) + 1):
            for iy in range(math.floor(box[1] / b), math.floor(box[3] / b) + 1):
                yield side, ix, iy

    def add(self, box: Box, side: int):
        for key in self._keys(box, side):
            self.cells[key].append(box)

    def free(self, box: Box, side: int) -> bool:
        for key in self._keys(box, side):
            for o in self.cells.get(key, ()):
                if box[0] < o[2] - EPS and o[0] < box[2] - EPS and box[1] < o[3] - EPS and o[1] < box[3] - EPS:
                    return False
        return True


def _fits(occ, comp, place) -> bool:
    side = int(comp.bottom)
    return occ.free(place.box(comp), side) and all(occ.free(h, 1 - side) for h in place.backside_boxes(comp))


def _take(occ, comp, place):
    side = int(comp.bottom)
    occ.add(place.box(comp), side)
    for h in place.backside_boxes(comp):
        occ.add(h, 1 - side)


def legalize(problem: Problem, solution: Solution, grid: float = 0.1):
    """Return (legal solution, refs that could not be legalized)."""
    region = problem.region
    sizes = [max(c.width, c.height) for c in problem.components] or [1.0]
    occ = _Occupancy(bucket=max(2.0, 2 * sorted(sizes)[len(sizes) // 2]))
    for comp in problem.components:
        if comp.fixed:
            _take(occ, comp, Placement(comp.cx, comp.cy, 0))

    def snap(v, lo, hi):
        # snap the centre to the grid but keep the box inside [lo, hi]
        v = round(v / grid) * grid
        if v < lo:
            v = math.ceil((lo - EPS) / grid) * grid
        if v > hi:
            v = math.floor((hi + EPS) / grid) * grid
        return v if lo - EPS <= v <= hi + EPS else (lo + hi) / 2

    movable = [c for c in problem.components if not c.fixed and c.key in solution]
    movable.sort(key=lambda c: c.width * c.height, reverse=True)
    max_rings = int(math.ceil(max(region.width, region.height) / grid)) + 1

    result: Solution = {}
    failed = []
    for comp in movable:
        p = solution[comp.key]
        w, h = p.rotated_size(comp)
        lo_x, hi_x = region.left + w / 2, region.right - w / 2
        lo_y, hi_y = region.top + h / 2, region.bottom - h / 2
        if lo_x > hi_x or lo_y > hi_y:
            failed.append(comp.ref)
            result[comp.key] = p
            continue
        x0, y0 = snap(p.cx, lo_x, hi_x), snap(p.cy, lo_y, hi_y)

        best = None
        for k in range(max_rings):
            if best is not None and k * grid > best[0]:
                break  # every later ring is farther than what we have
            ring = []
            for i in range(-k, k + 1):
                for j in ((-k, k) if abs(i) != k else range(-k, k + 1)):
                    x, y = x0 + i * grid, y0 + j * grid
                    if lo_x - EPS <= x <= hi_x + EPS and lo_y - EPS <= y <= hi_y + EPS:
                        ring.append((math.hypot(x - p.cx, y - p.cy), x, y))
            ring.sort()
            for d, x, y in ring:
                if best is not None and d >= best[0]:
                    break
                if _fits(occ, comp, Placement(x, y, p.quarter_turns)):
                    best = (d, x, y)
                    break
        if best is None:
            failed.append(comp.ref)
            x, y = x0, y0
        else:
            _, x, y = best
        result[comp.key] = Placement(x, y, p.quarter_turns)
        _take(occ, comp, result[comp.key])
    return result, failed
