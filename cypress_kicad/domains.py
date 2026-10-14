"""Isolation domains: keep two groups of nets apart, as the board's rules say.

A custom rule like

  (rule "field logic isolation"
    (condition "A.NetClass == 'Logic*' && B.NetClass != 'Logic*'")
    (constraint clearance (min 0.75mm)))

splits the board into two domains: nets whose class matches the pattern
("in") and all others ("out"). A part belongs to the domain of its pads'
nets; a part with pads in both is a bridge (isolator, optocoupler,
connector). The domain with less part area is the inner one.

Where the designer drew the inner domain -- a copper pour on its nets that
covers only part of the board, such as an isolated ground island -- that
outline is its region. Placement then runs in two phases:

  1. the outer domain is placed outside region + bridge band + rule gap,
     with the inner parts and bridges parked at the region's centre so their
     nets still pull the outer parts towards it;
  2. the outer domain is fixed and the inner parts and bridges are placed
     inside region + band; bridges can straddle the band.

Measured on a 4-layer isolated I/O board: routed unconnected nets went from
38-44 (plain) to 13-17, while regions chosen without the designer's pour
did not help (32-34). So without a drawn region the domains are reported
but not enforced.

Pure Python: the KiCad side (rule text, net classes, zone outlines) is read
by board.py.
"""

import fnmatch
import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

from .model import Box, Component, Pin, Placement, Problem, Solution

BAND = 3.0  # mm around the region in which bridges may straddle the boundary
CELL = 0.5  # mm raster used to turn region outlines into rectangles

_RULE_NAME = re.compile(r"\(\s*rule\s+(\"[^\"]*\"|[^\s()]+)")
_COND = re.compile(
    r"([AB])\s*\.\s*NetClass\s*==\s*['\"]([^'\"]+)['\"]\s*&&\s*([AB])\s*\.\s*NetClass\s*!=\s*['\"]([^'\"]+)['\"]")
_MIN = re.compile(r"\(\s*constraint\s+clearance\s+\(\s*min\s+([0-9.]+)\s*(mm|mil)?\s*\)")


@dataclass
class Domains:
    rule: str
    pattern: str  # net class pattern of the "in" domain (KiCad wildcards)
    gap: float  # mm the rule keeps between the domains
    inner: str = "in"  # the domain confined to the region: "in" or "out"
    region: List[Box] = field(default_factory=list)  # rectangles covering the inner region
    band: float = BAND


def _rules(text: str):
    """(name, body) of every top-level (rule ...) list; # comments skipped."""
    text = re.sub(r"(?m)^\s*#.*$", "", text or "")
    i = 0
    while True:
        m = _RULE_NAME.search(text, i)
        if not m:
            return
        depth, j, in_str = 0, m.start(), False
        while j < len(text):
            ch = text[j]
            if in_str:
                in_str = ch != '"' or text[j - 1] == "\\"
            elif ch == '"':
                in_str = True
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        yield m.group(1).strip('"'), text[m.end():j]
        i = j + 1


def read_barrier(dru_text: str) -> Optional[Tuple[str, str, float]]:
    """(rule name, class pattern, gap mm) of the widest clearance rule that
    separates one net-class pattern from everything else, or None."""
    best = None
    for name, body in _rules(dru_text):
        if re.search(r"\(\s*severity\s+ignore\s*\)", body) or re.search(r"\(\s*layer\s", body):
            continue
        c = _COND.search(body)
        v = _MIN.search(body)
        if not c or not v or c.group(1) == c.group(3) or c.group(2) != c.group(4):
            continue
        gap = float(v.group(1)) * (0.0254 if v.group(2) == "mil" else 1.0)
        if best is None or gap > best[2]:
            best = (name, c.group(2), gap)
    return best


def in_class(class_name: str, pattern: str) -> bool:
    """KiCad's NetClass == pattern: wildcard match, case-insensitive; a net
    with several classes reports them comma-separated."""
    names = [n.strip() for n in (class_name or "Default").split(",")] or ["Default"]
    return any(fnmatch.fnmatchcase(n.lower(), pattern.lower()) for n in names)


def domain_of(sides: set) -> Optional[str]:
    if not sides:
        return None
    return "bridge" if len(sides) == 2 else next(iter(sides))


def rects_of(inside: Callable[[float, float], bool], region, grow: float = 0.0,
             invert: bool = False, cell: float = CELL) -> List[Box]:
    """Cover the raster cells of ``region`` (x0, y0, x1, y1) where ``inside``
    holds -- grown by ``grow`` mm, or with ``invert`` where it does not --
    with few rectangles: runs per row, rows with identical runs merged."""
    x0, y0, x1, y1 = region
    nx, ny = int(math.ceil((x1 - x0) / cell)), int(math.ceil((y1 - y0) / cell))
    grid = [[inside(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell) for i in range(nx)] for j in range(ny)]
    r = int(math.ceil(grow / cell - 1e-9)) if grow > 0 else 0
    if r:
        grown = [[False] * nx for _ in range(ny)]
        for j in range(ny):
            for i in range(nx):
                if grid[j][i]:
                    for jj in range(max(0, j - r), min(ny, j + r + 1)):
                        row = grown[jj]
                        for ii in range(max(0, i - r), min(nx, i + r + 1)):
                            row[ii] = True
        grid = grown
    if invert:
        grid = [[not v for v in row] for row in grid]
    rects: List[Box] = []
    open_runs: Dict[Tuple[int, int], int] = {}
    for j in range(ny + 1):
        runs = set()
        if j < ny:
            i = 0
            while i < nx:
                if grid[j][i]:
                    k = i
                    while k < nx and grid[j][k]:
                        k += 1
                    runs.add((i, k))
                    i = k
                else:
                    i += 1
        for run in [run for run in open_runs if run not in runs]:
            j0 = open_runs.pop(run)
            rects.append((x0 + run[0] * cell, y0 + j0 * cell, x0 + run[1] * cell, y0 + j * cell))
        for run in runs:
            open_runs.setdefault(run, j)
    return rects


def _inside(rects: List[Box]):
    return lambda x, y: any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in rects)


def _blockers(rects: List[Box], tag: str) -> List[Component]:
    out = []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        for bottom in (False, True):
            out.append(Component(key="domain-%s-%d-%s" % (tag, i, "bottom" if bottom else "top"),
                                 ref="domain keep-out", width=x1 - x0, height=y1 - y0,
                                 cx=(x0 + x1) / 2, cy=(y0 + y1) / 2, bottom=bottom,
                                 fixed=True, is_keepout=True))
    return out


def placed_copy(comp: Component, place: Placement) -> Component:
    """``comp`` fixed where ``place`` puts it, with its geometry turned."""
    def turn(dx, dy):
        for _ in range(place.quarter_turns % 4):
            dx, dy = dy, -dx  # counterclockwise on screen, as Placement._turn
        return dx, dy

    def rel(boxes):
        return [(b[0] - place.cx, b[1] - place.cy, b[2] - place.cx, b[3] - place.cy)
                for b in place._turn(boxes)]

    w, h = place.rotated_size(comp)
    return replace(comp, width=w, height=h, cx=place.cx, cy=place.cy, fixed=True,
                   pins=[Pin(p.net, *turn(p.dx, p.dy)) for p in comp.pins],
                   backside=rel(comp.backside), keepout=rel(comp.keepout),
                   backside_keepout=rel(comp.backside_keepout))


def _box(problem: Problem):
    r = problem.region
    return (r.left, r.top, r.right, r.bottom)


def phase1(problem: Problem) -> Problem:
    """Outer domain movable, kept out of region + band + gap; inner parts and
    bridges parked at the region's centre."""
    d = problem.domains
    rx0 = min(b[0] for b in d.region)
    ry0 = min(b[1] for b in d.region)
    rx1 = max(b[2] for b in d.region)
    ry1 = max(b[3] for b in d.region)
    cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    comps = []
    for c in problem.components:
        if not c.fixed and c.domain in (d.inner, "bridge"):
            c = replace(c, cx=cx, cy=cy, fixed=True)
        comps.append(c)
    keep = rects_of(_inside(d.region), _box(problem), grow=d.band + d.gap)
    return replace(problem, components=comps + _blockers(keep, "out"))


def phase2(problem: Problem, outer: Solution) -> Problem:
    """Outer domain fixed where phase 1 put it; inner parts and bridges
    movable, kept inside region + band."""
    d = problem.domains
    comps = []
    for c in problem.components:
        if not c.fixed and c.domain not in (d.inner, "bridge") and c.key in outer:
            c = placed_copy(c, outer[c.key])
        comps.append(c)
    keep = rects_of(_inside(d.region), _box(problem), grow=d.band, invert=True)
    return replace(problem, components=comps + _blockers(keep, "in"))


def enforced(problem: Problem) -> bool:
    d = problem.domains
    return bool(d and d.region and any(not c.fixed and c.domain in (d.inner, "bridge")
                                       for c in problem.components))


def place(problem: Problem, solve: Callable[[Problem], Solution]) -> Solution:
    """Two-phase placement for ``problem`` with enforced domains. ``solve``
    runs global placement (and legalization) on a problem and returns its
    solution; the merged solution is for the original components."""
    first = phase1(problem)
    outer = solve(first)
    second = phase2(problem, outer)
    inner = solve(second)
    merged = dict(outer)
    merged.update(inner)
    return {k: v for k, v in merged.items() if not k.startswith("domain-")}


def summary(problem: Problem) -> str:
    d = problem.domains
    if not d:
        return ""
    n = {k: sum(1 for c in problem.components if c.domain == k) for k in ("in", "out", "bridge")}
    where = ("region from the designer's pour" if d.region
             else "no region drawn for it (a partial pour on its nets), so not enforced")
    return ("domains from rule '%s' (%s, %.2f mm): %d in, %d out, %d bridges; inner '%s', %s"
            % (d.rule, d.pattern, d.gap, n["in"], n["out"], n["bridge"], d.inner, where))
