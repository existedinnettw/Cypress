"""Placement problem extracted from a board, independent of KiCad and Bookshelf.

Coordinates are millimetres in KiCad's frame (x right, y down). A component's
position is the centre of its placement box, not the footprint anchor.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Box = Tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass
class Pin:
    net: str
    # offset of the pad from the component's box centre, KiCad frame (mm)
    dx: float
    dy: float


@dataclass
class Component:
    key: str  # stable id used to find the footprint again (KiCad UUID)
    ref: str
    width: float  # box size at the current orientation (mm)
    height: float
    cx: float  # box centre (mm)
    cy: float
    bottom: bool
    fixed: bool
    pins: List[Pin] = field(default_factory=list)
    # what the part occupies on the other side of the board, as boxes relative
    # to the box centre: its other-side courtyard, drilled holes, and pads
    # with copper on the other outer layer
    backside: List[Box] = field(default_factory=list)
    # keep-out areas the footprint carries (e.g. a module's antenna area),
    # relative to the box centre, on the part's own side and on the other
    # side. No other part may sit in them, but they may overlap each other
    # and may reach past the board outline.
    keepout: List[Box] = field(default_factory=list)
    backside_keepout: List[Box] = field(default_factory=list)


@dataclass
class Region:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class Problem:
    name: str
    region: Region
    components: List[Component]

    def movable(self) -> List[Component]:
        return [c for c in self.components if not c.fixed]


@dataclass
class Placement:
    """New location for one component: box centre and extra CCW rotation."""

    cx: float
    cy: float
    quarter_turns: int = 0  # counterclockwise as seen from the top, 0..3

    def rotated_size(self, comp: Component) -> Tuple[float, float]:
        if self.quarter_turns % 2:
            return comp.height, comp.width
        return comp.width, comp.height

    def box(self, comp: Component) -> Box:
        w, h = self.rotated_size(comp)
        return (self.cx - w / 2, self.cy - h / 2, self.cx + w / 2, self.cy + h / 2)

    def backside_boxes(self, comp: Component) -> List[Box]:
        return self._turn(comp.backside)

    def shapes(self, comp: Component) -> List[Tuple[int, Box, bool]]:
        """Everything the part occupies as (side, box, is_keepout); side 0 is
        top, 1 is bottom."""
        side = int(comp.bottom)
        out = [(side, self.box(comp), False)]
        out += [(1 - side, b, False) for b in self._turn(comp.backside)]
        out += [(side, b, True) for b in self._turn(comp.keepout)]
        out += [(1 - side, b, True) for b in self._turn(comp.backside_keepout)]
        return out

    def _turn(self, boxes: List[Box]) -> List[Box]:
        out = []
        for x0, y0, x1, y1 in boxes:
            # counterclockwise on screen (y down): (dx, dy) -> (dy, -dx)
            pts = [(x0, y0), (x1, y1)]
            for _ in range(self.quarter_turns % 4):
                pts = [(dy, -dx) for dx, dy in pts]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            out.append((self.cx + min(xs), self.cy + min(ys), self.cx + max(xs), self.cy + max(ys)))
        return out


Solution = Dict[str, Placement]  # Component.key -> Placement


def boxes_overlap(a: Box, b: Box, tol: float = 1e-6) -> bool:
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


def shapes_collide(a, b, tol: float = 1e-6) -> bool:
    """Do two shape lists (from Placement.shapes) collide? Shapes collide when
    they are on the same side and overlap, unless both are keep-outs."""
    return any(sa == sb and not (ka and kb) and boxes_overlap(ba, bb, tol)
               for sa, ba, ka in a for sb, bb, kb in b)


def overlaps(problem: Problem, solution: Optional[Solution] = None, tol: float = 1e-6):
    """Return pairs of refs that collide: boxes on the same side, one part's
    backside (other-side courtyard, holes, pads) under a part on the other
    side, or a part inside another part's keep-out area.

    Pairs of two fixed parts are skipped: the placer did not put them there.
    """
    placed = []
    for c in problem.components:
        p = (solution or {}).get(c.key) or Placement(c.cx, c.cy, 0)
        placed.append((c, p.shapes(c)))
    out = []
    for i, (ci, si) in enumerate(placed):
        for cj, sj in placed[i + 1:]:
            if ci.fixed and cj.fixed:
                continue
            if shapes_collide(si, sj, tol):
                out.append((ci.ref, cj.ref))
    return out
