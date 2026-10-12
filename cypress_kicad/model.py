"""Placement problem extracted from a board, independent of KiCad and Bookshelf.

Coordinates are millimetres in KiCad's frame (x right, y down). A component's
position is the centre of its placement box, not the footprint anchor.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    through_hole: bool = False  # blocks both sides during legalization
    pins: List[Pin] = field(default_factory=list)


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


Solution = Dict[str, Placement]  # Component.key -> Placement


def overlaps(problem: Problem, solution: Optional[Solution] = None, tol: float = 1e-6):
    """Return pairs of refs whose boxes overlap on the same side.

    Pairs of two fixed parts are skipped: the placer did not put them there.
    """
    boxes = []
    for c in problem.components:
        p = (solution or {}).get(c.key)
        if p is None:
            p = Placement(c.cx, c.cy, 0)
        w, h = p.rotated_size(c)
        boxes.append((c, p.cx - w / 2, p.cy - h / 2, p.cx + w / 2, p.cy + h / 2))
    out = []
    for i in range(len(boxes)):
        ci, ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            cj, bx0, by0, bx1, by1 = boxes[j]
            if ci.fixed and cj.fixed:
                continue
            same_side = ci.bottom == cj.bottom or ci.through_hole or cj.through_hole
            if not same_side:
                continue
            if ax0 < bx1 - tol and bx0 < ax1 - tol and ay0 < by1 - tol and by0 < ay1 - tol:
                out.append((ci.ref, cj.ref))
    return out
