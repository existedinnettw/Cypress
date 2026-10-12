"""Translate a Problem to Bookshelf files for Cypress, and read its .pl back.

Bookshelf uses a y-up frame with the origin at the lower-left corner of the
placement region, so y is mirrored relative to KiCad. With the mirror, a
counterclockwise rotation in Bookshelf is also counterclockwise on screen in
KiCad, which keeps orientation handling a plain quarter-turn count.
"""

import math
import os
import re
from collections import OrderedDict
from typing import Dict, Tuple

from .model import Placement, Problem, Solution

ORIENT_TURNS = {"N": 0, "W": 1, "S": 2, "E": 3}


class BookshelfDesign:
    """Bookshelf encoding of a Problem; remembers names to decode results."""

    def __init__(self, problem: Problem, units_per_mm: float = 10.0):
        self.problem = problem
        self.scale = units_per_mm
        self.node_names: Dict[str, str] = {}  # Component.key -> node name
        self.node_sizes: Dict[str, Tuple[int, int]] = {}  # node name -> (w, h)
        self.key_by_node: Dict[str, str] = {}
        for i, comp in enumerate(problem.components):
            name = "n%d" % i
            self.node_names[comp.key] = name
            self.key_by_node[name] = comp.key
            self.node_sizes[name] = (
                max(1, math.ceil(comp.width * self.scale - 1e-9)),
                max(1, math.ceil(comp.height * self.scale - 1e-9)),
            )

    # frame conversion -----------------------------------------------------
    def to_bs(self, x: float, y: float) -> Tuple[float, float]:
        r = self.problem.region
        return (x - r.left) * self.scale, (r.bottom - y) * self.scale

    def from_bs(self, x: float, y: float) -> Tuple[float, float]:
        r = self.problem.region
        return x / self.scale + r.left, r.bottom - y / self.scale

    # writing ----------------------------------------------------------------
    def nets(self):
        nets = OrderedDict()
        for comp in self.problem.components:
            for pin in comp.pins:
                nets.setdefault(pin.net, []).append((comp, pin))
        return OrderedDict((k, v) for k, v in nets.items() if len({c.key for c, _ in v}) > 1)

    def write(self, directory: str) -> str:
        """Write the design and return the path of the .aux file."""
        os.makedirs(directory, exist_ok=True)
        name = self.problem.name
        path = lambda ext: os.path.join(directory, "%s.%s" % (name, ext))
        comps = self.problem.components

        lines = ["UCLA nodes 1.0", ""]
        lines.append("NumNodes : %d" % len(comps))
        lines.append("NumTerminals : %d" % sum(c.fixed for c in comps))
        for comp in comps:
            node = self.node_names[comp.key]
            w, h = self.node_sizes[node]
            lines.append("%s %d %d%s" % (node, w, h, " terminal" if comp.fixed else ""))
        _write(path("nodes"), lines)

        nets = self.nets()
        lines = ["UCLA nets 1.0", ""]
        lines.append("NumNets : %d" % len(nets))
        lines.append("NumPins : %d" % sum(len(p) for p in nets.values()))
        for i, (net, pins) in enumerate(nets.items()):
            lines.append("NetDegree : %d net%d" % (len(pins), i))
            for comp, pin in pins:
                # place_io mirrors pin offsets of FN (bottom) nodes about the
                # vertical axis; pre-mirror so the placer sees board-view pins
                dx = -pin.dx if comp.bottom else pin.dx
                lines.append("\t%s I : %.4f %.4f" % (
                    self.node_names[comp.key], dx * self.scale, -pin.dy * self.scale))
        _write(path("nets"), lines)

        lines = ["UCLA pl 1.0", ""]
        for comp in comps:
            node = self.node_names[comp.key]
            w, h = self.node_sizes[node]
            x, y = self.to_bs(comp.cx, comp.cy)
            lines.append("%s %.4f %.4f : %s%s" % (
                node, x - w / 2, y - h / 2, "FN" if comp.bottom else "N",
                " /FIXED" if comp.fixed else ""))
        _write(path("pl"), lines)

        region = self.problem.region
        num_rows = max(1, math.ceil(region.height * self.scale))
        num_sites = max(1, math.ceil(region.width * self.scale))
        lines = ["UCLA scl 1.0", "", "NumRows : %d" % num_rows, ""]
        for i in range(num_rows):
            lines += [
                "CoreRow Horizontal",
                "    Coordinate   : %d" % i,
                "    Height       : 1",
                "    Sitewidth    : 1",
                "    Sitespacing  : 1",
                "    Siteorient   : %d" % (1 - i % 2),
                "    Sitesymmetry : 1",
                "    SubrowOrigin : 0  NumSites : %d" % num_sites,
                "End",
            ]
        _write(path("scl"), lines)

        _write(path("aux"), ["RowBasedPlacement : %s.nodes %s.nets %s.pl %s.scl" % ((name,) * 4)])
        return path("aux")

    # reading ----------------------------------------------------------------
    def read_solution(self, pl_file: str) -> Solution:
        """Decode a .pl written by Cypress into box centres and rotations."""
        pattern = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s*:\s*(F?)([NSEW])\b")
        solution: Solution = {}
        with open(pl_file) as f:
            for line in f:
                m = pattern.match(line.strip())
                if not m or m.group(1) not in self.key_by_node:
                    continue
                node = m.group(1)
                w, h = self.node_sizes[node]
                # (x, y) is the lower-left corner of the unrotated box; the
                # placer rotates around the box centre
                cx, cy = self.from_bs(float(m.group(2)) + w / 2, float(m.group(3)) + h / 2)
                solution[self.key_by_node[node]] = Placement(cx, cy, ORIENT_TURNS[m.group(5)])
        return solution


def _write(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
