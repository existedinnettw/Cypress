"""Geometry tests for the KiCad <-> Bookshelf translation.

The fake engine below re-implements how place_io reads our Bookshelf files
(pin offsets made relative to the lower-left corner, FN nodes mirrored) and
how the pin_pos kernel rotates pins. If the pads KiCad reports after
``board.apply`` match those pin positions, the placer optimised the same
geometry the board ends up with.

Run with a Python that can import pcbnew:
  python -m unittest discover -s cypress_kicad/tests -t .
"""

import math
import os
import random
import re
import tempfile
import unittest

from cypress_kicad import board as kboard
from cypress_kicad.bookshelf import BookshelfDesign
from cypress_kicad.legalize import legalize
from cypress_kicad.model import Component, Placement, Problem, Region, overlaps

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOARD = os.path.join(REPO, "artifacts", "kicad", "small-1.kicad_pcb")

try:
    pcbnew = kboard.import_pcbnew()
except ImportError:
    pcbnew = None


def _lines(path):
    with open(path) as f:
        return f.readlines()


def read_bookshelf(aux):
    d = os.path.dirname(aux)
    name = os.path.splitext(os.path.basename(aux))[0]
    sizes = {}
    for line in _lines(os.path.join(d, name + ".nodes")):
        m = re.match(r"^(n\d+) (\d+) (\d+)", line)
        if m:
            sizes[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    orient = {}
    for line in _lines(os.path.join(d, name + ".pl")):
        m = re.match(r"^(n\d+) \S+ \S+ : (\S+)", line)
        if m:
            orient[m.group(1)] = m.group(2)
    pins = []  # (node, offset from lower-left as place_io stores it)
    for line in _lines(os.path.join(d, name + ".nets")):
        m = re.match(r"^\s+(n\d+) I : (\S+) (\S+)", line)
        if m:
            node = m.group(1)
            w, h = sizes[node]
            ox, oy = float(m.group(2)) + w / 2, float(m.group(3)) + h / 2
            if orient[node].startswith("F"):
                ox = w - ox  # PlaceDB::updateNodePinOffset for N -> FN
            pins.append((node, ox, oy))
    return sizes, pins


def kernel_pin(x, y, w, h, ox, oy, turns):
    # pin_pos_cuda_kernel.cu::computePinPos
    t = turns * math.pi / 2
    c, s = math.cos(t), math.sin(t)
    return ((x + w / 2) + (ox - w / 2) * c - (oy - h / 2) * s,
            (y + h / 2) + (ox - w / 2) * s + (oy - h / 2) * c)


@unittest.skipIf(pcbnew is None, "pcbnew not importable")
class RoundTrip(unittest.TestCase):
    def test_pads_match_placer_geometry(self):
        board = pcbnew.LoadBoard(BOARD)
        fps = list(board.GetFootprints())
        for fp in fps[::3]:  # put a third of the parts on the bottom side
            try:
                fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
            except (AttributeError, TypeError):
                fp.Flip(fp.GetPosition(), False)
        problem = kboard.extract(board, ignore_nets="", spacing=0)
        self.assertTrue(any(c.bottom for c in problem.components))

        design = BookshelfDesign(problem, units_per_mm=10)
        with tempfile.TemporaryDirectory() as tmp:
            aux = design.write(tmp)
            sizes, pins = read_bookshelf(aux)

            rng = random.Random(1)
            result = {}
            lines = ["UCLA pl 1.0", ""]
            for node, (w, h) in sizes.items():
                x, y, turns = rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 3)
                result[node] = (x, y, turns)
                comp = problem.components[int(node[1:])]
                lines.append("%s %d %d : %s%s" % (node, x, y, "F" if comp.bottom else "", "NWSE"[turns]))
            pl = os.path.join(tmp, "out.pl")
            with open(pl, "w") as f:
                f.write("\n".join(lines) + "\n")
            solution = design.read_solution(pl)

        expected = {}  # node -> list of predicted pad positions in KiCad mm
        for node, ox, oy in pins:
            x, y, turns = result[node]
            w, h = sizes[node]
            px, py = kernel_pin(x, y, w, h, ox, oy, turns)
            expected.setdefault(node, []).append(design.from_bs(px, py))

        kboard.apply(board, problem, solution)
        by_key = {kboard.footprint_key(fp): fp for fp in board.GetFootprints()}
        nets = set(design.nets())
        checked = 0
        for comp in problem.components:
            node = design.node_names[comp.key]
            pads = [p for p in by_key[comp.key].Pads() if p.GetNetCode() > 0 and p.GetNetname() in nets]
            got = sorted((round(pcbnew.ToMM(p.GetPosition().x), 3), round(pcbnew.ToMM(p.GetPosition().y), 3))
                         for p in pads)
            want = sorted((round(x, 3), round(y, 3)) for x, y in expected.get(node, []))
            self.assertEqual(len(got), len(want), comp.ref)
            for g, e in zip(got, want):
                self.assertAlmostEqual(g[0], e[0], delta=0.002, msg=comp.ref)
                self.assertAlmostEqual(g[1], e[1], delta=0.002, msg=comp.ref)
                checked += 1
        self.assertGreater(checked, 50)


class Legalizer(unittest.TestCase):
    def test_removes_overlaps_and_respects_sides(self):
        rng = random.Random(2)
        comps = [Component(key=str(i), ref="U%d" % i, width=rng.uniform(1, 6), height=rng.uniform(1, 6),
                           cx=0, cy=0, bottom=i % 4 == 0, fixed=False) for i in range(60)]
        comps.append(Component(key="J1", ref="J1", width=8, height=8, cx=20, cy=20,
                               bottom=False, fixed=True, through_hole=True))
        problem = Problem("t", Region(0, 0, 60, 50), comps)
        sol = {c.key: Placement(30 + rng.uniform(-3, 3), 25 + rng.uniform(-3, 3), rng.randint(0, 3))
               for c in comps if not c.fixed}
        legal, failed = legalize(problem, sol, grid=0.1)
        self.assertEqual(failed, [])
        self.assertEqual(overlaps(problem, legal), [])
        for c in comps:
            if c.fixed:
                continue
            p = legal[c.key]
            w, h = p.rotated_size(c)
            self.assertGreaterEqual(p.cx - w / 2, -1e-6)
            self.assertLessEqual(p.cx + w / 2, 60 + 1e-6)
            self.assertGreaterEqual(p.cy - h / 2, -1e-6)
            self.assertLessEqual(p.cy + h / 2, 50 + 1e-6)
            self.assertEqual(p.quarter_turns, sol[c.key].quarter_turns)


if __name__ == "__main__":
    unittest.main()
