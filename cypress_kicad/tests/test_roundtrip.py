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


@unittest.skipIf(pcbnew is None, "pcbnew not importable")
class Keepouts(unittest.TestCase):
    def test_rule_areas_become_fixed_blockers(self):
        board = pcbnew.LoadBoard(BOARD)
        zone = pcbnew.ZONE(board)
        zone.SetIsRuleArea(True)
        zone.SetDoNotAllowFootprints(True)
        zone.SetLayer(pcbnew.B_Cu)
        outline = zone.Outline()
        outline.NewOutline()
        for x, y in ((30, 20), (40, 20), (40, 26), (30, 26)):
            outline.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
        board.Add(zone)
        problem = kboard.extract(board)
        blockers = [c for c in problem.components if c.key.startswith("keepout")]
        self.assertEqual(len(blockers), 1)
        k = blockers[0]
        self.assertTrue(k.fixed and k.bottom and not k.pins)
        self.assertAlmostEqual(k.width, 10, places=3)
        self.assertAlmostEqual(k.height, 6, places=3)
        self.assertAlmostEqual(k.cx, 35, places=3)
        self.assertAlmostEqual(k.cy, 23, places=3)

    def test_footprint_keepouts_move_with_the_part(self):
        board = pcbnew.LoadBoard(BOARD)
        fp = next(f for f in board.GetFootprints() if len(list(f.Pads())) > 0)
        fp.SetLocked(False)
        if fp.IsFlipped():
            fp.Flip(fp.GetPosition(), pcbnew.FLIP_DIRECTION_LEFT_RIGHT)
        pos = fp.GetPosition()
        x0, y0 = pcbnew.ToMM(pos.x) + 5, pcbnew.ToMM(pos.y) - 2
        zone = pcbnew.ZONE(fp)
        zone.SetIsRuleArea(True)
        zone.SetDoNotAllowFootprints(False)
        zone.SetDoNotAllowPads(True)  # keeps parts out just as well
        layers = pcbnew.LSET()
        layers.AddLayer(pcbnew.F_Cu)
        layers.AddLayer(pcbnew.B_Cu)
        zone.SetLayerSet(layers)
        outline = zone.Outline()
        outline.NewOutline()
        for dx, dy in ((0, 0), (8, 0), (8, 4), (0, 4)):
            outline.Append(pcbnew.FromMM(x0 + dx), pcbnew.FromMM(y0 + dy))
        fp.Add(zone)

        problem = kboard.extract(board, keep_locked=False)
        comp = next(c for c in problem.components if c.key == kboard.footprint_key(fp))
        self.assertFalse(comp.fixed)
        self.assertEqual(len(comp.keepout), 1)
        self.assertEqual(len(comp.backside_keepout), 1)
        kx0, ky0, kx1, ky1 = comp.keepout[0]
        self.assertAlmostEqual(kx1 - kx0, 8, places=3)
        self.assertAlmostEqual(ky1 - ky0, 4, places=3)
        self.assertAlmostEqual(comp.cx + kx0, x0, places=3)
        # a movable part's keep-out is not a global-placement blocker
        self.assertFalse(any(c.key.startswith("keepout-" + comp.key) for c in problem.components))

        fp.SetLocked(True)
        problem = kboard.extract(board, keep_locked=True)
        blockers = [c for c in problem.components if c.key.startswith("keepout-" + kboard.footprint_key(fp))]
        self.assertEqual(sorted(c.bottom for c in blockers), [False, True])


class FootprintKeepouts(unittest.TestCase):
    def _module(self, **kw):
        # 10 x 6 module whose 10 x 5 antenna keep-out sticks out above it (y down)
        return Component(key="U", ref="U", width=10, height=6, cx=0, cy=0, bottom=False,
                         keepout=[(-5, -8, 5, -3)], **kw)

    def test_parts_stay_out_of_a_keepout(self):
        u = self._module(fixed=True)
        c = Component(key="C", ref="C", width=1, height=1, cx=0, cy=0, bottom=False, fixed=False)
        cb = Component(key="CB", ref="CB", width=1, height=1, cx=0, cy=0, bottom=True, fixed=False)
        problem = Problem("t", Region(-20, -20, 20, 20), [u, c, cb])
        self.assertEqual(overlaps(problem, {"C": Placement(0, -6), "CB": Placement(0, -6)}), [("U", "C")])
        legal, failed = legalize(problem, {"C": Placement(0, -6), "CB": Placement(0, -6)}, grid=0.1)
        self.assertEqual(failed, [])
        self.assertEqual(overlaps(problem, legal), [])
        self.assertEqual((legal["CB"].cx, legal["CB"].cy), (0, -6))  # other side is free

    def test_keepouts_rotate_and_may_overlap_each_other(self):
        u = self._module(fixed=False)
        # a quarter turn counterclockwise on screen: the antenna now points left
        turned = [tuple(round(v, 6) for v in box) for side, box, keepout in Placement(0, 0, 1).shapes(u) if keepout]
        self.assertEqual(turned, [(-8, -5, -3, 5)])
        # two modules side by side whose antenna areas overlap, bodies apart
        v = Component(key="V", ref="V", width=10, height=6, cx=0, cy=0, bottom=False, fixed=False,
                      keepout=[(-5, -8, 5, -3)])
        problem = Problem("t", Region(-30, -30, 30, 30), [u, v])
        sol = {"U": Placement(0, 0), "V": Placement(0, -11, 2)}  # V turned round, antenna down
        self.assertEqual(overlaps(problem, sol), [])
        legal, failed = legalize(problem, sol, grid=0.1)
        self.assertEqual(failed, [])
        self.assertEqual({k: (p.cx, p.cy) for k, p in legal.items()}, {"U": (0, 0), "V": (0, -11)})

    def test_removes_overlaps_and_respects_sides(self):
        rng = random.Random(2)
        comps = [Component(key=str(i), ref="U%d" % i, width=rng.uniform(1, 6), height=rng.uniform(1, 6),
                           cx=0, cy=0, bottom=i % 4 == 0, fixed=False) for i in range(60)]
        # a top connector with a drilled peg that also blocks the bottom
        comps.append(Component(key="J1", ref="J1", width=8, height=8, cx=20, cy=20,
                               bottom=False, fixed=True, backside=[(2, 2, 3.5, 3.5)]))
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


class Holes(unittest.TestCase):
    def test_holes_block_the_other_side_only_where_drilled(self):
        # top connector 10 x 4 at (0, 0) with a peg at its right end
        j = Component(key="J", ref="J", width=10, height=4, cx=0, cy=0, bottom=False, fixed=True,
                      backside=[(3.5, -0.5, 4.5, 0.5)])
        c = Component(key="C", ref="C", width=1, height=1, cx=0, cy=0, bottom=True, fixed=False)
        problem = Problem("t", Region(-20, -20, 20, 20), [j, c])
        self.assertEqual(overlaps(problem, {"C": Placement(-2, 0)}), [])  # under the body
        self.assertEqual(overlaps(problem, {"C": Placement(4, 0)}), [("J", "C")])  # under the peg
        # rotated a quarter turn counterclockwise the peg goes to the top (y down)
        p = Placement(0, 0, 1)
        self.assertEqual([tuple(round(v, 6) for v in b) for b in p.backside_boxes(j)], [(-0.5, -4.5, 0.5, -3.5)])
        legal, failed = legalize(problem, {"C": Placement(4, 0)}, grid=0.1)
        self.assertEqual(failed, [])
        self.assertEqual(overlaps(problem, legal), [])
        self.assertLess(math.hypot(legal["C"].cx - 4, legal["C"].cy), 1.6)


if __name__ == "__main__":
    unittest.main()
