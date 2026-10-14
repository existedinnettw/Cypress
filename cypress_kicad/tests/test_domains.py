"""Isolation domains (domains.py): rule parsing, region rectangles and the
two-phase placement, with a stand-in for the engine.

  python -m unittest discover -s cypress_kicad/tests -t .
"""

import unittest

from cypress_kicad import domains as D
from cypress_kicad.legalize import legalize
from cypress_kicad.model import Component, Pin, Placement, Problem, Region, overlaps

RULES = """(version 1)
# Logic and field side of an isolated I/O board
(rule "field logic isolation"
	(condition "A.NetClass == 'Logic*' && B.NetClass != 'Logic*'")
	(constraint clearance (min 0.75mm)))

(rule "U5 fine pitch"
	(condition "A.Type == 'Pad' && B.Type == 'Pad' && A.memberOfFootprint('U5')")
	(constraint clearance (min 0.15mm)))

(rule "narrower barrier"
	(condition "B.NetClass == 'HV' && A.NetClass != 'HV'")
	(constraint clearance (min 20mil)))
"""


class Rules(unittest.TestCase):
    def test_widest_class_barrier_wins(self):
        self.assertEqual(D.read_barrier(RULES), ("field logic isolation", "Logic*", 0.75))

    def test_other_shapes_are_not_barriers(self):
        self.assertIsNone(D.read_barrier("(version 1)"))
        self.assertIsNone(D.read_barrier(RULES.split('(rule "narrower')[0].replace("field logic", "x")
                                         .replace("B.NetClass != 'Logic*'", "B.NetClass != 'Other'")))
        self.assertEqual(D.read_barrier(RULES.replace("0.75mm", "0.1mm"))[1], "HV")
        self.assertIsNone(D.read_barrier('(rule "x" (severity ignore) '
                                         '(condition "A.NetClass == \'L\' && B.NetClass != \'L\'") '
                                         '(constraint clearance (min 1mm)))'))

    def test_class_matching_is_kicad_like(self):
        self.assertTrue(D.in_class("Logic_Power", "Logic*"))
        self.assertTrue(D.in_class("logic", "Logic*"))
        self.assertTrue(D.in_class("Default,Logic", "Logic*"))
        self.assertFalse(D.in_class("Field_Power", "Logic*"))
        self.assertFalse(D.in_class("", "Logic*"))  # no class is Default


class Rectangles(unittest.TestCase):
    def test_cover_is_exact_on_the_grid(self):
        inside = lambda x, y: x < 5 or y < 2  # an L shape
        rects = D.rects_of(inside, (0, 0, 10, 10))
        self.assertEqual(sorted(rects), [(0, 0, 10, 2), (0, 2, 5, 10)])
        self.assertEqual(D.rects_of(inside, (0, 0, 10, 10), invert=True), [(5, 2, 10, 10)])

    def test_grow(self):
        rects = D.rects_of(lambda x, y: 4 < x < 6 and 4 < y < 6, (0, 0, 10, 10), grow=1)
        self.assertEqual(rects, [(3, 3, 7, 7)])


def _part(key, domain, x=0, y=0, w=2, h=1, fixed=False, bottom=False, nets=("n",)):
    return Component(key=key, ref=key, width=w, height=h, cx=x, cy=y, bottom=bottom, fixed=fixed,
                     pins=[Pin(n, 0.5 * i, 0) for i, n in enumerate(nets)], domain=domain)


class Phases(unittest.TestCase):
    def setUp(self):
        # inner region: the left 10 x 10 of a 40 x 30 board
        comps = [_part("L%d" % i, "in", nets=("l%d" % i, "b")) for i in range(6)]
        comps += [_part("F%d" % i, "out", w=3, h=2, nets=("f%d" % i, "g")) for i in range(20)]
        comps += [_part("B", "bridge", w=4, h=2, nets=("b", "g"))]
        comps += [_part("J", "out", x=35, y=25, w=4, h=4, fixed=True, nets=("g",))]
        self.problem = Problem("t", Region(0, 0, 40, 30), comps,
                               domains=D.Domains("iso", "L*", 0.75, region=[(0, 0, 10, 10)]))

    def test_rotated_fixed_copy(self):
        c = Component(key="X", ref="X", width=4, height=2, cx=0, cy=0, bottom=False, fixed=False,
                      pins=[Pin("n", 1.5, 0)], keepout=[(2, -1, 4, 1)])
        f = D.placed_copy(c, Placement(10, 10, 1))
        self.assertTrue(f.fixed)
        self.assertEqual((f.width, f.height, f.cx, f.cy), (2, 4, 10, 10))
        self.assertEqual((f.pins[0].dx, f.pins[0].dy), (0, -1.5))  # +x turns to -y (up)
        self.assertEqual(f.keepout, [(-1.0, -4.0, 1.0, -2.0)])

    def test_two_phases_keep_the_domains_apart(self):
        calls = []

        def solve(prob):
            # stand-in engine: everything piles up at the board centre, then
            # the legalizer spreads it (it honours the domain blockers)
            calls.append(prob)
            sol = {c.key: Placement(20, 15) for c in prob.components if not c.fixed}
            sol, failed = legalize(prob, sol, grid=0.5)
            self.assertEqual(failed, [])
            return sol

        self.assertTrue(D.enforced(self.problem))
        sol = D.place(self.problem, solve)
        self.assertEqual(len(calls), 2)
        self.assertEqual(set(sol), {c.key for c in self.problem.components if not c.fixed})
        self.assertEqual(overlaps(self.problem, sol), [])
        d = self.problem.domains
        for c in self.problem.components:
            if c.fixed:
                continue
            x0, y0, x1, y1 = sol[c.key].box(c)
            if c.domain in ("in", "bridge"):  # inside region + band
                self.assertLessEqual(x1, 10 + d.band + 1e-6, c.ref)
                self.assertLessEqual(y1, 10 + d.band + 1e-6, c.ref)
            else:  # outside region + band + gap
                self.assertTrue(x0 >= 10 + d.band + d.gap - 1e-6 or y0 >= 10 + d.band + d.gap - 1e-6, c.ref)

    def test_not_enforced_without_a_region(self):
        self.problem.domains.region = []
        self.assertFalse(D.enforced(self.problem))
        self.assertIn("not enforced", D.summary(self.problem))


if __name__ == "__main__":
    unittest.main()
