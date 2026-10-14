"""End-to-end run of the placement engine on a small board.

Guards engine bugs that only show in a full run: a fixed part crashed
PlaceDB.update_side_info, net crossing on CPU indexed cell locations with pin
ids (heap corruption), and a run stopped at its iteration limit wrote its
input locations back as the result.

Needs pcbnew and a working engine; skipped otherwise. Pick the engine with
CYPRESS_TEST_RUNNER=docker|local (default docker) and, for docker,
CYPRESS_TEST_IMAGE (default cypress:cpu).
"""

import math
import os
import unittest

from cypress_kicad import board as kboard
from cypress_kicad import engine

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BOARD = os.path.join(REPO, "artifacts", "kicad", "small-1.kicad_pcb")

try:
    pcbnew = kboard.import_pcbnew()
except ImportError:
    pcbnew = None


def test_settings():
    s = engine.Settings()
    s.runner = os.environ.get("CYPRESS_TEST_RUNNER", "docker")
    s.docker_image = os.environ.get("CYPRESS_TEST_IMAGE", "cypress:cpu")
    s.threads = 2
    s.iterations = 60  # far from converged on purpose
    s.rotation = True
    s.net_crossing = True
    return s


SETTINGS = test_settings()
SKIP = "pcbnew not importable" if pcbnew is None else engine.check(SETTINGS)


@unittest.skipIf(SKIP, SKIP)
class EngineRun(unittest.TestCase):
    def test_fixed_part_and_unconverged_run(self):
        board = pcbnew.LoadBoard(BOARD)
        locked = max(board.GetFootprints(), key=lambda fp: len(fp.Pads()))
        locked.SetLocked(True)
        problem = kboard.extract(board)
        fixed = [c for c in problem.components if c.fixed]
        self.assertIn(locked.GetReference(), [c.ref for c in fixed])

        job = engine.Job(problem, SETTINGS)
        try:
            job.start()
            code = job.wait()
            log = "\n".join(job.drain()[-20:])
            self.assertEqual(code, 0, "engine failed:\n" + log)
            solution = job.result()
            self.assertIsNotNone(job.unconverged_overflow(), "expected an unconverged run")
        finally:
            job.cleanup()

        # the engine's own result, not the input locations written back
        moved = [math.hypot(solution[c.key].cx - c.cx, solution[c.key].cy - c.cy)
                 for c in problem.movable()]
        self.assertGreater(sum(d > 1.0 for d in moved), len(moved) // 2)
        for c in fixed:  # fixed parts come back where they were (to the 0.1 mm grid)
            p = solution.get(c.key)
            if p is not None:
                self.assertLess(math.hypot(p.cx - c.cx, p.cy - c.cy), 0.15, c.ref)


if __name__ == "__main__":
    unittest.main()
