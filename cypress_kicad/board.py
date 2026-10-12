"""Read a placement Problem from a KiCad board and write a Solution back.

Works on a live board inside the PCB editor (action plugin) and on a board
loaded with ``pcbnew.LoadBoard`` (CLI). Requires KiCad 7 or newer.
"""

import glob
import os
import re
import shutil
import sys
from typing import Optional

from .model import Component, Pin, Problem, Region, Solution


def import_pcbnew():
    """Import KiCad's pcbnew module, searching common install locations."""
    try:
        import pcbnew  # noqa: F401
        return pcbnew
    except ImportError:
        pass
    for path in _pcbnew_candidates():
        if os.path.isfile(os.path.join(path, "pcbnew.py")) and path not in sys.path:
            sys.path.append(path)
            try:
                import pcbnew  # noqa: F401
                return pcbnew
            except ImportError:
                sys.path.remove(path)
    raise ImportError(
        "Cannot import KiCad's pcbnew module. Run with the Python that ships with "
        "KiCad, or point CYPRESS_PCBNEW_PATH at the directory containing pcbnew.py.")


def _pcbnew_candidates():
    env = os.environ.get("CYPRESS_PCBNEW_PATH")
    if env:
        yield env
    ver = "python%d.%d" % sys.version_info[:2]
    yield "/usr/lib/python3/dist-packages"
    yield "/usr/lib/%s/site-packages" % ver
    yield "/usr/local/lib/%s/site-packages" % ver
    yield "/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/lib/%s/site-packages" % ver
    # Nix and other wrapper-based installs export PYTHONPATH from a shell wrapper
    for tool in ("pcbnew", "kicad"):
        exe = shutil.which(tool)
        if not exe:
            continue
        exe = os.path.realpath(exe)
        yield from glob.glob(os.path.join(os.path.dirname(os.path.dirname(exe)), "lib", ver, "site-packages"))
        try:
            with open(exe, errors="ignore") as f:
                head = f.read(65536)
        except OSError:
            continue
        for m in re.finditer(r"PYTHONPATH='([^']*)'", head):
            yield from (p for p in m.group(1).split(":") if "kicad" in p)


def _mm(pcbnew, nm):
    return pcbnew.ToMM(nm)


def _nm(pcbnew, mm):
    return pcbnew.FromMM(mm)


def footprint_key(fp):
    return fp.m_Uuid.AsString()


def _placement_box(pcbnew, fp):
    """Axis-aligned box used for placement: courtyard, else body without text."""
    layer = pcbnew.B_CrtYd if fp.IsFlipped() else pcbnew.F_CrtYd
    try:
        fp.BuildCourtyardCaches()
    except AttributeError:
        pass
    try:
        courtyard = fp.GetCourtyard(layer)
        if courtyard.OutlineCount() > 0:
            return courtyard.BBox()
    except AttributeError:
        pass
    return fp.GetBoundingBox(False)


def extract(board, name: Optional[str] = None, selected_only: bool = False,
            ignore_nets: str = r"(?i)GND|GROUND", spacing: float = 0.0,
            edge_clearance: float = 0.0, keep_locked: bool = True) -> Problem:
    """Build the placement problem for ``board``.

    selected_only   only selected footprints move; all others are fixed
    ignore_nets     regex; matching nets are left out of wirelength (ground)
    spacing         extra gap (mm) kept between component boxes
    edge_clearance  gap (mm) kept between components and the board outline
    keep_locked     locked footprints are fixed obstacles

    Footprints without any (non-ignored) net are always fixed.
    """
    import pcbnew

    ignore = re.compile(ignore_nets) if ignore_nets else None
    comps = []
    for fp in board.GetFootprints():
        pads = list(fp.Pads())
        if not pads:
            continue  # logos and other artwork take no part in placement
        box = _placement_box(pcbnew, fp)
        fixed = (keep_locked and fp.IsLocked()) or (selected_only and not fp.IsSelected())
        cx, cy = _mm(pcbnew, box.GetCenter().x), _mm(pcbnew, box.GetCenter().y)
        comp = Component(
            key=footprint_key(fp),
            ref=fp.GetReference(),
            width=_mm(pcbnew, box.GetWidth()) + spacing,
            height=_mm(pcbnew, box.GetHeight()) + spacing,
            cx=cx,
            cy=cy,
            bottom=bool(fp.IsFlipped()),
            fixed=bool(fixed),
            through_hole=any(p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH for p in pads),
        )
        for pad in pads:
            net = pad.GetNetname()
            if pad.GetNetCode() <= 0 or not net or (ignore and ignore.search(net)):
                continue
            pos = pad.GetPosition()
            comp.pins.append(Pin(net, _mm(pcbnew, pos.x) - cx, _mm(pcbnew, pos.y) - cy))
        if not comp.pins:
            # mounting holes, fiducials and the like: nothing pulls them
            # anywhere, so keep them as obstacles where the designer put them
            comp.fixed = True
        comps.append(comp)

    if not comps:
        raise ValueError("The board has no footprints with pads")
    if not any(not c.fixed for c in comps):
        raise ValueError("Nothing to place: every footprint is fixed (locked or not selected)")

    edges = board.GetBoardEdgesBoundingBox()
    if edges.GetWidth() > 0 and edges.GetHeight() > 0:
        region = Region(_mm(pcbnew, edges.GetLeft()) + edge_clearance,
                        _mm(pcbnew, edges.GetTop()) + edge_clearance,
                        _mm(pcbnew, edges.GetRight()) - edge_clearance,
                        _mm(pcbnew, edges.GetBottom()) - edge_clearance)
    else:
        # no outline yet: use the current spread of parts plus a margin
        x0 = min(c.cx - c.width / 2 for c in comps)
        y0 = min(c.cy - c.height / 2 for c in comps)
        x1 = max(c.cx + c.width / 2 for c in comps)
        y1 = max(c.cy + c.height / 2 for c in comps)
        mx, my = 0.1 * (x1 - x0) + 1, 0.1 * (y1 - y0) + 1
        region = Region(x0 - mx, y0 - my, x1 + mx, y1 + my)
    if region.width <= 0 or region.height <= 0:
        raise ValueError("Placement region is empty; check the board outline and edge clearance")

    if not name:
        name = os.path.splitext(os.path.basename(board.GetFileName() or "board"))[0]
    name = re.sub(r"[^A-Za-z0-9_]", "_", name) or "board"
    return Problem(name=name, region=region, components=comps)


def apply(board, problem: Problem, solution: Solution) -> int:
    """Move and rotate footprints to ``solution``; returns how many moved."""
    import pcbnew

    by_key = {footprint_key(fp): fp for fp in board.GetFootprints()}
    moved = 0
    for comp in problem.components:
        place = solution.get(comp.key)
        fp = by_key.get(comp.key)
        if comp.fixed or place is None or fp is None:
            continue
        centre = pcbnew.VECTOR2I(_nm(pcbnew, comp.cx), _nm(pcbnew, comp.cy))
        if place.quarter_turns % 4:
            # rotating about the box centre keeps the box centre in place
            fp.Rotate(centre, pcbnew.EDA_ANGLE(90.0 * (place.quarter_turns % 4), pcbnew.DEGREES_T))
        fp.Move(pcbnew.VECTOR2I(_nm(pcbnew, place.cx - comp.cx), _nm(pcbnew, place.cy - comp.cy)))
        moved += 1
    return moved


def selected_count(board) -> int:
    return sum(1 for fp in board.GetFootprints() if fp.IsSelected())

