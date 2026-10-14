"""Command line: place footprints of a .kicad_pcb with Cypress.

  python -m cypress_kicad place board.kicad_pcb -o placed.kicad_pcb
  python -m cypress_kicad export board.kicad_pcb -d bookshelf_dir
  python -m cypress_kicad check

Must run with a Python that can import KiCad's pcbnew module (KiCad's own
Python, or set CYPRESS_PCBNEW_PATH). Options not given on the command line come
from the settings saved by the KiCad plugin dialog.
"""

import argparse
import os
import sys
import time
from dataclasses import fields

from . import engine
from .engine import Settings


def _add_settings_args(p):
    d = Settings.load()
    g = p.add_argument_group("engine")
    g.add_argument("--runner", choices=["local", "docker"], default=d.runner)
    g.add_argument("--cypress-root", default=d.cypress_root, help="Cypress checkout with install/ (local runner)")
    g.add_argument("--python", default=d.python, help="Python with torch for the local runner")
    g.add_argument("--docker-image", default=d.docker_image)
    g.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=d.gpu)
    g.add_argument("--threads", type=int, default=d.threads)
    g = p.add_argument_group("placement")
    g.add_argument("--iterations", type=int, default=d.iterations)
    g.add_argument("--target-density", type=float, default=d.target_density)
    g.add_argument("--rotation", action=argparse.BooleanOptionalAction, default=d.rotation,
                   help="let the placer rotate parts in 90 degree steps")
    g.add_argument("--net-crossing", action=argparse.BooleanOptionalAction, default=d.net_crossing,
                   help="penalise crossing ratsnest lines")
    g.add_argument("--seed", type=int, default=d.seed)
    g.add_argument("--keep-initial", action=argparse.BooleanOptionalAction, default=d.keep_initial,
                   help="start from the current positions instead of the board centre")
    g = p.add_argument_group("board")
    g.add_argument("--keep-locked", action=argparse.BooleanOptionalAction, default=d.keep_locked,
                   help="treat locked footprints as fixed")
    g.add_argument("--ignore-nets", default=d.ignore_nets, help="regex of nets left out of wirelength")
    g.add_argument("--spacing", type=float, default=d.spacing, help="mm added around every part")
    g.add_argument("--edge-clearance", type=float, default=d.edge_clearance, help="mm kept from the outline")
    g.add_argument("--units-per-mm", type=float, default=d.units_per_mm)
    g.add_argument("--legalize", action=argparse.BooleanOptionalAction, default=d.legalize,
                   help="remove remaining overlaps after global placement")
    g.add_argument("--grid", type=float, default=d.grid, help="legalization grid (mm)")
    g = p.add_argument_group("engine parameters")
    g.add_argument("--param", dest="params", action="append", type=_param, default=None,
                   metavar="KEY=VALUE",
                   help="override a default_params.json key; VALUE is JSON (repeatable); "
                        "merged over the parameters saved in the settings")


def _param(text):
    import json
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("expected KEY=VALUE, got %r" % text)
    try:
        return key, json.loads(value)
    except ValueError:
        return key, value


def _settings(args) -> Settings:
    s = Settings()
    for f in fields(Settings):
        if hasattr(args, f.name) and f.name != "params":
            setattr(s, f.name, getattr(args, f.name))
    # --param adds to the saved parameters instead of replacing them
    s.params = dict(Settings.load().params or {})
    s.params.update(getattr(args, "params", None) or [])
    return s


def _load(path, settings, refs=None):
    from .board import extract, import_pcbnew
    pcbnew = import_pcbnew()
    board = pcbnew.LoadBoard(path)
    if refs:
        wanted = set(refs)
        for fp in board.GetFootprints():
            if fp.GetReference() in wanted:
                fp.SetSelected()
            else:
                fp.ClearSelected()
    problem = extract(board, selected_only=bool(refs), ignore_nets=settings.ignore_nets,
                      spacing=settings.spacing, edge_clearance=settings.edge_clearance,
                      keep_locked=settings.keep_locked)
    return board, problem


def cmd_place(args):
    from .board import apply
    from .legalize import legalize
    from .model import overlaps

    settings = _settings(args)
    problem_msg = engine.check(settings)
    if problem_msg:
        sys.exit("error: " + problem_msg)
    board, problem = _load(args.board, settings, args.refs)
    print("%s: %d parts (%d movable), region %.1f x %.1f mm" % (
        problem.name, len(problem.components), len(problem.movable()),
        problem.region.width, problem.region.height))

    job = engine.Job(problem, settings, args.workdir)
    t0 = time.time()
    job.start()
    code = job.wait(on_line=print if args.verbose else _progress)
    print()
    try:
        solution = job.result()
    except engine.EngineError as e:
        sys.exit("error: %s" % e)
    print("global placement finished in %.1f s" % (time.time() - t0))
    overflow = job.unconverged_overflow()
    if overflow is not None:
        print("warning: not fully spread after %d iterations (overflow %.3f); "
              "raise --iterations or lower --target-density" % (settings.iterations, overflow))

    if settings.legalize:
        solution, failed = legalize(problem, solution, settings.grid)
        if failed:
            print("warning: could not find a free spot for %s" % ", ".join(failed))
    left = overlaps(problem, solution)
    print("overlapping pairs involving moved parts: %d" % len(left))

    moved = apply(board, problem, solution)
    out = args.output or args.board
    board.Save(out)
    print("moved %d footprints, wrote %s" % (moved, out))
    if args.workdir is None:
        job.cleanup()
    else:
        print("engine files kept in", job.workdir)
    return code


def _progress(line):
    if "iteration" in line and "HPWL" in line:
        msg = line.split(" - ", 1)[-1].strip()
        sys.stdout.write("\r" + msg[:110].ljust(110))
        sys.stdout.flush()
    elif "Error" in line or "Traceback" in line or "error" in line:
        print("\n" + line)


def cmd_export(args):
    from .bookshelf import BookshelfDesign
    settings = _settings(args)
    _, problem = _load(args.board, settings, args.refs)
    aux = BookshelfDesign(problem, settings.units_per_mm).write(args.dir)
    print("wrote", aux)


def cmd_check(args):
    msg = engine.check(_settings(args))
    print(msg or "engine OK")
    return 1 if msg else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="cypress_kicad", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("place", help="place footprints and save the board")
    sp.add_argument("board")
    sp.add_argument("-o", "--output", help="output .kicad_pcb (default: overwrite input)")
    sp.add_argument("--refs", nargs="+", help="only move these references; everything else stays fixed")
    sp.add_argument("--workdir", help="keep engine files here instead of a temp dir")
    sp.add_argument("-v", "--verbose", action="store_true", help="print the full engine log")
    _add_settings_args(sp)
    sp.set_defaults(func=cmd_place)

    sp = sub.add_parser("export", help="write the board as Bookshelf files only")
    sp.add_argument("board")
    sp.add_argument("-d", "--dir", required=True)
    sp.add_argument("--refs", nargs="+")
    _add_settings_args(sp)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("check", help="check that the placement engine can run")
    _add_settings_args(sp)
    sp.set_defaults(func=cmd_check)

    args = p.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
