import os
import re
import time
import traceback

import pcbnew
import wx

from cypress_kicad import board as kboard
from cypress_kicad import domains
from cypress_kicad import engine
from cypress_kicad.engine import Settings
from cypress_kicad.legalize import legalize
from cypress_kicad.model import overlaps

ICON = os.path.join(os.path.dirname(os.path.realpath(__file__)), "icon.png")

# (attribute, label, kind, tooltip); kind: bool | int | float | str | dir | file | choice:a,b
FIELDS = [
    ("Engine", None, None, None),
    ("runner", "Run engine", "choice:local,docker",
     "local: a Cypress build with its own Python + PyTorch; docker: the image from docker/Dockerfile"),
    ("cypress_root", "Cypress checkout", "dir", "Folder containing install/dreamplace/Placer.py"),
    ("python", "Python with PyTorch", "file", "Interpreter used by the local runner"),
    ("docker_image", "Docker image", "str", "Image used by the docker runner"),
    ("gpu", "Use CUDA GPU", "bool", "Needs a CUDA build of Cypress (or the CUDA image + --gpus)"),
    ("threads", "CPU threads", "int", None),
    ("Placement", None, None, None),
    ("selected_only", "Move selected footprints only", "bool", "Everything else stays fixed"),
    ("keep_locked", "Keep locked footprints fixed", "bool", None),
    ("iterations", "Max iterations", "int", "Global placement stops earlier once spread out"),
    ("target_density", "Target density", "float", "0.3 (spread out) .. 1.0 (packed)"),
    ("rotation", "Allow 90° rotations", "bool", None),
    ("net_crossing", "Minimise ratsnest crossings", "bool", None),
    ("domains", "Keep isolation domains apart", "bool",
     "From a custom rule like A.NetClass == 'Logic*' && B.NetClass != 'Logic*': the smaller domain "
     "stays inside its own partial copper pour, the other keeps the rule's gap (two passes)"),
    ("keep_initial", "Start from current positions", "bool", "Otherwise start from the board centre"),
    ("seed", "Random seed", "int", None),
    ("ignore_nets", "Ignore nets (regex)", "str", "Nets left out of wirelength, typically ground planes"),
    ("spacing", "Part spacing (mm)", "float", "Gap kept around every courtyard"),
    ("edge_clearance", "Board edge clearance (mm)", "float", None),
    ("legalize", "Remove overlaps afterwards", "bool", None),
    ("grid", "Legalization grid (mm)", "float", None),
]


class _Cancelled(Exception):
    pass


class SettingsDialog(wx.Dialog):
    def __init__(self, parent, settings: Settings, n_selected: int):
        super().__init__(parent, title="Cypress Placer",
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.settings = settings
        self.controls = {}
        outer = wx.BoxSizer(wx.VERTICAL)
        grid = None
        for name, label, kind, tip in FIELDS:
            if kind is None:
                box = wx.StaticBoxSizer(wx.VERTICAL, self, name)
                grid = wx.FlexGridSizer(0, 2, 4, 8)
                grid.AddGrowableCol(1)
                box.Add(grid, 1, wx.EXPAND | wx.ALL, 4)
                outer.Add(box, 0, wx.EXPAND | wx.ALL, 6)
                continue
            ctrl = self._make(box.GetStaticBox(), kind, getattr(settings, name))
            if tip:
                ctrl.SetToolTip(tip)
            self.controls[name] = (ctrl, kind)
            grid.Add(wx.StaticText(box.GetStaticBox(), label=label), 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(ctrl, 1, wx.EXPAND)
        sel = self.controls["selected_only"][0]
        if n_selected == 0:
            sel.SetValue(False)
            sel.Disable()
        else:
            sel.SetLabel("(%d selected)" % n_selected)
        outer.Add(self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL), 0, wx.EXPAND | wx.ALL, 8)
        self.FindWindowById(wx.ID_OK).SetLabel("Place")
        self.SetSizerAndFit(outer)
        self.SetMinSize((480, -1))

    def _make(self, parent, kind, value):
        if kind == "bool":
            c = wx.CheckBox(parent)
            c.SetValue(bool(value))
        elif kind.startswith("choice:"):
            c = wx.Choice(parent, choices=kind[7:].split(","))
            c.SetStringSelection(str(value))
        elif kind == "dir":
            c = wx.DirPickerCtrl(parent, path=str(value), style=wx.DIRP_USE_TEXTCTRL)
        elif kind == "file":
            c = wx.FilePickerCtrl(parent, path=str(value), style=wx.FLP_USE_TEXTCTRL | wx.FLP_OPEN)
        else:
            c = wx.TextCtrl(parent, value=str(value))
        return c

    def apply(self):
        """Copy control values into settings; returns an error message or None."""
        for name, (c, kind) in self.controls.items():
            try:
                if kind == "bool":
                    v = c.GetValue()
                elif kind.startswith("choice:"):
                    v = c.GetStringSelection()
                elif kind in ("dir", "file"):
                    v = c.GetPath()
                elif kind == "int":
                    v = int(c.GetValue())
                elif kind == "float":
                    v = float(c.GetValue())
                else:
                    v = c.GetValue()
                    if name == "ignore_nets" and v:
                        re.compile(v)
            except (ValueError, re.error) as e:
                return "%s: %s" % (name, e)
            setattr(self.settings, name, v)
        return None


class CypressPlacerPlugin(pcbnew.ActionPlugin):
    def defaults(self):
        self.name = "Cypress Placer"
        self.category = "Placement"
        self.description = "GPU-accelerated analytic placement of footprints (Cypress / DREAMPlace)"
        self.show_toolbar_button = True
        self.icon_file_name = ICON
        self.dark_icon_file_name = ICON

    def Run(self):
        board = pcbnew.GetBoard()
        parent = wx.FindWindowByName("PcbFrame")
        settings = Settings.load()
        n_selected = kboard.selected_count(board)
        settings.selected_only = settings.selected_only and n_selected > 0

        dlg = SettingsDialog(parent, settings, n_selected)
        try:
            while True:
                if dlg.ShowModal() != wx.ID_OK:
                    return
                err = dlg.apply() or engine.check(settings)
                if not err:
                    break
                wx.MessageBox(err, "Cypress Placer", wx.OK | wx.ICON_ERROR, dlg)
        finally:
            dlg.Destroy()
        settings.save()

        try:
            self._place(parent, board, settings)
        except Exception as e:
            traceback.print_exc()
            wx.MessageBox("Placement failed:\n%s" % e, "Cypress Placer", wx.OK | wx.ICON_ERROR, parent)

    def _place(self, parent, board, settings):
        problem = kboard.extract(
            board, selected_only=settings.selected_only, ignore_nets=settings.ignore_nets,
            spacing=settings.spacing, edge_clearance=settings.edge_clearance,
            keep_locked=settings.keep_locked, domains=settings.domains)
        two_pass = domains.enforced(problem)
        log = []
        jobs = []
        failed = []
        overflow = None
        progress = wx.ProgressDialog(
            "Cypress Placer", "Placing %d footprints..." % len(problem.movable()),
            maximum=max(1, settings.iterations), parent=parent,
            style=wx.PD_CAN_ABORT | wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_AUTO_HIDE)

        def solve(prob):
            """Global placement and legalization of one problem; one pass of
            two when isolation domains are enforced."""
            nonlocal overflow
            job = engine.Job(prob, settings)
            jobs.append(job)
            label = ("Pass %d of 2: " % len(jobs)) if two_pass else ""
            job.start()
            done = 0
            while True:
                code = job.poll()
                for line in job.drain():
                    log.append(line)
                    m = re.search(r"iteration\s+(\d+).*?HPWL\s+(\S+)", line)
                    if m:
                        done = min(int(m.group(1)), settings.iterations - 1)
                        progress.Update(done, "%sIteration %s, HPWL %s" % (label, m.group(1), m.group(2)))
                if code is not None:
                    break
                keep_going, _ = progress.Update(done)
                if not keep_going:
                    job.cancel()
                    raise _Cancelled()
                time.sleep(0.1)
            try:
                sol = job.result()
            except engine.EngineError as e:
                tail = "\n".join(log[-15:])
                raise engine.EngineError("%s\n\n%s" % (e, tail))
            if job.unconverged_overflow() is not None:
                overflow = job.unconverged_overflow()
            if settings.legalize:
                sol, f = legalize(prob, sol, settings.grid)
                failed.extend(f)
            return sol

        try:
            solution = domains.place(problem, solve) if two_pass else solve(problem)
            progress.Update(settings.iterations - 1, "Applying placement...")
            moved = kboard.apply(board, problem, solution)
        except _Cancelled:
            return
        finally:
            progress.Destroy()
            for job in jobs:
                job.cancel()
                job.cleanup()
        pcbnew.Refresh()

        remaining = overlaps(problem, solution)
        msg = "Moved %d footprints." % moved
        if problem.domains:
            msg += "\n" + domains.summary(problem)
        if overflow is not None:
            msg += ("\nGlobal placement stopped at the iteration limit before parts were "
                    "spread out (overflow %.3f): raise Max iterations or lower Target density."
                    % overflow)
        if failed:
            msg += "\nNo free spot for: %s" % ", ".join(failed)
        if remaining:
            msg += "\n%d overlapping pairs involving moved parts remain." % len(remaining)
        msg += "\nUse Edit > Undo to revert."
        wx.MessageBox(msg, "Cypress Placer", wx.OK | wx.ICON_INFORMATION, parent)
