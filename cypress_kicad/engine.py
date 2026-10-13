"""Run the Cypress placement engine on a Problem.

The engine needs PyTorch and the compiled DREAMPlace ops, which KiCad's Python
does not have, so it always runs as a separate process: either a local build
(``<cypress_root>/install``) with its own Python, or the Docker image built
from ``docker/Dockerfile``.
"""

import copy
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, fields
from typing import Callable, List, Optional

from .bookshelf import BookshelfDesign
from .model import Problem, Solution

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SETTINGS_PATH = os.path.join(
    os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config"),
    "cypress-kicad", "settings.json")
CONTAINER_ROOT = "/Cypress"


@dataclass
class Settings:
    # how to run the engine
    runner: str = "local"  # "local" or "docker"
    cypress_root: str = REPO_ROOT  # checkout containing install/dreamplace/Placer.py
    python: str = "python3"  # interpreter with torch, for the local runner
    docker_image: str = "cypress:cpu"
    gpu: bool = False
    threads: int = max(1, min(4, (os.cpu_count() or 2) // 2))
    # placement
    iterations: int = 1000
    target_density: float = 0.6
    rotation: bool = True
    net_crossing: bool = True
    seed: int = 1000
    keep_initial: bool = False  # start from current positions instead of the centre
    # board translation
    selected_only: bool = False
    keep_locked: bool = True
    ignore_nets: str = r"(?i)GND|GROUND"
    spacing: float = 0.2  # mm added around every part
    edge_clearance: float = 0.5  # mm kept from the board outline
    units_per_mm: float = 10.0
    # post-processing
    legalize: bool = True
    grid: float = 0.1  # mm

    @classmethod
    def load(cls, path: str = SETTINGS_PATH) -> "Settings":
        s = cls()
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return s
        names = {f.name for f in fields(cls)}
        for k, v in data.items():
            if k in names:
                setattr(s, k, v)
        return s

    def save(self, path: str = SETTINGS_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


class EngineError(RuntimeError):
    pass


def placer_script(settings: Settings) -> str:
    root = CONTAINER_ROOT if settings.runner == "docker" else settings.cypress_root
    return os.path.join(root, "install", "dreamplace", "Placer.py")


def check(settings: Settings) -> Optional[str]:
    """Return a human-readable problem with the engine setup, or None."""
    if settings.runner == "docker":
        if not shutil.which("docker"):
            return "docker is not on PATH"
        r = subprocess.run(["docker", "image", "inspect", settings.docker_image],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            return ("Docker image %r not found. Build it with:\n  docker build -f docker/Dockerfile -t %s ."
                    % (settings.docker_image, settings.docker_image))
        return None
    if settings.runner != "local":
        return "unknown runner %r (use 'local' or 'docker')" % settings.runner
    script = placer_script(settings)
    if not os.path.isfile(script):
        return ("Cypress is not built: %s is missing. Build it (see README) or use the docker runner."
                % script)
    if not shutil.which(settings.python) and not os.path.isfile(settings.python):
        return "Python interpreter %r not found" % settings.python
    return None


def _pow2_bins(extent: float, feature: float) -> int:
    # about two bins across the smallest part, power of two as the DCT needs
    n = 2 ** math.ceil(math.log2(max(1.0, 2 * extent / max(feature, 1e-3))))
    return int(min(512, max(64, n)))


def make_config(problem: Problem, design: BookshelfDesign, settings: Settings,
                aux: str, result_dir: str) -> dict:
    with open(os.path.join(HERE, "default_params.json")) as f:
        cfg = json.load(f)
    movable = problem.movable()
    feature = min(min(c.width, c.height) for c in movable)
    bx = _pow2_bins(problem.region.width, feature)
    by = _pow2_bins(problem.region.height, feature)
    stage = copy.deepcopy(cfg["global_place_stages"][0])
    stage.update(num_bins_x=bx, num_bins_y=by, iteration=int(settings.iterations))
    cfg.update(
        aux_input=aux,
        result_dir=result_dir,
        gpu=int(bool(settings.gpu)),
        num_threads=int(settings.threads),
        num_bins_x=bx,
        num_bins_y=by,
        route_num_bins_x=bx,
        route_num_bins_y=by,
        global_place_stages=[stage],
        target_density=float(settings.target_density),
        enable_rotation=bool(settings.rotation),
        net_crossing_flag=bool(settings.net_crossing),
        random_seed=int(settings.seed),
        random_center_init_flag=0 if settings.keep_initial else 1,
        legalize_flag=0,
        detailed_place_flag=0,
        plot_flag=0,
    )
    return cfg


class Job:
    """One engine run. Drive it with ``poll()`` (GUI) or ``wait()`` (CLI)."""

    def __init__(self, problem: Problem, settings: Settings, workdir: Optional[str] = None):
        self.problem = problem
        self.settings = settings
        self.own_workdir = workdir is None
        self.workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="cypress-kicad-"))
        self.design = BookshelfDesign(problem, settings.units_per_mm)
        self.proc: Optional[subprocess.Popen] = None
        self.lines: "queue.Queue[str]" = queue.Queue()
        self._log_pos = 0

    @property
    def log_file(self):
        return os.path.join(self.workdir, "DREAMPlace.log")

    @property
    def result_pl(self):
        name = self.problem.name
        return os.path.join(self.workdir, "results", name, "%s.gp.pl" % name)

    def command(self, config_path: str) -> List[str]:
        s = self.settings
        if s.runner == "docker":
            cmd = ["docker", "run", "--rm", "-v", "%s:%s" % (self.workdir, self.workdir),
                   "-w", self.workdir, "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/tmp"]
            for k, v in self.omp_env().items():
                cmd += ["-e", "%s=%s" % (k, v)]
            if hasattr(os, "getuid"):
                cmd += ["-u", "%d:%d" % (os.getuid(), os.getgid())]
            if s.gpu:
                cmd += ["--gpus", "all"]
            return cmd + [s.docker_image, "python3", placer_script(s), config_path]
        return [s.python, placer_script(s), config_path]

    def omp_env(self):
        # OpenMP's default spin-waiting makes the tiny per-iteration kernels
        # hundreds of times slower when other processes compete for the CPU
        return {"OMP_NUM_THREADS": str(self.settings.threads), "OMP_WAIT_POLICY": "PASSIVE"}

    def start(self):
        if not self.problem.movable():
            raise EngineError("nothing to place")
        aux = self.design.write(os.path.join(self.workdir, "bookshelf"))
        cfg = make_config(self.problem, self.design, self.settings, aux,
                          os.path.join(self.workdir, "results"))
        config_path = os.path.join(self.workdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        cmd = self.command(config_path)
        self.lines.put("$ " + " ".join(cmd))
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # KiCad's site-packages must not leak in
        env.pop("PYTHONHOME", None)
        env.update(self.omp_env())
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        self.proc = subprocess.Popen(cmd, cwd=self.workdir, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     universal_newlines=True, bufsize=1, **kwargs)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip())

    def _tail_log(self):
        # Placer.py logs to DREAMPlace.log only, so follow the file for progress
        try:
            with open(self.log_file, errors="replace") as f:
                f.seek(self._log_pos)
                chunk = f.read()
                self._log_pos = f.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            self.lines.put(line)

    def poll(self) -> Optional[int]:
        """Collect new output; return the exit code once finished."""
        self._tail_log()
        return self.proc.poll() if self.proc else None

    def drain(self) -> List[str]:
        out = []
        while True:
            try:
                out.append(self.lines.get_nowait())
            except queue.Empty:
                return out

    def cancel(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def wait(self, on_line: Optional[Callable[[str], None]] = None, interval: float = 0.5) -> int:
        while True:
            code = self.poll()
            for line in self.drain():
                if on_line:
                    on_line(line)
            if code is not None:
                self._tail_log()
                for line in self.drain():
                    if on_line:
                        on_line(line)
                return code
            try:
                self.proc.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                pass

    def result(self) -> Solution:
        if self.proc is None or self.proc.returncode != 0:
            raise EngineError("placement engine failed (exit code %s); see %s"
                              % (None if self.proc is None else self.proc.returncode, self.log_file))
        if not os.path.isfile(self.result_pl):
            raise EngineError("placement engine produced no result (%s missing); see %s"
                              % (self.result_pl, self.log_file))
        solution = self.design.read_solution(self.result_pl)
        missing = [c.ref for c in self.problem.movable() if c.key not in solution]
        if missing:
            raise EngineError("result is missing %d parts, e.g. %s" % (len(missing), ", ".join(missing[:5])))
        return solution

    def unconverged_overflow(self) -> Optional[float]:
        """Density overflow left when global placement hit its iteration
        limit before spreading parts out enough, else None."""
        try:
            with open(self.log_file, errors="replace") as f:
                for line in f:
                    m = re.search(r"overflow is significant ([0-9.]+)", line)
                    if m:
                        return float(m.group(1))
        except OSError:
            pass
        return None

    def cleanup(self):
        if self.own_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)
