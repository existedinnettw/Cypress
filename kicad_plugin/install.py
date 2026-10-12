#!/usr/bin/env python3
"""Install the Cypress Placer action plugin into KiCad's user plugin folder.

  python3 kicad_plugin/install.py              # symlink (default; tracks this checkout)
  python3 kicad_plugin/install.py --copy       # copy files (Windows without symlink rights)
  python3 kicad_plugin/install.py --uninstall
  python3 kicad_plugin/install.py --kicad 9.0  # pick a KiCad version folder

Restart KiCad's PCB editor (or Tools > External Plugins > Refresh Plugins) after
installing.
"""

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NAME = "cypress_placer"
sys.path.insert(0, REPO)


def kicad_roots():
    home = os.path.expanduser("~")
    if sys.platform.startswith("linux"):
        base = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
        return [os.path.join(base, "kicad")]
    docs = os.path.join(home, "Documents", "KiCad")
    return [docs]


def find_targets(version):
    dirs = []
    for root in kicad_roots():
        versions = [version] if version else sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(root, "[0-9]*.[0-9]*")))
        dirs += [os.path.join(root, v, "scripting", "plugins") for v in versions]
    return dirs


def remove(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kicad", help="KiCad version folder, e.g. 9.0 (default: all found)")
    p.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    p.add_argument("--uninstall", action="store_true")
    args = p.parse_args()

    targets = find_targets(args.kicad)
    if not targets:
        sys.exit("No KiCad user folder found; run KiCad once or pass --kicad <version>")
    for plugins in targets:
        dest = os.path.join(plugins, NAME)
        if os.path.lexists(dest):
            remove(dest)
        if args.uninstall:
            print("removed", dest)
            continue
        os.makedirs(plugins, exist_ok=True)
        if args.copy:
            ignore = shutil.ignore_patterns("__pycache__", "install.py")
            shutil.copytree(HERE, dest, ignore=ignore)
            shutil.copytree(os.path.join(REPO, "cypress_kicad"), os.path.join(dest, "cypress_kicad"), ignore=ignore)
        else:
            os.symlink(HERE, dest, target_is_directory=True)
        print("installed", dest)

    if not args.uninstall:
        from cypress_kicad.engine import SETTINGS_PATH, Settings
        settings = Settings.load()
        if not os.path.exists(SETTINGS_PATH):
            settings.cypress_root = REPO
            if not os.path.isfile(os.path.join(REPO, "install", "dreamplace", "Placer.py")):
                settings.runner = "docker"  # no local build yet
            settings.save()
            print("wrote default settings to", SETTINGS_PATH)
        print("Restart the PCB editor; the plugin is under Tools > External Plugins > Cypress Placer.")


if __name__ == "__main__":
    main()
