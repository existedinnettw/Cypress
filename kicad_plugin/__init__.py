"""KiCad action plugin: Tools > External Plugins > Cypress Placer."""

import os
import sys

# The plugin directory is either a symlink to kicad_plugin/ in the Cypress
# checkout or a copy that bundles cypress_kicad/ next to this file.
_here = os.path.dirname(os.path.realpath(__file__))
for _path in (_here, os.path.dirname(_here)):
    if os.path.isdir(os.path.join(_path, "cypress_kicad")):
        if _path not in sys.path:
            sys.path.insert(0, _path)
        break

try:
    from .plugin import CypressPlacerPlugin

    CypressPlacerPlugin().register()
except Exception:  # never break KiCad's plugin loading
    import traceback

    traceback.print_exc()
