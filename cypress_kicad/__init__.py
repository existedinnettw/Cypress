"""KiCad front end for the Cypress PCB placer.

The package is split so that only ``board`` touches KiCad's ``pcbnew`` module
and only ``engine`` touches the placement engine (which runs in its own Python
with PyTorch, locally or in Docker). Everything else is plain Python and can be
imported from KiCad's bundled interpreter.
"""

__version__ = "0.1.0"
