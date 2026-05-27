"""RoboTwin project package.

`rlinf.projects.robotwin` is the project-facing layer of this repository:

- `rlinf.projects.robotwin.cli` contains the primary training entrypoints.
- `rlinf.projects.robotwin.configs` owns project configs.
- `rlinf.projects.robotwin.integration` bridges project code to vendored `rlinf` and `openpi`.

The upstream-style RLinf core remains under `rlinf`; this package only contains
RoboTwin/pi0.5 project glue code.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
