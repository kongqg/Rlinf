"""Bentele project package.

`bentele` is the project-facing layer of this repository:

- `bentele.cli` contains the primary training entrypoints.
- `bentele.configs` owns project configs.
- `bentele.integration` bridges project code to vendored `rlinf` and `openpi`.

The older lightweight scaffold modules under `bentele.algorithms`, `bentele.data`,
`bentele.envs`, `bentele.models`, `bentele.runners`, and `bentele.workers`
remain available for interface prototyping and local smoke tests, but they are
no longer the primary embodied RL runtime path.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
