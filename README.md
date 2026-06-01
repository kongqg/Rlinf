# bentele

`bentele` is the project-facing package for this repository.

The repository now has a deliberate three-layer split:

- `src/bentele/`: project layer, entrypoints, project configs, integration glue
- `src/rlinf/`: vendored RL engine layer
- `src/openpi/` and `src/openpi_client/`: vendored pi0.5 model and runtime layer

Inside `bentele`, there are still two kinds of code:

- the new primary path:
  - `bentele.cli`
  - `bentele.configs`
  - `bentele.integration`
- the older lightweight local scaffold:
  - `bentele.algorithms`
  - `bentele.data`
  - `bentele.envs`
  - `bentele.models`
  - `bentele.runners`
  - `bentele.scheduler`
  - `bentele.workers`

That split is intentional: `bentele` owns project orchestration, while `rlinf`
and `openpi` stay vendored as reusable engine packages.

The repository still avoids copying the entire upstream surface:

- Ray cluster scheduling
- multi-node placement backends
- Megatron / vLLM / SGLang infrastructure
- most benchmark-specific task implementations

At the same time, the repository now vendors:

- a trimmed upstream `RLinf` subset under `src/rlinf/`
- the `RLinf/openpi` runtime needed for `pi0.5` loading under `src/openpi/`
- the matching `openpi_client` helpers under `src/openpi_client/`

This keeps the original PPO, SAC, and `pi0.5` weight-loading paths available
locally without dragging in the full upstream project surface. See `THIRD_PARTY.md`.

The current local vendor is intentionally narrowed to a single simulation path:

- `RoboTwin place_phone_stand`
- `pi0.5 / openpi`
- PPO training through `bentele -> rlinf`

## Layout

```text
src/bentele/
  cli/
  configs/
  integration/
  algorithms/        # older local scaffold
  data/              # older local scaffold
  envs/              # older local scaffold
  models/            # older local scaffold
  runners/           # older local scaffold
  scheduler/         # older local scaffold
  utils/             # older local scaffold
  workers/           # older local scaffold

src/rlinf/
src/openpi/
src/openpi_client/

scripts/
  compute_robotwin_norm_stats.py
  trace_robotwin_pipeline.py
```

## Current goal

This scaffold is meant to be the clean base for:

- validating whether pi0.5 weights can be loaded cleanly
- running a shared VLA baseline with staged prompts
- wiring in RL before task-specific MDP choices are finalized
- later attaching real robot adapters and task definitions

## Quick smoke test

```bash
cd /home/kqg/bentele
PYTHONPATH=src python3 scripts/smoke_scaffold.py
```

## OpenPI Setup

```bash
cd /home/kqg/bentele
pip install -r requirements.txt
python scripts/patch_transformers_for_openpi.py
```

If you need robot-node extras such as camera or spacemouse support, install
`requirements-realworld.txt` on that machine instead of the base requirements
file.

## RoboTwin Phone Task

```bash
cd /home/kqg/bentele
python scripts/patch_transformers_for_openpi.py
export OPENPI_MODEL_PATH=/abs/path/to/pi05_base_torch
python -m bentele.cli.train_embodied \
  --config-name robotwin_place_phone_stand_ppo_openpi_pi05 \
  env.train.assets_path=/abs/path/to/RoboTwin \
  env.eval.assets_path=/abs/path/to/RoboTwin
```

The repo is intentionally trimmed around this one path so the active code is:

- `src/bentele/configs/embodiment/robotwin_place_phone_stand_ppo_openpi_pi05.yaml`
- `src/bentele/configs/embodiment/env/robotwin_place_phone_stand.yaml`
- `src/rlinf/envs/robotwin/`
- `src/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py`

## Next steps

- define project-specific observation and action schemas
- attach the real robot adapter
- add task-specific reward, reset, and evaluation logic
