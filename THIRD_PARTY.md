# Third-Party Code

This repository vendors selected third-party source code to preserve upstream
training implementations while keeping local project code separate.

## RLinf

- Upstream repository: `https://github.com/RLinf/RLinf`
- Upstream package location in this repo: `src/rlinf/`
- Upstream example reference location in this repo: `vendor_examples/rlinf/embodiment/`
- License: Apache License 2.0
- License copy: `THIRD_PARTY_LICENSES/RLinf.LICENSE`

The vendored `rlinf` tree is intentionally trimmed down to a local minimal
subset centered on:

- embodied PPO and SAC training paths
- `openpi` / pi0.5 VLA support
- CNN-based SAC support
- real-world environment hooks, primarily around Franka workflows
- HuggingFace rollout and FSDP-based actor paths

Local project-specific code should continue to live under `src/rlinf/projects/robotwin/`.

## OpenPI

- Upstream repository: `https://github.com/RLinf/openpi`
- Upstream package locations in this repo: `src/openpi/`, `src/openpi_client/`
- License: Apache License 2.0
- Additional notice: Gemma-related upstream notice retained from OpenPI
- License copies:
  - `THIRD_PARTY_LICENSES/openpi.LICENSE`
  - `THIRD_PARTY_LICENSES/openpi.GEMMA.LICENSE`

The vendored `openpi` tree is kept to support the local `pi0.5` path used by
the trimmed RLinf embodied stack, especially:

- `pi0.5` / OpenPI PyTorch policy loading
- normalization, transforms, and dataconfig definitions
- checkpoint download / norm-stat loading helpers
- LeRobot-backed SFT and co-training data loaders
- local `transformers_replace` patches required by the OpenPI PyTorch model
