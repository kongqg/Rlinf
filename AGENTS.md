# RLinf RoboTwin Multi-Agent Guide

Use structured collaboration by default for non-trivial work in this repository.

## Objective

Build a clean VLA and RL scaffold around pi0.5 with:

- shared policy baseline support
- staged task and prompt interfaces
- SFT and RL-ready training entrypoints
- real-robot adapter boundaries
- replay, logging, and evaluation hooks

## Default Roles

- `planner`
- `explorer`
- `innovation_scout`
- `event_pit_scout`
- `ensemble_scout`
- `implementer`
- `critic`

Use one writer and multiple read-only roles unless there is a concrete reason to do otherwise.

## Working Rules

- Keep interfaces and data contracts clear before writing task-specific logic.
- Prefer minimal relevant file reads over broad codebase scans.
- Prefer stronger abstractions, cleaner data flow, or clearer execution boundaries over tiny constant tweaks.
- Keep policy-facing signals separate from debug-only signals.
- Do not leak privileged state, labels, or evaluation-only metadata into policy observations by accident.
- Do not refactor unrelated files.
- Validate every code change before summarizing.

## Validation Commands

Run these commands after code changes unless the task explicitly does not touch code:

```bash
cd /home/kqg/bentele
python3 -m compileall src examples scripts
```

Append repository-specific smoke tests here as runnable entrypoints are added.
