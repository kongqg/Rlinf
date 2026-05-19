---
name: implement-feature
description: Use when this repository needs structured collaboration across explorer, innovation_scout, event_pit_scout, ensemble_scout, planner, implementer, and critic.
---

# Implement Feature

Use this skill when the task needs structured multi-agent collaboration rather than a single straight-line edit.

Start by reading:

- `../../AGENTS.md`
- `../../agents/planner.md`
- `../../agents/explorer.md`
- `../../agents/innovation_scout.md`
- `../../agents/event_pit_scout.md`
- `../../agents/ensemble_scout.md`
- `../../agents/implementer.md`
- `../../agents/critic.md`

Keep the skill lightweight. Use only the roles that materially improve the current task, but default to the full flow for non-trivial changes.

## Workflow

1. Ask `explorer` to map the minimal relevant files, current implementation path, task interfaces, data sources, configs, and experiment outputs.
2. Ask `innovation_scout` to propose multiple next-step candidate families instead of only small parameter nudges.
3. Ask `event_pit_scout` to identify high-value temporal, contact, reset, or stage-boundary opportunities.
4. Ask `ensemble_scout` to identify multi-signal, multi-camera, state-plus-vision, or teacher-policy combination opportunities.
5. Ask `planner` to rank those routes by expected upside, real-world feasibility, implementation cost, interface risk, and data leakage risk, then choose the next batch.
6. Ask `implementer` to implement only the chosen batch with minimal real code changes.
7. Ask `critic` to review leakage risk, overfitting risk, interface drift, and missing validation.
8. Run validation commands from `../../AGENTS.md`.
9. Summarize:
   changed files
   what was tried
   why the chosen batch was selected
   validation results
   remaining risks
   what should be explored next

## Constraints

- Prefer stronger interfaces, better task information, better data collection routes, or cleaner execution logic over tiny constant changes.
- Do not refactor unrelated files.
- Default to one writer and multiple read-only agents.
- Do not stop at the first plausible route if better alternatives have not been screened.
- Protect holdout tasks, scenes, seeds, prompts, and instructions from unnecessary reuse.
- Keep policy observations separate from debug-only signals.
- When touching data collection, replay, export, prompt, env, or deployment code, explicitly check interface compatibility.
- If the target is still unmet, continue with the next highest-priority batch unless there is a concrete blocker.
