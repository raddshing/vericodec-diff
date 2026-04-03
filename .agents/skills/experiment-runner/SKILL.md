# Skill: experiment-runner

Use this skill when adding or modifying an experiment.

## Workflow
1. Read the relevant experiment config and the project AGENTS.md.
2. State the exact experiment scope in one sentence.
3. Identify inputs, outputs, and success criteria before coding.
4. Implement the smallest end-to-end path that can run on one sample.
5. Add or update a smoke test.
6. Save metrics as JSON or CSV with deterministic names.
7. Save `resolved_config.yaml` in the run directory.
8. Write a short memo if the implementation differs from the spec.

## Guardrails
- Do not silently change patch size, resolution, or metric weights.
- Do not introduce notebooks into the critical path.
- Do not add a heavy dependency unless it solves a specific blocker.
