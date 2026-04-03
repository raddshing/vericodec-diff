# Skill: repro-audit

Use this skill before milestone freezes and before paper-table generation.

## Checklist
- Confirm the correct conda env is active.
- Confirm the git commit hash is recorded.
- Confirm seed handling is explicit.
- Confirm `resolved_config.yaml` exists for each run.
- Confirm outputs live in the expected deterministic path.
- Confirm third-party checkpoints are referenced, not copied into git.
- Confirm metrics JSONs are complete and parseable.
- Confirm any unavailable baseline is marked honestly in the registry.

## Output
Write a concise audit note to `outputs/memos/repro_audit_<date>.md`.
