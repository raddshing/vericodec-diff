# Environment check

Run `python scripts/check_environment.py` from the repo root.

The script reads `environment/check_environment.yaml`, saves `logs/tests/resolved_config.yaml`, and writes:

- `logs/tests/environment_report.json`
- `logs/tests/environment_report.md`

Package checks are derived from the tracked environment YAML files listed in the config after applying the configured exclusions and distribution aliases.

Checkpoint checks are opt-in. Add entries under `checkpoints` with `name`, `path`, and optional `type` (`file`, `directory`, or `any`) when a run should require specific local artifacts.
