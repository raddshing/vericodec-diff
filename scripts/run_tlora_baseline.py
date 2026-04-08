from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.baselines import (
    build_cli_overrides,
    build_tlora_command,
    deep_update,
    display_path,
    load_yaml_mapping,
    resolve_tlora_config,
    save_json,
    validate_tlora_launch_plan,
    verify_registered_baseline,
    write_shell_command,
)
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap the official T-LoRA SDXL training entrypoint without reimplementing it."
    )
    parser.add_argument(
        "--config",
        default="configs/tlora_baseline.yaml",
        help="Repo-relative or absolute YAML config for the T-LoRA wrapper.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Single-token run name under outputs/baselines/t_lora/.",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values with dotted key=value pairs. Repeat as needed.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch the generated accelerate command.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    raw_config = load_yaml_mapping(config_path)
    cli_overrides: dict[str, object] = {}
    if args.repo_root is not None:
        cli_overrides["paths"] = {"repo_root": args.repo_root}
    raw_config = deep_update(raw_config, cli_overrides)
    raw_config = deep_update(raw_config, build_cli_overrides(args.set_values))
    return resolve_tlora_config(REPO_ROOT, raw_config)


def main() -> int:
    args = parse_args()
    config = _load_config(args)
    plan = build_tlora_command(config, run_name=args.run_name)
    repo_root = Path(config["paths"]["repo_root"])
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    registry_validation = verify_registered_baseline(config)
    if not registry_validation["ok"]:
        raise SystemExit("\n".join(registry_validation["issues"]))

    launch_validation = validate_tlora_launch_plan(plan, repo_root=repo_root)
    launch_script_path = output_dir / "launch_command.sh"
    write_shell_command(launch_script_path, plan["command"])

    summary_payload = {
        "baseline_id": config["baseline"]["id"],
        "critical_path": bool(config["baseline"]["critical_path"]),
        "execute": bool(args.execute),
        "official_repo": registry_validation["official_repo"],
        "pinned_commit": registry_validation["pinned_commit"],
        "local_checkout_path": registry_validation["local_checkout_path"],
        "registry_path": registry_validation["registry_path"],
        "resolved_config": display_path(resolved_config_path, repo_root),
        "launch_script": display_path(launch_script_path, repo_root),
        "manual_command": plan["manual_command"],
        "registry_validation": registry_validation,
        "launch_validation": launch_validation,
        "wandb_api_key_env": config["run"]["wandb_api_key_env"],
        "wandb_api_key_present": bool(
            config["run"]["pass_wandb_api_key"] and os.environ.get(str(config["run"]["wandb_api_key_env"]))
        ),
    }
    summary_path = output_dir / "launch_summary.json"
    save_json(summary_path, summary_payload)

    command = list(plan["command"])
    if config["run"]["pass_wandb_api_key"]:
        api_key = os.environ.get(str(config["run"]["wandb_api_key_env"]))
        if api_key:
            command.extend(["--wandb_api_key", api_key])

    if args.execute:
        subprocess.run(command, cwd=repo_root, check=True)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"launch_script={launch_script_path}")
    print(f"manual_command={plan['manual_command']}")
    print(f"official_repo={registry_validation['official_repo']}")
    print(f"pinned_commit={registry_validation['pinned_commit']}")
    print(f"critical_path={int(config['baseline']['critical_path'])}")
    print(f"execute={int(args.execute)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
