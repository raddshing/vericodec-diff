from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.baselines import (
    build_cli_overrides,
    build_intlora_command,
    deep_update,
    display_path,
    load_yaml_mapping,
    resolve_intlora_config,
    save_json,
    validate_intlora_launch_plan,
    verify_registered_baseline,
    write_shell_command,
)
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap the official IntLoRA SD1.5 appendix training entrypoint without porting it to SDXL."
    )
    parser.add_argument(
        "--config",
        default="configs/intlora_sd15.yaml",
        help="Repo-relative or absolute YAML config for the IntLoRA appendix wrapper.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Single-token run name under outputs/baselines/int_lora/.",
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
        help="Actually launch the generated IntLoRA command.",
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
    return resolve_intlora_config(REPO_ROOT, raw_config)


def main() -> int:
    args = parse_args()
    config = _load_config(args)
    plan = build_intlora_command(config, run_name=args.run_name)
    repo_root = Path(config["paths"]["repo_root"])
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    registry_validation = verify_registered_baseline(config)
    if not registry_validation["ok"]:
        raise SystemExit("\n".join(registry_validation["issues"]))

    launch_validation = validate_intlora_launch_plan(plan, repo_root=repo_root)
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
    }
    summary_path = output_dir / "launch_summary.json"
    save_json(summary_path, summary_payload)

    if args.execute:
        subprocess.run(plan["command"], cwd=repo_root, check=True)

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
