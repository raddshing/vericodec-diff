from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.config import OmegaConf
from vericodec_diff.environment_check import (
    build_environment_report,
    render_environment_report_markdown,
    resolve_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local VeriCodec-Diff runtime environment.")
    parser.add_argument(
        "--config",
        default="environment/check_environment.yaml",
        help="Repo-relative or absolute YAML config for environment checks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    config = resolve_runtime_config(REPO_ROOT, raw_config)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    OmegaConf.save(resolved_config, output_dir / "resolved_config.yaml")

    report = build_environment_report(config)
    report_json_path = output_dir / "environment_report.json"
    report_md_path = output_dir / "environment_report.md"

    with report_json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    report_md_path.write_text(render_environment_report_markdown(report), encoding="utf-8")

    print(f"resolved_config={output_dir / 'resolved_config.yaml'}")
    print(f"report_json={report_json_path}")
    print(f"report_md={report_md_path}")
    print(f"status={report['summary']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
