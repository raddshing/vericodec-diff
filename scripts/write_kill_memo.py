from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.config import OmegaConf
from vericodec_diff.patch_metrics import deep_update
from vericodec_diff.verifier_training import (
    SUPPORTED_BUDGET_METRICS,
    resolve_kill_memo_config,
    write_kill_memo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the week-3 verifier kill memo against the locked threshold gates."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional repo-relative or absolute YAML config file for kill-memo generation.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives outputs/memos/ for this run.",
    )
    parser.add_argument(
        "--eval-json",
        default=None,
        help="Repo-relative or absolute verifier evaluation JSON path.",
    )
    parser.add_argument(
        "--budget-curves-csv",
        default=None,
        help="Repo-relative or absolute verifier budget-curves CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Repo-relative or absolute directory receiving the kill memo and resolved config.",
    )
    parser.add_argument(
        "--memo-md-path",
        default=None,
        help="Repo-relative or absolute Markdown memo output path.",
    )
    parser.add_argument(
        "--memo-json-path",
        default=None,
        help="Repo-relative or absolute JSON memo output path.",
    )
    parser.add_argument(
        "--minimum-best-model-auprc",
        type=float,
        default=None,
        help="Minimum allowed best-model test AUPRC.",
    )
    parser.add_argument(
        "--minimum-best-model-auroc",
        type=float,
        default=None,
        help="Minimum allowed best-model test AUROC.",
    )
    parser.add_argument(
        "--minimum-auprc-lift-over-heuristic",
        type=float,
        default=None,
        help="Minimum allowed AUPRC lift of the best model over the best heuristic baseline.",
    )
    parser.add_argument(
        "--budget-metric-name",
        choices=SUPPORTED_BUDGET_METRICS,
        default=None,
        help="Budget-curve metric evaluated by the kill gate.",
    )
    parser.add_argument(
        "--minimum-budget-recovery",
        default=None,
        help="Comma-separated percent:value thresholds, for example 10:0.5,20:0.75.",
    )
    return parser.parse_args()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must parse to a mapping")
    return data


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {
        "paths": {
            "repo_root": args.repo_root,
        }
    }

    for arg_name, config_key in (
        ("eval_json", "eval_json"),
        ("budget_curves_csv", "budget_curves_csv"),
        ("output_dir", "output_dir"),
        ("memo_md_path", "memo_md_path"),
        ("memo_json_path", "memo_json_path"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides["paths"][config_key] = value

    threshold_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("minimum_best_model_auprc", "minimum_best_model_auprc"),
        ("minimum_best_model_auroc", "minimum_best_model_auroc"),
        ("minimum_auprc_lift_over_heuristic", "minimum_auprc_lift_over_heuristic"),
        ("budget_metric_name", "budget_metric_name"),
        ("minimum_budget_recovery", "minimum_budget_recovery"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            threshold_overrides[config_key] = value
    if threshold_overrides:
        overrides["thresholds"] = threshold_overrides

    return overrides


def main() -> int:
    args = parse_args()
    raw_config: dict[str, object] = {}
    if args.config is not None:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        raw_config = _load_yaml_mapping(config_path)

    raw_config = deep_update(raw_config, _build_cli_overrides(args))
    config = resolve_kill_memo_config(REPO_ROOT, raw_config)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    result = write_kill_memo(config)
    payload = dict(result["payload"])

    print(f"resolved_config={resolved_config_path}")
    print(f"memo_md={result['memo_md_path']}")
    print(f"memo_json={result['memo_json_path']}")
    print(f"decision={payload['decision']}")
    print(f"best_checkpoint={payload['best_model']['checkpoint_filename']}")
    print(f"best_model_test_auprc={payload['best_model']['test_metrics']['auprc']}")
    print(f"best_model_test_auroc={payload['best_model']['test_metrics']['auroc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
