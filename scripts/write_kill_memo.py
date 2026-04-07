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
    resolve_kill_memo_config,
    write_kill_memo,
)


DEFAULT_KILL_TEST_CONFIG_PATH = "configs/kill_test.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the week-3 verifier kill memo against the locked threshold gates."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_KILL_TEST_CONFIG_PATH,
        help="Repo-relative or absolute YAML config file for kill-memo generation.",
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
        "--error-map-root",
        default=None,
        help="Repo-relative or absolute root containing outputs/error_maps/<split>/ files.",
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
        "--sparsity-metric-name",
        default=None,
        help="Continuous patch metric used for concentration and Gini, for example patch_error.",
    )
    parser.add_argument(
        "--concentration-at-15-min",
        type=float,
        default=None,
        help="Minimum allowed mean failure concentration at the top 15%% of patches.",
    )
    parser.add_argument(
        "--gini-mean-min",
        type=float,
        default=None,
        help="Minimum allowed mean failure-map Gini coefficient.",
    )
    parser.add_argument(
        "--verifier-auprc-min",
        type=float,
        default=None,
        help="Minimum allowed verifier test AUPRC.",
    )
    parser.add_argument(
        "--auprc-multiplier-over-best-heuristic-min",
        type=float,
        default=None,
        help="Minimum allowed verifier AUPRC multiplier over the best heuristic baseline.",
    )
    parser.add_argument(
        "--generation-proxy-concentration-at-15-min",
        type=float,
        default=None,
        help="Minimum allowed generation proxy concentration at the 15%% patch budget.",
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
        ("error_map_root", "error_map_root"),
        ("output_dir", "output_dir"),
        ("memo_md_path", "memo_md_path"),
        ("memo_json_path", "memo_json_path"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides["paths"][config_key] = value

    if args.sparsity_metric_name is not None:
        overrides["metrics"] = {"sparsity_metric_name": args.sparsity_metric_name}

    threshold_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("concentration_at_15_min", "concentration_at_15_min"),
        ("gini_mean_min", "gini_mean_min"),
        ("verifier_auprc_min", "verifier_auprc_min"),
        ("auprc_multiplier_over_best_heuristic_min", "auprc_multiplier_over_best_heuristic_min"),
        ("generation_proxy_concentration_at_15_min", "generation_proxy_concentration_at_15_min"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            threshold_overrides[config_key] = value
    if threshold_overrides:
        overrides["thresholds"] = threshold_overrides

    return overrides


def main() -> int:
    args = parse_args()
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
    print(f"failure_concentration_at_15={payload['measured']['failure_concentration_at_15']}")
    print(f"failure_gini_mean={payload['measured']['failure_gini_mean']}")
    print(f"verifier_test_auprc={payload['measured']['verifier_test_auprc']}")
    print(
        "auprc_multiplier_over_best_heuristic="
        f"{payload['measured']['auprc_multiplier_over_best_heuristic']}"
    )
    print(
        "generation_proxy_concentration_at_15="
        f"{payload['measured']['generation_proxy_concentration_at_15']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
