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
from vericodec_diff.patch_metrics import (
    compute_patch_errors,
    deep_update,
    display_path,
    resolve_patch_error_config,
    write_patch_error_records_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 64x64 patchwise reconstruction-error maps for VeriCodec-Diff."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional repo-relative or absolute YAML config file for patch-error computation.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives outputs/ for this patch-error run.",
    )
    parser.add_argument(
        "--records-csv",
        default=None,
        help="CSV describing sample_id, split, source image path, and optionally reconstruction path.",
    )
    parser.add_argument(
        "--reconstruction-root",
        default=None,
        help="Repo-relative or absolute root for DC-AE reconstructions or precomputed recon PNGs.",
    )
    parser.add_argument(
        "--error-map-root",
        default=None,
        help="Repo-relative or absolute root receiving outputs/error_maps/<split>/ files.",
    )
    parser.add_argument(
        "--sample-id-column",
        default=None,
        help="Optional CSV column name used as sample_id.",
    )
    parser.add_argument(
        "--image-path-column",
        default=None,
        help="Optional CSV column name containing source image paths.",
    )
    parser.add_argument(
        "--reconstruction-path-column",
        default=None,
        help="Optional CSV column name containing reconstruction image paths.",
    )
    parser.add_argument(
        "--split-column",
        default=None,
        help="Optional CSV column name containing split names.",
    )
    parser.add_argument(
        "--default-split",
        default=None,
        help="Fallback split when the records CSV does not contain a split column.",
    )
    parser.add_argument(
        "--split-filter",
        default=None,
        help="Optional comma-separated split filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for smoke runs.",
    )
    parser.add_argument(
        "--reconstruction-mode",
        choices=("dc_ae", "precomputed"),
        default=None,
        help="Either compute DC-AE reconstructions on demand or reuse precomputed recon PNGs.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Diffusers model id or local checkpoint directory for AutoencoderDC.from_pretrained().",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for LPIPS and DC-AE inference, for example cuda or cpu.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Torch dtype name for LPIPS/DC-AE backends, for example float32 or bfloat16.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require DC-AE model loading from local cache only.",
    )
    parser.add_argument(
        "--lpips-backend",
        choices=("official", "pixel_l2"),
        default=None,
        help="LPIPS backend. pixel_l2 is a deterministic smoke/debug surrogate, not the locked paper metric.",
    )
    parser.add_argument(
        "--lpips-batch-size",
        type=int,
        default=None,
        help="Batch size for official LPIPS evaluation.",
    )
    parser.add_argument(
        "--wavelet",
        default=None,
        help="Wavelet family used for high-frequency wavelet L1.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing reconstructions, NPZ files, and preview PNGs.",
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
        ("records_csv", "records_csv"),
        ("reconstruction_root", "reconstruction_root"),
        ("error_map_root", "error_map_root"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides["paths"][config_key] = value

    data_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("sample_id_column", "sample_id_column"),
        ("image_path_column", "image_path_column"),
        ("reconstruction_path_column", "reconstruction_path_column"),
        ("split_column", "split_column"),
        ("default_split", "default_split"),
        ("split_filter", "split_filter"),
        ("limit", "limit"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            data_overrides[config_key] = value
    if data_overrides:
        overrides["data"] = data_overrides

    reconstruction_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("reconstruction_mode", "mode"),
        ("model_id", "model_id"),
        ("device", "device"),
        ("torch_dtype", "torch_dtype"),
        ("local_files_only", "local_files_only"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            reconstruction_overrides[config_key] = value
    if reconstruction_overrides:
        overrides["reconstruction"] = reconstruction_overrides

    metrics_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("lpips_backend", "lpips_backend"),
        ("lpips_batch_size", "lpips_batch_size"),
        ("wavelet", "wavelet"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            metrics_overrides[config_key] = value
    if metrics_overrides:
        overrides["metrics"] = metrics_overrides

    if args.overwrite is not None:
        overrides["runtime"] = {"overwrite": args.overwrite}

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
    config = resolve_patch_error_config(REPO_ROOT, raw_config)

    error_map_root = Path(config["paths"]["error_map_root"])
    error_map_root.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    resolved_config_path = error_map_root / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    result = compute_patch_errors(config)

    repo_root = Path(config["paths"]["repo_root"])
    records_csv_path = error_map_root / "patch_error_records.csv"
    write_patch_error_records_csv(records_csv_path, result["records"])

    summary = dict(result["summary"])
    summary["patch_error_records_csv"] = display_path(records_csv_path, repo_root)
    summary_path = error_map_root / "patch_error_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"records_csv={records_csv_path}")
    print(f"error_map_root={error_map_root}")
    print(f"planned_sample_count={summary['planned_sample_count']}")
    print(f"generated_error_map_count={summary['generated_error_map_count']}")
    print(f"generated_reconstruction_count={summary['generated_reconstruction_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
