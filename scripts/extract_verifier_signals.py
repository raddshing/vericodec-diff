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
from vericodec_diff.verifier_features import (
    deep_update,
    extract_verifier_features,
    resolve_verifier_feature_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 64x64 patchwise verifier signals for frozen Sana outputs."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional repo-relative or absolute YAML config file for verifier signal extraction.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives data/processed/verifier_features/ outputs.",
    )
    parser.add_argument(
        "--records-csv",
        default=None,
        help="CSV describing sample_id, split, image_path, and optionally trace_path.",
    )
    parser.add_argument(
        "--feature-root",
        default=None,
        help="Repo-relative or absolute root receiving data/processed/verifier_features/<split>/ files.",
    )
    parser.add_argument(
        "--sample-id-column",
        default=None,
        help="Optional CSV column name used as sample_id.",
    )
    parser.add_argument(
        "--image-path-column",
        default=None,
        help="Optional CSV column name containing generated image paths.",
    )
    parser.add_argument(
        "--trace-path-column",
        default=None,
        help="Optional CSV column name containing trace payload paths.",
    )
    parser.add_argument(
        "--trace-status-column",
        default=None,
        help="Optional CSV column name containing trace availability/status labels.",
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
        "--signals",
        default=None,
        help="Optional comma-separated signal list. Defaults to the first-pass image and trace signals.",
    )
    parser.add_argument(
        "--enable-cross-attention-instability",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request the optional cross_attention_instability signal when trace payloads provide it.",
    )
    parser.add_argument(
        "--decode-reencode-backend",
        choices=("dc_ae", "jpeg"),
        default=None,
        help="Backend used for decode_reencode_consistency. jpeg is a deterministic smoke/debug surrogate.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Diffusers model id or local checkpoint directory for AutoencoderDC.from_pretrained().",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for DC-AE decode/reencode inference, for example cuda or cpu.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Torch dtype name for the decode/reencode backend, for example float32 or bfloat16.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require model loading from local cache only for the DC-AE decode/reencode backend.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=None,
        help="JPEG quality for the smoke/debug decode/reencode backend.",
    )
    parser.add_argument(
        "--wavelet",
        default=None,
        help="Wavelet family used by wavelet_instability.",
    )
    parser.add_argument(
        "--max-trace-images",
        type=int,
        default=None,
        help="Optional cap on how many decoded trace frames are loaded per sample.",
    )
    parser.add_argument(
        "--entropy-bins",
        type=int,
        default=None,
        help="Histogram bin count for patch_entropy.",
    )
    parser.add_argument(
        "--allow-missing-optional-signals",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow optional signals such as cross_attention_instability to be recorded as missing instead of failing.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing signal NPZ files and split indexes.",
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
        ("feature_root", "feature_root"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides["paths"][config_key] = value

    data_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("sample_id_column", "sample_id_column"),
        ("image_path_column", "image_path_column"),
        ("trace_path_column", "trace_path_column"),
        ("trace_status_column", "trace_status_column"),
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

    decode_reencode_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("decode_reencode_backend", "backend"),
        ("model_id", "model_id"),
        ("device", "device"),
        ("torch_dtype", "torch_dtype"),
        ("local_files_only", "local_files_only"),
        ("jpeg_quality", "jpeg_quality"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            decode_reencode_overrides[config_key] = value
    if decode_reencode_overrides:
        overrides["decode_reencode"] = decode_reencode_overrides

    trace_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("wavelet", "wavelet"),
        ("max_trace_images", "max_images"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            trace_overrides[config_key] = value
    if trace_overrides:
        overrides["trace"] = trace_overrides

    if args.signals is not None or args.enable_cross_attention_instability is not None:
        signal_overrides: dict[str, object] = {}
        if args.signals is not None:
            signal_overrides["enabled"] = args.signals
        if args.enable_cross_attention_instability is not None:
            signal_overrides["cross_attention_instability"] = args.enable_cross_attention_instability
        overrides["signals"] = signal_overrides

    if args.entropy_bins is not None:
        overrides["heuristics"] = {"entropy_bins": args.entropy_bins}

    runtime_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("allow_missing_optional_signals", "allow_missing_optional_signals"),
        ("overwrite", "overwrite"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            runtime_overrides[config_key] = value
    if runtime_overrides:
        overrides["runtime"] = runtime_overrides

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
    config = resolve_verifier_feature_config(REPO_ROOT, raw_config)

    feature_root = Path(config["paths"]["feature_root"])
    feature_root.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    resolved_config_path = feature_root / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    result = extract_verifier_features(config)

    summary = dict(result["summary"])
    summary_path = feature_root / "verifier_feature_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"feature_root={feature_root}")
    print(f"planned_sample_count={summary['planned_sample_count']}")
    print(f"generated_feature_count={summary['generated_feature_count']}")
    print(f"requested_signals={','.join(summary['requested_signals'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
