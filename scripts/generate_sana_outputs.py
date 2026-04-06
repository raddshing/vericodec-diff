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
from vericodec_diff.sana_backbone import (
    deep_update,
    display_path,
    generate_sana_outputs,
    resolve_sana_generation_config,
    write_generation_records_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VeriCodec-Diff prompt-manifest outputs with the frozen Sana stack."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional repo-relative or absolute YAML config file for Sana generation.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives outputs/ for this generation run.",
    )
    parser.add_argument(
        "--prompt-manifest",
        default=None,
        help="Repo-relative or absolute prompt manifest CSV.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Repo-relative or absolute root under which outputs/sana/<suite>/ is created.",
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="Suite name under outputs/sana/<suite>/. Defaults to the manifest stem.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Diffusers model id or local checkpoint directory for SanaPipeline.from_pretrained().",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device used for Sana inference, for example cuda or cpu.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=None,
        help="Torch dtype name for pipeline load, for example bfloat16 or float16.",
    )
    parser.add_argument(
        "--vae-dtype",
        default=None,
        help="Optional dtype override for the VAE module.",
    )
    parser.add_argument(
        "--text-encoder-dtype",
        default=None,
        help="Optional dtype override for the text encoder.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional manifest row limit, useful for smoke runs.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require model loading from local cache only.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overwrite existing sample PNGs and trace files.",
    )
    parser.add_argument(
        "--save-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save late-step decoded traces to <sample_id>__trace.pt when supported.",
    )
    parser.add_argument(
        "--late-step-count",
        type=int,
        default=None,
        help="How many late denoising steps to retain when trace saving is enabled.",
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
    if args.prompt_manifest is not None:
        overrides["paths"]["prompt_manifest"] = args.prompt_manifest
    if args.output_root is not None:
        overrides["paths"]["output_root"] = args.output_root
    if args.suite is not None:
        overrides["paths"]["suite"] = args.suite

    model_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("model_id", "model_id"),
        ("device", "device"),
        ("torch_dtype", "torch_dtype"),
        ("vae_dtype", "vae_dtype"),
        ("text_encoder_dtype", "text_encoder_dtype"),
        ("local_files_only", "local_files_only"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            model_overrides[config_key] = value
    if model_overrides:
        overrides["model"] = model_overrides

    generation_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("guidance_scale", "guidance_scale"),
        ("num_inference_steps", "num_inference_steps"),
        ("limit", "limit"),
        ("overwrite", "overwrite"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            generation_overrides[config_key] = value
    if generation_overrides:
        overrides["generation"] = generation_overrides

    trace_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("save_trace", "save_trace"),
        ("late_step_count", "late_step_count"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            trace_overrides[config_key] = value
    if trace_overrides:
        overrides["trace"] = trace_overrides

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
    config = resolve_sana_generation_config(REPO_ROOT, raw_config)

    repo_root = Path(config["paths"]["repo_root"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    result = generate_sana_outputs(config)

    records_csv_path = output_dir / "generation_records.csv"
    write_generation_records_csv(records_csv_path, result["records"])

    summary = dict(result["summary"])
    summary["generation_records_csv"] = display_path(records_csv_path, repo_root)
    summary_path = output_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"records_csv={records_csv_path}")
    print(f"output_dir={output_dir}")
    print(f"planned_sample_count={summary['planned_sample_count']}")
    print(f"generated_sample_count={summary['generated_sample_count']}")
    print(f"trace_saved_count={summary['trace']['saved_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
