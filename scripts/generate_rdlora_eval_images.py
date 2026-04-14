from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval.generate import generate_eval_images_for_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate held-out evaluation images for one RD-LoRA run.")
    parser.add_argument("--run_dir", required=True, help="Path to a single training run directory.")
    parser.add_argument("--manifest", required=True, help="Path to the evaluation prompt manifest CSV.")
    parser.add_argument("--output_dir", required=True, help="Directory for PNGs and generation_records.csv.")
    parser.add_argument(
        "--base_model_id",
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Base SDXL model id used to load the eval pipeline.",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of denoising steps used for every generated sample.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for inference.")
    parser.add_argument("--torch_dtype", default="float16", help="Torch dtype name, for example float16.")
    return parser.parse_args()


def _write_resolved_config(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config.yaml"
    _write_resolved_config(
        resolved_config_path,
        {
            "base_model_id": args.base_model_id,
            "device": args.device,
            "guidance_scale": args.guidance_scale,
            "manifest": str(manifest_path),
            "num_inference_steps": args.num_inference_steps,
            "output_dir": str(output_dir),
            "run_dir": str(run_dir),
            "torch_dtype": args.torch_dtype,
        },
    )

    try:
        records_path = generate_eval_images_for_run(
            run_dir=run_dir,
            prompt_manifest_path=manifest_path,
            output_dir=output_dir,
            base_model_id=args.base_model_id,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(records_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

