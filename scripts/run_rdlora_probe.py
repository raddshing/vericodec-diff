from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.cells import parse_candidate_ranks_argument
from rd_lora.probe import (
    ProbeValidationError,
    build_run_request,
    default_probe_config,
    dispatch_probe,
    resolve_probe_config,
    write_probe_outputs,
)
from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_probe.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe RD-LoRA-Diff cells with either a real SDXL GPU forward/backward path or an explicit cpu_mock path."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--task", choices=("subject_personalization", "style_domain"), required=True)
    parser.add_argument("--run_mode", choices=("real_gpu", "cpu_mock"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--candidate_ranks", default=None, help="Comma-separated ranks, for example 0,2,4,8,16.")
    parser.add_argument("--probe_inner_steps", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_val_batches", type=int, default=None)
    return parser.parse_args(argv)


def _load_config(config_path: str) -> dict[str, object]:
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    raw = load_yaml_mapping(path) if path.is_file() else {}
    return deep_update(default_probe_config(), raw)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = resolve_probe_config(REPO_ROOT, _load_config(args.config))
    request = build_run_request(
        config=config,
        task=args.task,
        run_mode=args.run_mode,
        output_dir=output_dir,
        max_cells=args.max_cells,
        candidate_ranks=parse_candidate_ranks_argument(args.candidate_ranks) if args.candidate_ranks else None,
        probe_inner_steps=args.probe_inner_steps,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(request), resolved_config_path)

    try:
        payload = dispatch_probe(request)
    except ProbeValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    outputs = write_probe_outputs(
        output_dir,
        payload,
        task=str(request["task"]),
        candidate_ranks=request["probe"]["candidate_ranks"],
    )

    print(f"resolved_config={resolved_config_path}")
    for key, value in outputs.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
