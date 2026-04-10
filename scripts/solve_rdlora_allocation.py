from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.allocator import (
    _solve_layer_only,
    _solve_proposed,
    _solve_timestep_only,
    _solve_uniform,
    build_allocation_manifest,
    build_allocation_schema,
    canonicalize_scored_rows,
)
from rd_lora.cells import DEFAULT_TARGET_MODULES
from rd_lora.runtime.allocation_manifest import write_allocation_manifest
from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping, save_json
from rd_lora.surrogate import load_probe_dataframe, score_records
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_vanilla.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the RD-LoRA rank allocation and emit standardized allocation manifests."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--probe_dir", required=True, help="Probe output directory with cell_utility.csv.")
    parser.add_argument(
        "--surrogate_dir",
        required=True,
        help="Surrogate output directory with surrogate_model.pkl.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for allocation artifacts.")
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "allocation": {
            "rank_budget_total": 96,
            "utility_scale": 1000000,
            "max_exact_state_count": 500000,
            "fallback_mode": "deterministic_greedy",
        },
        "training": {
            "target_modules": list(DEFAULT_TARGET_MODULES),
        },
    }


def _load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    raw = load_yaml_mapping(config_path) if config_path.is_file() else {}
    return deep_update(_default_config(), raw)


def _load_model(surrogate_dir: Path) -> Any:
    with (surrogate_dir / "surrogate_model.pkl").open("rb") as handle:
        return pickle.load(handle)


def _target_modules(config: dict[str, Any]) -> list[str]:
    raw_modules = config.get("training", {}).get("target_modules", DEFAULT_TARGET_MODULES)
    target_modules = [str(module_name) for module_name in raw_modules]
    return target_modules or list(DEFAULT_TARGET_MODULES)


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    config = _load_config(args.config)
    probe_dir = Path(args.probe_dir).expanduser().resolve()
    surrogate_dir = Path(args.surrogate_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = deep_update(
        config,
        {
            "paths": {
                "repo_root": str(REPO_ROOT),
                "probe_dir": str(probe_dir),
                "surrogate_dir": str(surrogate_dir),
                "output_dir": str(output_dir),
            }
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    probe_rows = load_probe_dataframe(probe_dir).to_dict(orient="records")
    model = _load_model(surrogate_dir)
    scored_rows = score_records(model, probe_rows)
    schema = build_allocation_schema(scored_rows)
    canonical_rows = canonicalize_scored_rows(scored_rows, schema=schema)

    allocation_config = resolved_config.get("allocation", {})
    rank_budget_total = int(allocation_config.get("rank_budget_total", allocation_config.get("budget", 96)))
    utility_scale = int(allocation_config.get("utility_scale", 1000000))
    max_exact_state_count = int(allocation_config.get("max_exact_state_count", 500000))
    fallback_mode = str(allocation_config.get("fallback_mode", "deterministic_greedy"))

    cell_ranks_by_backend = {
        "uniform": _solve_uniform(
            canonical_rows,
            schema=schema,
            rank_budget_total=rank_budget_total,
        ),
        "layer_only": _solve_layer_only(
            canonical_rows,
            schema=schema,
            rank_budget_total=rank_budget_total,
            utility_scale=utility_scale,
            max_exact_state_count=max_exact_state_count,
            fallback_mode=fallback_mode,
        ),
        "timestep_only": _solve_timestep_only(
            canonical_rows,
            schema=schema,
            rank_budget_total=rank_budget_total,
            utility_scale=utility_scale,
            max_exact_state_count=max_exact_state_count,
            fallback_mode=fallback_mode,
        ),
        "proposed": _solve_proposed(
            canonical_rows,
            schema=schema,
            rank_budget_total=rank_budget_total,
            utility_scale=utility_scale,
            max_exact_state_count=max_exact_state_count,
            fallback_mode=fallback_mode,
        ),
    }

    target_modules = _target_modules(resolved_config)
    manifest_files: dict[str, str] = {}
    backends = ["uniform", "layer_only", "timestep_only", "proposed"]
    for backend in backends:
        manifest = build_allocation_manifest(
            backend,
            schema=schema,
            rank_budget_total=rank_budget_total,
            cell_ranks=cell_ranks_by_backend[backend],
            target_modules=target_modules,
        )
        path = output_dir / f"{backend}.json"
        write_allocation_manifest(manifest, path)
        manifest_files[backend] = str(path)

    summary_path = output_dir / "allocation_summary.json"
    save_json(
        summary_path,
        {
            "schema_version": "1.0",
            "rank_budget_total": rank_budget_total,
            "manifest_files": manifest_files,
            "backends": backends,
            "wall_time_seconds": round(time.monotonic() - started_at, 6),
        },
    )

    print(f"resolved_config={resolved_config_path}")
    print(f"allocation_summary={summary_path}")
    for backend in backends:
        print(f"{backend}_manifest={output_dir / f'{backend}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
