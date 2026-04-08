from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.baselines import build_tlora_command, resolve_tlora_config, validate_tlora_launch_plan
from rd_lora.stage_d import (
    DEFAULT_STAGE_D_CONFIG_PATH,
    TRAINING_SUMMARY_FILENAME,
    build_method_allocations,
    collect_preflight_report,
    resolve_stage_d_config,
)
from rd_lora.substrate.diffusers_sdxl import (
    build_cli_overrides,
    build_diffusers_sdxl_lora_command,
    deep_update,
    load_rdlora_tasks,
    load_yaml_mapping,
    prepare_pilot_imagefolder,
    resolve_vanilla_lora_config,
    save_json,
    validate_launch_plan,
    write_shell_command,
)
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or execute Stage D pilot training plans for the RD-LoRA week-2 gate."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_STAGE_D_CONFIG_PATH,
        help="Repo-relative or absolute YAML config for the Stage D pilot harness.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Single-token run name under outputs/rd_lora/stage_d/.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch supported training backends after writing launch plans.",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values with dotted key=value pairs. Repeat as needed.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    raw_config = load_yaml_mapping(config_path)
    cli_overrides: dict[str, Any] = {}
    if args.repo_root is not None:
        cli_overrides.setdefault("paths", {})["repo_root"] = args.repo_root
    if args.run_name is not None:
        cli_overrides.setdefault("run", {})["name"] = args.run_name
    if args.execute:
        cli_overrides.setdefault("training", {})["execute"] = True
    raw_config = deep_update(raw_config, cli_overrides)
    raw_config = deep_update(raw_config, build_cli_overrides(args.set_values))
    return resolve_stage_d_config(REPO_ROOT, raw_config)


def _load_vanilla_base_config(config: dict[str, Any]) -> dict[str, Any]:
    raw_vanilla = load_yaml_mapping(Path(config["paths"]["vanilla_config"]))
    overrides = {
        "paths": {
            "repo_root": config["paths"]["repo_root"],
            "tasks_config": config["paths"]["tasks_config"],
            "accelerate_config": config["paths"]["accelerate_config"],
        },
        "training": {
            "gradient_checkpointing": bool(config["training"]["gradient_checkpointing"]),
            "mixed_precision": str(config["training"]["mixed_precision"]),
        },
    }
    return deep_update(raw_vanilla, overrides)


def _uniform_rank(method_allocations: dict[str, dict[str, Any]]) -> int:
    selections = list(method_allocations["uniform"]["selections"])
    ranks = sorted({int(selection["candidate_rank"]) for selection in selections})
    if len(ranks) != 1:
        raise SystemExit(f"Uniform allocation must resolve to a single shared rank, found: {ranks}")
    return int(ranks[0])


def _list_checkpoint_files(output_dir: Path) -> list[str]:
    return sorted(path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file())


def _query_gpu_memory_mib(gpu_index: int) -> int | None:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    for raw_line in completed.stdout.splitlines():
        parts = [item.strip() for item in raw_line.split(",")]
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
            memory_used = int(parts[1])
        except ValueError:
            continue
        if index == gpu_index:
            return memory_used
    return None


def _execute_command(
    command: list[str],
    *,
    cwd: Path,
    output_dir: Path,
    gpu_index: int,
) -> dict[str, Any]:
    baseline_vram_mib = _query_gpu_memory_mib(gpu_index)
    peak_vram_mib = baseline_vram_mib
    started_at = time.monotonic()
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=None,
        )
        while process.poll() is None:
            current = _query_gpu_memory_mib(gpu_index)
            if current is not None:
                peak_vram_mib = max(peak_vram_mib or current, current)
            time.sleep(1.0)
    duration_seconds = round(time.monotonic() - started_at, 3)
    checkpoint_files = _list_checkpoint_files(output_dir)
    return {
        "return_code": int(process.returncode or 0),
        "baseline_vram_mib": baseline_vram_mib,
        "peak_vram_mib": peak_vram_mib,
        "wall_time_seconds": duration_seconds,
        "train_log": str(log_path),
        "checkpoint_files": checkpoint_files,
    }


def main() -> int:
    args = parse_args()
    config = _load_config(args)
    repo_root = Path(config["paths"]["repo_root"])
    training_dir = Path(config["paths"]["training_dir"])
    training_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = training_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    preflight = collect_preflight_report(config)
    method_allocations = build_method_allocations(config)
    uniform_rank = _uniform_rank(method_allocations)

    raw_vanilla = _load_vanilla_base_config(config)
    vanilla_tasks_bundle = load_rdlora_tasks(config["paths"]["tasks_config"])

    task_runs: list[dict[str, Any]] = []
    issues: list[str] = []
    artifact_inventory = {
        str(resolved_config_path.relative_to(repo_root)),
    }

    for task_id in config["pilot"]["task_ids"]:
        prepare_pilot_imagefolder(
            resolve_vanilla_lora_config(REPO_ROOT, raw_vanilla),
            vanilla_tasks_bundle,
            task_id,
        )

    for task_id in config["pilot"]["task_ids"]:
        for method in config["methods"]["enabled"]:
            method_root = training_dir / method / task_id
            method_root.mkdir(parents=True, exist_ok=True)
            allocation_spec_path = method_root / "allocation_spec.json"
            save_json(allocation_spec_path, method_allocations[method])
            artifact_inventory.add(str(allocation_spec_path.relative_to(repo_root)))

            summary: dict[str, Any] = {
                "task_id": task_id,
                "method": method,
                "execute_requested": bool(config["training"]["execute"]),
                "preflight_ok": bool(preflight["ok"]),
                "allocation_spec": str(allocation_spec_path.relative_to(repo_root)),
                "training_backend": None,
                "status": "planned",
                "issues": [],
                "manual_command": None,
                "launch_script": None,
                "output_dir": str(method_root.relative_to(repo_root)),
                "checkpoint_files": [],
                "checkpoint_exists": False,
                "used_gpu": None,
                "peak_vram_mib": None,
                "wall_time_seconds": None,
            }

            if method == "uniform":
                summary["training_backend"] = "diffusers_sdxl_lora"
                raw_uniform = deep_update(
                    raw_vanilla,
                    {
                        "paths": {
                            "run_root": str((training_dir / method).relative_to(repo_root)),
                        },
                        "run": {
                            "name": config["run"]["name"],
                        },
                        "training": {
                            "rank": uniform_rank,
                            "gradient_checkpointing": bool(config["training"]["gradient_checkpointing"]),
                            "mixed_precision": str(config["training"]["mixed_precision"]),
                        },
                    },
                )
                uniform_config = resolve_vanilla_lora_config(REPO_ROOT, raw_uniform)
                plan = build_diffusers_sdxl_lora_command(
                    uniform_config,
                    task_id=task_id,
                    run_name=config["run"]["name"],
                    smoke=False,
                )
                output_dir = Path(plan["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                resolved_uniform_config_path = output_dir / "resolved_config.yaml"
                OmegaConf.save(OmegaConf.create(uniform_config), resolved_uniform_config_path)
                launch_script_path = output_dir / "launch_command.sh"
                write_shell_command(launch_script_path, plan["command"])
                launch_validation = validate_launch_plan(plan, repo_root=repo_root)
                summary.update(
                    {
                        "manual_command": plan["manual_command"],
                        "launch_script": str(launch_script_path.relative_to(repo_root)),
                        "output_dir": str(output_dir.relative_to(repo_root)),
                        "launch_validation": launch_validation,
                    }
                )
                artifact_inventory.add(str(resolved_uniform_config_path.relative_to(repo_root)))
                artifact_inventory.add(str(launch_script_path.relative_to(repo_root)))
                if config["training"]["execute"]:
                    if not preflight["ok"]:
                        summary["status"] = "blocked_preflight"
                        summary["issues"] = list(preflight["issues"])
                    else:
                        execution = _execute_command(
                            list(plan["command"]),
                            cwd=repo_root,
                            output_dir=output_dir,
                            gpu_index=int(config["preflight"]["required_gpu_index"]),
                        )
                        summary["checkpoint_files"] = execution["checkpoint_files"]
                        summary["checkpoint_exists"] = any(
                            path.endswith("pytorch_lora_weights.safetensors") for path in execution["checkpoint_files"]
                        )
                        summary["used_gpu"] = execution["peak_vram_mib"] is not None
                        summary["peak_vram_mib"] = execution["peak_vram_mib"]
                        summary["wall_time_seconds"] = execution["wall_time_seconds"]
                        if Path(execution["train_log"]).is_file():
                            artifact_inventory.add(str(Path(execution["train_log"]).relative_to(repo_root)))
                        if execution["return_code"] != 0:
                            summary["status"] = "failed"
                            summary["issues"].append(
                                f"uniform training exited with code {execution['return_code']}"
                            )
                        elif not summary["checkpoint_exists"]:
                            summary["status"] = "failed"
                            summary["issues"].append(
                                "uniform training completed without a LoRA checkpoint artifact"
                            )
                        else:
                            summary["status"] = "completed"
                task_runs.append(summary)
                continue

            if method == "t_lora":
                summary["training_backend"] = "t_lora"
                if task_id != "t_lora_dog_subject":
                    summary["status"] = "skipped"
                    summary["issues"].append("T-LoRA is only wired for the local dog subject task.")
                    task_runs.append(summary)
                    continue
                raw_tlora = load_yaml_mapping(Path(config["paths"]["tlora_config"]))
                raw_tlora = deep_update(
                    raw_tlora,
                    {
                        "paths": {
                            "repo_root": config["paths"]["repo_root"],
                            "output_root": str((training_dir / method).relative_to(repo_root)),
                        },
                        "run": {
                            "name": config["run"]["name"],
                        },
                        "training": {
                            "mixed_precision": str(config["training"]["mixed_precision"]),
                        },
                    },
                )
                tlora_config = resolve_tlora_config(REPO_ROOT, raw_tlora)
                tlora_plan = build_tlora_command(tlora_config, run_name=config["run"]["name"])
                launch_script_path = method_root / "launch_command.sh"
                write_shell_command(launch_script_path, tlora_plan["command"])
                summary.update(
                    {
                        "manual_command": tlora_plan["manual_command"],
                        "launch_script": str(launch_script_path.relative_to(repo_root)),
                        "output_dir": str(Path(tlora_plan["output_dir"]).relative_to(repo_root)),
                        "launch_validation": validate_tlora_launch_plan(tlora_plan, repo_root=repo_root),
                        "status": "blocked_budget_mapping",
                    }
                )
                summary["issues"].append(
                    "The repo-local T-LoRA wrapper does not expose an exact matched-budget mapping for the Stage D cell budget."
                )
                artifact_inventory.add(str(launch_script_path.relative_to(repo_root)))
                task_runs.append(summary)
                continue

            summary["status"] = "blocked_backend_missing"
            summary["issues"].append(
                f"{method} training backend is not implemented in the repo-local Stage D harness."
            )
            task_runs.append(summary)

    completed_uniform_runs = [
        run for run in task_runs if run["method"] == "uniform" and run["status"] == "completed"
    ]
    probe_allocation_overhead_ratio = None
    if completed_uniform_runs:
        probe_summary_path = Path(config["paths"]["probe_run_dir"]) / "probe_summary.json"
        surrogate_summary_path = (
            Path(config["paths"]["allocator_run_dir"]) / "surrogate" / "surrogate_training_summary.json"
        )
        probe_summary = load_yaml_mapping(probe_summary_path) if probe_summary_path.suffix in {".yaml", ".yml"} else None
        if probe_summary is None:
            try:
                probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
            except Exception:
                probe_summary = None
        try:
            surrogate_summary = json.loads(surrogate_summary_path.read_text(encoding="utf-8"))
        except Exception:
            surrogate_summary = None
        overhead_terms = []
        for payload in (probe_summary, surrogate_summary):
            if isinstance(payload, dict) and payload.get("wall_time_seconds") not in (None, ""):
                overhead_terms.append(float(payload["wall_time_seconds"]))
        uniform_wall = sum(float(run["wall_time_seconds"]) for run in completed_uniform_runs if run["wall_time_seconds"])
        if overhead_terms and uniform_wall > 0.0:
            probe_allocation_overhead_ratio = round(sum(overhead_terms) / uniform_wall, 6)

    training_summary = {
        "schema_version": 1,
        "resolved_config": str(resolved_config_path.relative_to(repo_root)),
        "preflight": preflight,
        "execute_requested": bool(config["training"]["execute"]),
        "uniform_rank": uniform_rank,
        "probe_allocation_overhead_ratio": probe_allocation_overhead_ratio,
        "task_runs": task_runs,
        "issues": issues,
        "artifact_inventory": sorted(artifact_inventory),
    }
    training_summary_path = Path(config["paths"]["training_summary_json"])
    save_json(training_summary_path, training_summary)

    print(f"resolved_config={resolved_config_path}")
    print(f"training_summary={training_summary_path}")
    print(f"preflight_ok={int(preflight['ok'])}")
    print(f"uniform_rank={uniform_rank}")
    print(f"task_run_count={len(task_runs)}")
    print(f"completed_run_count={sum(1 for item in task_runs if item['status'] == 'completed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
