from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.substrate.diffusers_sdxl import (
    build_cli_overrides,
    build_diffusers_sdxl_lora_command,
    deep_update,
    display_path,
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
        description="Prepare or execute the RD-LoRA SDXL LoRA smoke run through the official diffusers script."
    )
    parser.add_argument(
        "--config",
        default="configs/rdlora_vanilla.yaml",
        help="Repo-relative or absolute YAML config for the RD-LoRA SDXL wrapper.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Task id from configs/rdlora_tasks.yaml. Defaults to smoke.task_id from the wrapper config.",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values with dotted key=value pairs. Repeat as needed.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch the generated accelerate command after preparing the smoke run directory.",
    )
    return parser.parse_args()


def _load_wrapper_config(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    raw_config = load_yaml_mapping(config_path)
    cli_overrides: dict[str, object] = {}
    if args.repo_root is not None:
        cli_overrides["paths"] = {"repo_root": args.repo_root}
    raw_config = deep_update(raw_config, cli_overrides)
    raw_config = deep_update(raw_config, build_cli_overrides(args.set_values))
    return resolve_vanilla_lora_config(REPO_ROOT, raw_config)


def _probe_torch_cuda() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "available": False,
            "device_count": 0,
            "cuda_version": None,
            "device_name": None,
            "error": str(exc),
        }

    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if available else 0
    device_name = torch.cuda.get_device_name(0) if available and device_count > 0 else None
    return {
        "available": available,
        "device_count": device_count,
        "cuda_version": torch.version.cuda,
        "device_name": device_name,
        "error": None,
    }


def _find_training_processes(*, own_pid: int, parent_pid: int) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    matches: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        pid = int(parts[0])
        ppid = int(parts[1])
        command = parts[2]
        if pid in {own_pid, parent_pid}:
            continue
        if "train_text_to_image_lora_sdxl.py" not in command:
            continue
        matches.append({"pid": pid, "ppid": ppid, "command": command})
    return matches


def _kill_processes(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    killed: list[dict[str, Any]] = []
    for entry in processes:
        pid = int(entry["pid"])
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append({**entry, "signal": "SIGTERM"})
        except ProcessLookupError:
            continue

    deadline = time.time() + 5.0
    remaining = [int(entry["pid"]) for entry in processes]
    while remaining and time.time() < deadline:
        alive: list[int] = []
        for pid in remaining:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            break
        remaining = alive
        time.sleep(0.2)

    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append({"pid": pid, "ppid": None, "command": None, "signal": "SIGKILL"})
        except ProcessLookupError:
            continue
    return killed


def _query_gpu_memory_mib(gpu_index: int) -> int | None:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    for raw_line in result.stdout.splitlines():
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


def _write_gpu_memory_trace(path: Path, samples: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["elapsed_seconds", "memory_used_mib"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample)


def _list_output_files(output_dir: Path) -> list[str]:
    files = [path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()]
    return sorted(files)


def _snapshot_output_files(output_dir: Path) -> dict[str, int]:
    return {
        path.relative_to(output_dir).as_posix(): path.stat().st_mtime_ns
        for path in output_dir.rglob("*")
        if path.is_file()
    }


def _changed_output_files(output_dir: Path, before: dict[str, int]) -> list[str]:
    changed: list[str] = []
    after = _snapshot_output_files(output_dir)
    for relative_path, mtime_ns in sorted(after.items()):
        if before.get(relative_path) != mtime_ns:
            changed.append(relative_path)
    return changed


def main() -> int:
    args = parse_args()
    config = _load_wrapper_config(args)
    task_id = args.task_id or str(config["smoke"]["task_id"])

    smoke_root = Path(config["paths"]["smoke_root"]) / task_id / str(config["smoke"]["run_name"])
    smoke_root.mkdir(parents=True, exist_ok=True)
    preexisting_files = _snapshot_output_files(smoke_root)
    resolved_config_path = smoke_root / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    tasks_bundle = load_rdlora_tasks(config["paths"]["tasks_config"])
    prepared = prepare_pilot_imagefolder(config, tasks_bundle, task_id)
    plan = build_diffusers_sdxl_lora_command(
        config,
        task_id=task_id,
        run_name=str(config["smoke"]["run_name"]),
        smoke=True,
    )
    repo_root = Path(config["paths"]["repo_root"])
    validation = validate_launch_plan(plan, repo_root=repo_root)

    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_script_path = output_dir / "manual_gpu_command.sh"
    train_log_path = output_dir / "train.log"
    gpu_trace_path = output_dir / "gpu_memory_trace.csv"
    write_shell_command(launch_script_path, plan["command"])

    issues: list[str] = []
    killed_processes: list[dict[str, Any]] = []
    return_code: int | None = None
    baseline_vram_mib = _query_gpu_memory_mib(0)
    peak_vram_mib = baseline_vram_mib
    samples: list[dict[str, float | int]] = []
    torch_cuda = _probe_torch_cuda() if args.execute else None

    if args.execute and not bool(torch_cuda["available"]):
        issues.append("torch.cuda.is_available() returned False in the active environment.")

    if args.execute:
        existing_processes = _find_training_processes(own_pid=os.getpid(), parent_pid=os.getppid())
        killed_processes = _kill_processes(existing_processes)

    if args.execute and not issues:
        monitor_stop = threading.Event()
        monitor_start = time.monotonic()

        def _monitor_gpu_memory() -> None:
            while not monitor_stop.is_set():
                memory_used = _query_gpu_memory_mib(0)
                if memory_used is not None:
                    samples.append(
                        {
                            "elapsed_seconds": round(time.monotonic() - monitor_start, 3),
                            "memory_used_mib": memory_used,
                        }
                    )
                monitor_stop.wait(0.5)

        monitor_thread = threading.Thread(target=_monitor_gpu_memory, name="gpu-memory-monitor", daemon=True)
        monitor_thread.start()
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        with train_log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                plan["command"],
                cwd=repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = int(completed.returncode)
        monitor_stop.set()
        monitor_thread.join(timeout=2.0)
        if samples:
            peak_vram_mib = max(int(sample["memory_used_mib"]) for sample in samples)
            _write_gpu_memory_trace(gpu_trace_path, samples)
        else:
            issues.append("GPU memory trace was empty; could not verify VRAM usage.")
        if return_code != 0:
            issues.append(f"accelerate launch exited with code {return_code}. See train.log for details.")

    completed_successfully = bool(args.execute and not issues and return_code == 0)
    summary_payload = {
        "task_id": task_id,
        "prepared_image_count": prepared["image_count"],
        "wrapped_official_script": validation["wrapped_official_script"],
        "official_script": validation["official_script"],
        "train_data_dir": validation["train_data_dir"],
        "output_dir": validation["output_dir"],
        "resolved_config": display_path(resolved_config_path, repo_root),
        "manual_gpu_command_script": display_path(launch_script_path, repo_root),
        "manual_gpu_command": plan["manual_command"],
        "accelerate_found": validation["accelerate_found"],
        "accelerate_config_file": config["accelerate"]["launch"]["config_file"],
        "execute": bool(args.execute),
        "completed_successfully": completed_successfully,
        "return_code": return_code,
        "torch_cuda": torch_cuda,
        "killed_processes": killed_processes,
        "baseline_vram_mib": baseline_vram_mib,
        "peak_vram_mib": peak_vram_mib,
        "gpu_memory_trace": (
            display_path(gpu_trace_path, repo_root) if args.execute and gpu_trace_path.is_file() else None
        ),
        "train_log": display_path(train_log_path, repo_root) if args.execute and train_log_path.is_file() else None,
        "output_files": [],
        "issues": issues,
    }
    summary_path = smoke_root / "smoke_summary.json"
    save_json(summary_path, summary_payload)
    summary_payload["output_files"] = _changed_output_files(output_dir, preexisting_files)
    save_json(summary_path, summary_payload)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"manual_gpu_command={plan['manual_command']}")
    print(f"wrapped_official_script={int(validation['wrapped_official_script'])}")
    print(f"prepared_image_count={prepared['image_count']}")
    if args.execute:
        print(f"train_log={train_log_path}")
        print(f"gpu_memory_trace={gpu_trace_path if gpu_trace_path.is_file() else 'not_written'}")
        print(f"peak_vram_mib={peak_vram_mib if peak_vram_mib is not None else 'unknown'}")
        print(f"completed_successfully={int(completed_successfully)}")
    return 0 if not args.execute or completed_successfully else int(return_code or 1)


if __name__ == "__main__":
    raise SystemExit(main())
