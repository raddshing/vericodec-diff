from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_SCRIPT = REPO_ROOT / "scripts" / "run_rdlora_probe.py"
SURROGATE_SCRIPT = REPO_ROOT / "scripts" / "train_rdlora_surrogate.py"
ALLOCATE_SCRIPT = REPO_ROOT / "scripts" / "solve_rdlora_allocation.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_rdlora.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_rdlora.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "write_rdlora_gate_memo.py"


def _write_fake_runtime(tmp_path: Path) -> Path:
    runtime_dir = tmp_path / "fake_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "torch.py").write_text(
        "\n".join(
            [
                "__version__ = '2.5.0'",
                "",
                "class version:",
                "    cuda = '12.1'",
                "",
                "class _Cuda:",
                "    def is_available(self): return True",
                "    def device_count(self): return 1",
                "    def get_device_name(self, index): return f'Fake GPU {index}'",
                "    def max_memory_allocated(self): return 805306368",
                "",
                "cuda = _Cuda()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "diffusers.py").write_text("__version__ = '0.0.test'\n", encoding="utf-8")
    return runtime_dir


def _write_train_stub(tmp_path: Path) -> Path:
    path = tmp_path / "train_stub.py"
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import argparse",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser(add_help=False)",
                "parser.add_argument('--output_dir', required=True)",
                "args, _ = parser.parse_known_args()",
                "output_dir = Path(args.output_dir)",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "(output_dir / 'pytorch_lora_weights.safetensors').write_text('stub', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_config(tmp_path: Path, train_stub: Path) -> Path:
    accelerate_config = tmp_path / "single_gpu_fp16.yaml"
    accelerate_config.write_text("compute_environment: LOCAL_MACHINE\n", encoding="utf-8")
    config_path = tmp_path / "rdlora_test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  repo_root: .",
                f"  official_diffusers_script: {train_stub}",
                f"  accelerate_config: {accelerate_config}",
                "  expected_diffusers_substring: fake_runtime",
                "launcher:",
                "  kind: python",
                f"  executable: {sys.executable}",
                "  num_processes: 1",
                "training:",
                "  use_rslora: true",
                "  target_modules:",
                "    - to_k",
                "    - to_q",
                "    - to_v",
                "    - to_out.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_stage_outputs_match_required_schemas(tmp_path: Path) -> None:
    runtime_dir = _write_fake_runtime(tmp_path)
    train_stub = _write_train_stub(tmp_path)
    config_path = _write_config(tmp_path, train_stub)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime_dir)

    probe_dir = tmp_path / "probe_real"
    surrogate_dir = tmp_path / "surrogate_real"
    allocation_dir = tmp_path / "allocation_real"
    evaluation_dir = tmp_path / "evaluation_real"
    gate_dir = tmp_path / "gate_real"

    subprocess.run(
        [
            sys.executable,
            str(PROBE_SCRIPT),
            "--config",
            str(config_path),
            "--task",
            "subject_personalization",
            "--output_dir",
            str(probe_dir),
            "--run_mode",
            "real_gpu",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(SURROGATE_SCRIPT),
            "--config",
            str(config_path),
            "--probe_dir",
            str(probe_dir),
            "--output_dir",
            str(surrogate_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ALLOCATE_SCRIPT),
            "--config",
            str(config_path),
            "--probe_dir",
            str(probe_dir),
            "--surrogate_dir",
            str(surrogate_dir),
            "--output_dir",
            str(allocation_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    run_dirs = []
    for backend in ("uniform", "layer_only", "timestep_only", "proposed"):
        run_dir = tmp_path / f"{backend}_run"
        run_dirs.append(run_dir)
        subprocess.run(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(config_path),
                "--task",
                "subject_personalization",
                "--backend",
                backend,
                "--allocation",
                str(allocation_dir / f"{backend}.json"),
                "--run_mode",
                "real_gpu",
                "--output_dir",
                str(run_dir),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    subprocess.run(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            "--run_dir",
            *[str(run_dir) for run_dir in run_dirs],
            "--output_dir",
            str(evaluation_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--probe_dir",
            str(probe_dir),
            "--surrogate_dir",
            str(surrogate_dir),
            "--allocation_dir",
            str(allocation_dir),
            "--evaluation_dir",
            str(evaluation_dir),
            "--output_dir",
            str(gate_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    probe_summary = json.loads((probe_dir / "probe_summary.json").read_text(encoding="utf-8"))
    assert {"task", "cell_count", "candidate_ranks", "row_count"} <= set(probe_summary)
    assert (probe_dir / "run_provenance.json").is_file()
    assert (probe_dir / "cell_utility.csv").is_file()
    assert (probe_dir / "cell_utility.json").is_file()

    surrogate_summary = json.loads((surrogate_dir / "surrogate_summary.json").read_text(encoding="utf-8"))
    assert {"held_out_spearman_rho", "top_5_precision", "train_rows", "val_rows"} <= set(surrogate_summary)
    assert (surrogate_dir / "surrogate_model.pkl").is_file()

    allocation_summary = json.loads((allocation_dir / "allocation_summary.json").read_text(encoding="utf-8"))
    assert {"rank_budget_total", "manifest_files", "backends"} <= set(allocation_summary)
    for backend in ("uniform", "layer_only", "timestep_only", "proposed"):
        assert (allocation_dir / f"{backend}.json").is_file()

    for run_dir in run_dirs:
        assert (run_dir / "run_provenance.json").is_file()
        assert (run_dir / "train_summary.json").is_file()
        assert (run_dir / "metrics.json").is_file()
        assert (run_dir / "checkpoint_info.json").is_file()
        summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert {"backend", "task", "allocation_manifest", "train_steps", "checkpoint_path"} <= set(summary)

    evaluation_summary = json.loads((evaluation_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert {"tasks", "backends", "metric_names", "evaluated_run_dirs"} <= set(evaluation_summary)
    assert (evaluation_dir / "baseline_metrics.csv").is_file()
    assert (evaluation_dir / "baseline_metrics.json").is_file()

    gate_summary = json.loads((gate_dir / "gate_memo.json").read_text(encoding="utf-8"))
    assert {"gate_status", "thresholds", "metrics", "source_run_dirs", "invalid_reasons", "decision_summary"} <= set(gate_summary)
    assert (gate_dir / "gate_memo.md").is_file()
