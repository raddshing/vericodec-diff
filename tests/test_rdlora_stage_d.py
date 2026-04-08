from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


PROBE_SCRIPT = REPO_ROOT / "scripts" / "run_rdlora_probe.py"
SURROGATE_SCRIPT = REPO_ROOT / "scripts" / "train_rdlora_surrogate.py"
ALLOCATOR_SCRIPT = REPO_ROOT / "scripts" / "solve_rdlora_allocation.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_rdlora.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_rdlora.py"
MEMO_SCRIPT = REPO_ROOT / "scripts" / "write_rdlora_gate_memo.py"


def _run_stage_bc(temp_root: Path) -> tuple[Path, Path]:
    probe_root = temp_root / "probe_outputs"
    allocator_root = temp_root / "allocator_outputs"
    probe_run_dir = probe_root / "probe_smoke"
    allocator_run_dir = allocator_root / "alloc_smoke"

    subprocess.run(
        [
            sys.executable,
            str(PROBE_SCRIPT),
            "--config",
            "configs/rdlora_probe.yaml",
            "--run-name",
            "probe_smoke",
            "--set",
            f"paths.output_root={json.dumps(str(probe_root))}",
            "--set",
            "preflight.require_torch_cuda=false",
            "--set",
            "preflight.required_diffusers_version=null",
            "--set",
            "backend.mock.sample_count=2",
            "--set",
            "backend.mock.vector_size=4",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(SURROGATE_SCRIPT),
            "--config",
            "configs/rdlora_allocator.yaml",
            "--run-name",
            "alloc_smoke",
            "--set",
            f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
            "--set",
            f"paths.output_root={json.dumps(str(allocator_root))}",
            "--set",
            "preflight.require_torch_cuda=false",
            "--set",
            "preflight.required_diffusers_version=null",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(ALLOCATOR_SCRIPT),
            "--config",
            "configs/rdlora_allocator.yaml",
            "--run-name",
            "alloc_smoke",
            "--set",
            f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
            "--set",
            f"paths.output_root={json.dumps(str(allocator_root))}",
            "--set",
            "allocation.budget=96",
            "--set",
            "preflight.require_torch_cuda=false",
            "--set",
            "preflight.required_diffusers_version=null",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return probe_run_dir, allocator_run_dir


class StageDTests(unittest.TestCase):
    def test_train_script_writes_dry_run_summary(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            temp_root = Path(tmpdir)
            probe_run_dir, allocator_run_dir = _run_stage_bc(temp_root)
            output_root = temp_root / "stage_d"
            memo_root = temp_root / "memos"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--config",
                    "configs/rdlora_stage_d.yaml",
                    "--run-name",
                    "stage_d_smoke",
                    "--set",
                    f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
                    "--set",
                    f"paths.allocator_run_dir={json.dumps(str(allocator_run_dir))}",
                    "--set",
                    f"paths.output_root={json.dumps(str(output_root))}",
                    "--set",
                    f"paths.memo_output_dir={json.dumps(str(memo_root))}",
                    "--set",
                    "preflight.require_torch_cuda=false",
                    "--set",
                    "preflight.required_diffusers_version=null",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            training_summary_path = output_root / "stage_d_smoke" / "training" / "training_summary.json"
            payload = json.loads(training_summary_path.read_text(encoding="utf-8"))
            self.assertIn("preflight_ok=1", completed.stdout)
            self.assertEqual(payload["uniform_rank"], 4)
            self.assertTrue(any(item["method"] == "uniform" for item in payload["task_runs"]))
            self.assertTrue(any(item["status"] == "blocked_backend_missing" for item in payload["task_runs"]))
            uniform_launch = output_root / "stage_d_smoke" / "training" / "uniform" / "t_lora_dog_subject" / "stage_d_smoke" / "launch_command.sh"
            self.assertTrue(uniform_launch.is_file())

    def test_eval_script_writes_gate_metrics_and_baseline_table(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            temp_root = Path(tmpdir)
            probe_run_dir, allocator_run_dir = _run_stage_bc(temp_root)
            output_root = temp_root / "stage_d"
            memo_root = temp_root / "memos"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVAL_SCRIPT),
                    "--config",
                    "configs/rdlora_stage_d.yaml",
                    "--run-name",
                    "stage_d_eval",
                    "--set",
                    f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
                    "--set",
                    f"paths.allocator_run_dir={json.dumps(str(allocator_run_dir))}",
                    "--set",
                    f"paths.output_root={json.dumps(str(output_root))}",
                    "--set",
                    f"paths.memo_output_dir={json.dumps(str(memo_root))}",
                    "--set",
                    "preflight.require_torch_cuda=false",
                    "--set",
                    "preflight.required_diffusers_version=null",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            evaluation_summary_path = output_root / "stage_d_eval" / "evaluation" / "evaluation_summary.json"
            payload = json.loads(evaluation_summary_path.read_text(encoding="utf-8"))
            self.assertIn("decision=NO_GO", completed.stdout)
            self.assertEqual(payload["decision"], "NO_GO")
            methods = {row["method"]: row for row in payload["baseline_table"]}
            self.assertIn("uniform", methods)
            self.assertIn("proposed", methods)
            self.assertGreaterEqual(
                float(methods["proposed"]["total_measured_utility"]),
                float(methods["uniform"]["total_measured_utility"]),
            )

    def test_memo_script_writes_requested_outputs(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            temp_root = Path(tmpdir)
            probe_run_dir, allocator_run_dir = _run_stage_bc(temp_root)
            output_root = temp_root / "stage_d"
            memo_root = temp_root / "memos"
            memo_md_path = memo_root / "rdlora_gate_memo.md"
            memo_json_path = memo_root / "rdlora_gate_memo.json"

            subprocess.run(
                [
                    sys.executable,
                    str(EVAL_SCRIPT),
                    "--config",
                    "configs/rdlora_stage_d.yaml",
                    "--run-name",
                    "stage_d_eval",
                    "--set",
                    f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
                    "--set",
                    f"paths.allocator_run_dir={json.dumps(str(allocator_run_dir))}",
                    "--set",
                    f"paths.output_root={json.dumps(str(output_root))}",
                    "--set",
                    f"paths.memo_output_dir={json.dumps(str(memo_root))}",
                    "--set",
                    "preflight.require_torch_cuda=false",
                    "--set",
                    "preflight.required_diffusers_version=null",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(MEMO_SCRIPT),
                    "--config",
                    "configs/rdlora_stage_d.yaml",
                    "--run-name",
                    "stage_d_eval",
                    "--memo-md-path",
                    str(memo_md_path),
                    "--memo-json-path",
                    str(memo_json_path),
                    "--set",
                    f"paths.probe_run_dir={json.dumps(str(probe_run_dir))}",
                    "--set",
                    f"paths.allocator_run_dir={json.dumps(str(allocator_run_dir))}",
                    "--set",
                    f"paths.output_root={json.dumps(str(output_root))}",
                    "--set",
                    f"paths.memo_output_dir={json.dumps(str(memo_root))}",
                    "--set",
                    "preflight.require_torch_cuda=false",
                    "--set",
                    "preflight.required_diffusers_version=null",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(memo_json_path.read_text(encoding="utf-8"))
            self.assertIn("decision=NO_GO", completed.stdout)
            self.assertEqual(payload["decision"], "NO_GO")
            self.assertTrue(memo_md_path.is_file())
            self.assertIn("Decision: **NO_GO**", memo_md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
