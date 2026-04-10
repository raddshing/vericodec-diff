from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "preflight_real_gpu.py"


def _write_fake_runtime(tmp_path: Path, *, cuda_available: bool = True, device_count: int = 1) -> Path:
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
                f"    def is_available(self): return {cuda_available!r}",
                f"    def device_count(self): return {device_count!r}",
                "    def get_device_name(self, index): return f'Fake GPU {index}'",
                "    def max_memory_allocated(self): return 536870912",
                "",
                "cuda = _Cuda()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "diffusers.py").write_text("__version__ = '0.0.test'\n", encoding="utf-8")
    return runtime_dir


def test_preflight_real_gpu_writes_expected_json(tmp_path: Path) -> None:
    runtime_dir = _write_fake_runtime(tmp_path)
    accelerate_config = tmp_path / "single_gpu_fp16.yaml"
    accelerate_config.write_text("compute_environment: LOCAL_MACHINE\n", encoding="utf-8")
    output_path = tmp_path / "preflight.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(output_path),
            "--expected-accelerate-config",
            str(accelerate_config),
            "--expected-diffusers-substring",
            "fake_runtime",
            "--require-cuda",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["torch_cuda_is_available"] is True
    assert payload["torch_device_count"] == 1
    assert payload["accelerate_config_file"] == str(accelerate_config.resolve())
    assert "fake_runtime" in payload["diffusers_file"]
    assert payload["gpu_names"] == ["Fake GPU 0"]


def test_preflight_real_gpu_fails_when_diffusers_path_is_wrong(tmp_path: Path) -> None:
    runtime_dir = _write_fake_runtime(tmp_path)
    accelerate_config = tmp_path / "single_gpu_fp16.yaml"
    accelerate_config.write_text("compute_environment: LOCAL_MACHINE\n", encoding="utf-8")
    output_path = tmp_path / "preflight.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runtime_dir)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(REPO_ROOT),
            "--output",
            str(output_path),
            "--expected-accelerate-config",
            str(accelerate_config),
            "--expected-diffusers-substring",
            "not-present",
            "--require-cuda",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
