from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval import generate as generation


class _FakePipelineOutput:
    def __init__(self, image: Image.Image) -> None:
        self.images = [image]


class _FakeGenerator:
    def __init__(self, device: str) -> None:
        self.device = device
        self._seed: int | None = None

    def manual_seed(self, seed: int) -> "_FakeGenerator":
        self._seed = int(seed)
        return self

    def initial_seed(self) -> int:
        if self._seed is None:
            raise RuntimeError("Seed was not set")
        return self._seed


class _FakeTorch:
    float32 = "float32"
    Generator = _FakeGenerator


class _FakePipeline:
    last_instance: "_FakePipeline | None" = None
    last_model_id: str | None = None

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.lora_paths: list[str] = []
        self.to_calls: list[tuple[str, object]] = []
        self.unloaded = False

    @classmethod
    def from_pretrained(cls, base_model_id: str) -> "_FakePipeline":
        cls.last_model_id = base_model_id
        cls.last_instance = cls()
        return cls.last_instance

    def load_lora_weights(self, checkpoint_path: str) -> None:
        self.lora_paths.append(checkpoint_path)

    def to(self, device: str, torch_dtype: object) -> "_FakePipeline":
        self.to_calls.append((device, torch_dtype))
        return self

    def __call__(
        self,
        prompt: str,
        *,
        generator: _FakeGenerator,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> _FakePipelineOutput:
        seed = int(generator.initial_seed())
        self.calls.append(
            {
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
            }
        )
        return _FakePipelineOutput(Image.new("RGB", (8, 8), color=(seed % 255, 32, 96)))

    def unload_lora_weights(self) -> None:
        self.unloaded = True


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_manifest(path: Path) -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "prompt_id": "prompt_001",
                "prompt": "poster that says HELLO",
                "seed": 101,
                "expected_text": "HELLO",
            },
            {
                "prompt_id": "prompt_002",
                "prompt": "road sign that says LEFT",
                "seed": 202,
                "expected_text": "LEFT",
            },
            {
                "prompt_id": "prompt_003",
                "prompt": "menu board that says SOUP",
                "seed": 303,
                "expected_text": "SOUP",
            },
        ]
    )
    rows.to_csv(path, index=False)
    return rows


def _write_run_artifacts(run_dir: Path, checkpoint_path: Path) -> None:
    _write_json(
        run_dir / "train_summary.json",
        {
            "backend": "uniform",
            "task": "text_rendering",
        },
    )
    _write_json(
        run_dir / "checkpoint_info.json",
        {
            "checkpoint_path": str(checkpoint_path),
        },
    )


def test_generate_eval_images_for_run_writes_expected_records_and_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "checkpoint-12"
    checkpoint_dir.mkdir(parents=True)
    _write_run_artifacts(run_dir, checkpoint_dir)

    manifest_path = tmp_path / "manifest.csv"
    manifest = _write_manifest(manifest_path)
    output_dir = tmp_path / "outputs"

    monkeypatch.setattr(generation, "StableDiffusionXLPipeline", _FakePipeline)
    monkeypatch.setattr(generation, "torch", _FakeTorch)

    records_path = generation.generate_eval_images_for_run(
        run_dir=run_dir,
        prompt_manifest_path=manifest_path,
        output_dir=output_dir,
        base_model_id="fake/sdxl",
        guidance_scale=7.5,
        num_inference_steps=30,
        device="cpu",
        torch_dtype="float32",
    )

    records = pd.read_csv(records_path)
    expected_columns = [
        "run_dir",
        "backend",
        "task",
        "prompt_id",
        "prompt",
        "seed",
        "expected_text",
        "image_path",
    ]
    assert list(records.columns) == expected_columns
    assert len(records) == len(manifest)

    expected_names = [
        f"uniform__text_rendering__{row.prompt_id}__seed{int(row.seed)}.png"
        for row in manifest.itertuples(index=False)
    ]
    assert [Path(value).name for value in records["image_path"].tolist()] == expected_names
    for image_path in records["image_path"].tolist():
        assert Path(image_path).is_file()

    assert records["backend"].tolist() == ["uniform"] * len(manifest)
    assert records["task"].tolist() == ["text_rendering"] * len(manifest)
    assert records["run_dir"].tolist() == [str(run_dir.resolve())] * len(manifest)

    pipe = _FakePipeline.last_instance
    assert pipe is not None
    assert _FakePipeline.last_model_id == "fake/sdxl"
    assert pipe.lora_paths == [str(checkpoint_dir.resolve())]
    assert pipe.to_calls == [("cpu", _FakeTorch.float32)]
    assert [call["seed"] for call in pipe.calls] == [101, 202, 303]
    assert {call["guidance_scale"] for call in pipe.calls} == {7.5}
    assert {call["num_inference_steps"] for call in pipe.calls} == {30}
    assert pipe.unloaded is True


def test_generate_eval_images_for_run_raises_when_checkpoint_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    missing_checkpoint = tmp_path / "missing-checkpoint"
    _write_run_artifacts(run_dir, missing_checkpoint)

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        generation.generate_eval_images_for_run(
            run_dir=run_dir,
            prompt_manifest_path=manifest_path,
            output_dir=tmp_path / "outputs",
            device="cpu",
            torch_dtype="float32",
        )


def test_generate_eval_images_for_run_raises_when_train_summary_is_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = tmp_path / "checkpoint-12"
    checkpoint_dir.mkdir(parents=True)
    _write_json(
        run_dir / "checkpoint_info.json",
        {
            "checkpoint_path": str(checkpoint_dir),
        },
    )

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path)

    with pytest.raises(FileNotFoundError, match="train_summary.json"):
        generation.generate_eval_images_for_run(
            run_dir=run_dir,
            prompt_manifest_path=manifest_path,
            output_dir=tmp_path / "outputs",
            device="cpu",
            torch_dtype="float32",
        )
