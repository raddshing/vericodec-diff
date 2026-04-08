from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rd_lora.cells import (
    DEFAULT_CANDIDATE_RANKS,
    DEFAULT_LAYER_GROUP_COUNT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_TIMESTEP_BAND_COUNT,
    CellSchema,
    build_cell_schema,
    resolve_candidate_ranks,
    resolve_layer_catalog,
)
from rd_lora.features import UTILITY_RECORD_FIELDNAMES, build_cell_utility_records
from rd_lora.substrate.diffusers_sdxl import deep_update, display_path, resolve_path, save_json


DEFAULT_PROBE_CONFIG_PATH = "configs/rdlora_probe.yaml"
DEFAULT_OUTPUT_ROOT = "outputs/rd_lora/probe"
DEFAULT_ACCELERATE_CONFIG = "configs/accelerate/single_gpu_fp16.yaml"


class ProbeValidationError(ValueError):
    """Raised when the probe configuration or artifacts are invalid."""


def default_probe_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "output_root": DEFAULT_OUTPUT_ROOT,
            "accelerate_config": DEFAULT_ACCELERATE_CONFIG,
        },
        "run": {
            "name": "week2_probe",
            "seed": 20260409,
        },
        "preflight": {
            "require_torch_cuda": True,
            "required_diffusers_version": "0.38.0.dev0",
            "required_accelerate_config": DEFAULT_ACCELERATE_CONFIG,
        },
        "schema": {
            "layer_group_count": DEFAULT_LAYER_GROUP_COUNT,
            "timestep_band_count": DEFAULT_TIMESTEP_BAND_COUNT,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "candidate_ranks": list(DEFAULT_CANDIDATE_RANKS),
            "timesteps": None,
            "layer_catalog": [],
        },
        "features": {
            "include_attention_output_drift": True,
        },
        "backend": {
            "kind": "mock",
            "mock": {
                "sample_count": 2,
                "vector_size": 8,
            },
        },
    }


def _validate_repo_local_path(repo_root: Path, path: Path, *, name: str) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ProbeValidationError(f"{name} must resolve inside repo_root for deterministic paths") from exc


def _validate_single_path_token(raw_value: str, *, name: str) -> str:
    token = raw_value.strip()
    if not token:
        raise ProbeValidationError(f"{name} must be a non-empty path token")
    token_path = Path(token)
    if len(token_path.parts) != 1 or token_path.name != token or token in {".", ".."}:
        raise ProbeValidationError(f"{name} must be a single path component")
    return token


def _stable_float(*tokens: object) -> float:
    payload = "::".join(str(token) for token in tokens).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(1 << 64)


def _stable_signed_float(*tokens: object) -> float:
    return (2.0 * _stable_float(*tokens)) - 1.0


def _stable_vector(*, seed: int, sample_index: int, layer_id: str, step_index: int, size: int) -> list[float]:
    values: list[float] = []
    for feature_index in range(size):
        values.append(_stable_signed_float(seed, sample_index, layer_id, step_index, feature_index))
    return values


def resolve_probe_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_probe_config(), raw_config)
    paths = dict(config.get("paths", {}))
    run = dict(config.get("run", {}))
    preflight = dict(config.get("preflight", {}))
    schema = dict(config.get("schema", {}))
    features = dict(config.get("features", {}))
    backend = dict(config.get("backend", {}))
    mock_backend = dict(backend.get("mock", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    output_root = resolve_path(resolved_repo_root, str(paths.get("output_root", DEFAULT_OUTPUT_ROOT))).resolve()
    accelerate_config = resolve_path(
        resolved_repo_root,
        str(paths.get("accelerate_config", DEFAULT_ACCELERATE_CONFIG)),
    ).resolve()
    required_accelerate_config = resolve_path(
        resolved_repo_root,
        str(preflight.get("required_accelerate_config", DEFAULT_ACCELERATE_CONFIG)),
    ).resolve()
    output_dir = output_root / _validate_single_path_token(str(run.get("name", "week2_probe")), name="run.name")

    _validate_repo_local_path(resolved_repo_root, output_root, name="paths.output_root")
    _validate_repo_local_path(resolved_repo_root, output_dir, name="derived output_dir")
    _validate_repo_local_path(resolved_repo_root, accelerate_config, name="paths.accelerate_config")
    _validate_repo_local_path(
        resolved_repo_root,
        required_accelerate_config,
        name="preflight.required_accelerate_config",
    )

    layer_catalog = resolve_layer_catalog(schema.get("layer_catalog"))
    candidate_ranks = resolve_candidate_ranks(schema.get("candidate_ranks"))
    layer_group_count = int(schema.get("layer_group_count", DEFAULT_LAYER_GROUP_COUNT))
    timestep_band_count = int(schema.get("timestep_band_count", DEFAULT_TIMESTEP_BAND_COUNT))
    num_inference_steps = int(schema.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    if layer_group_count <= 0:
        raise ProbeValidationError("schema.layer_group_count must be positive")
    if timestep_band_count <= 0:
        raise ProbeValidationError("schema.timestep_band_count must be positive")
    if num_inference_steps <= 0:
        raise ProbeValidationError("schema.num_inference_steps must be positive")

    if str(backend.get("kind", "mock")).strip() != "mock":
        raise ProbeValidationError("backend.kind must currently be 'mock'")
    sample_count = int(mock_backend.get("sample_count", 2))
    vector_size = int(mock_backend.get("vector_size", 8))
    if sample_count <= 0:
        raise ProbeValidationError("backend.mock.sample_count must be positive")
    if vector_size <= 0:
        raise ProbeValidationError("backend.mock.vector_size must be positive")

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "accelerate_config": str(accelerate_config),
        },
        "run": {
            "name": str(run.get("name", "week2_probe")),
            "seed": int(run.get("seed", 20260409)),
        },
        "preflight": {
            "require_torch_cuda": bool(preflight.get("require_torch_cuda", True)),
            "required_diffusers_version": preflight.get("required_diffusers_version"),
            "required_accelerate_config": str(required_accelerate_config),
        },
        "schema": {
            "layer_group_count": layer_group_count,
            "timestep_band_count": timestep_band_count,
            "num_inference_steps": num_inference_steps,
            "candidate_ranks": list(candidate_ranks),
            "timesteps": [int(value) for value in schema.get("timesteps") or []],
            "layer_catalog": [layer.to_dict() for layer in layer_catalog],
        },
        "features": {
            "include_attention_output_drift": bool(features.get("include_attention_output_drift", True)),
        },
        "backend": {
            "kind": "mock",
            "mock": {
                "sample_count": sample_count,
                "vector_size": vector_size,
            },
        },
    }


def build_probe_schema(config: Mapping[str, Any]) -> CellSchema:
    schema_config = dict(config["schema"])
    return build_cell_schema(
        layer_catalog=resolve_layer_catalog(schema_config.get("layer_catalog")),
        layer_group_count=int(schema_config["layer_group_count"]),
        timestep_values=schema_config.get("timesteps") or None,
        num_inference_steps=int(schema_config["num_inference_steps"]),
        timestep_band_count=int(schema_config["timestep_band_count"]),
        candidate_ranks=schema_config["candidate_ranks"],
    )


def run_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    preflight = dict(config["preflight"])
    accelerate_config = Path(config["paths"]["accelerate_config"]).resolve()
    required_accelerate_config = Path(preflight["required_accelerate_config"]).resolve()
    report = {
        "ok": True,
        "issues": [],
        "torch_cuda_is_available": None,
        "diffusers_version": None,
        "accelerate_config": str(accelerate_config),
        "accelerate_config_matches": accelerate_config == required_accelerate_config,
        "required_accelerate_config": str(required_accelerate_config),
    }

    if not report["accelerate_config_matches"]:
        report["issues"].append("paths.accelerate_config does not match the required repo-local accelerate config.")

    if bool(preflight.get("require_torch_cuda", True)):
        try:
            import torch
        except Exception as exc:
            report["issues"].append(f"Unable to import torch: {type(exc).__name__}: {exc}")
            report["torch_cuda_is_available"] = False
        else:
            report["torch_cuda_is_available"] = bool(torch.cuda.is_available())
            if not report["torch_cuda_is_available"]:
                report["issues"].append("torch.cuda.is_available() must be True before probe runs.")

    required_diffusers_version = preflight.get("required_diffusers_version")
    if required_diffusers_version not in (None, ""):
        try:
            diffusers_version = importlib.metadata.version("diffusers")
        except importlib.metadata.PackageNotFoundError:
            report["issues"].append("diffusers is not installed in the active environment.")
        else:
            report["diffusers_version"] = diffusers_version
            if diffusers_version != str(required_diffusers_version):
                report["issues"].append(
                    f"diffusers version must be {required_diffusers_version}, found {diffusers_version}."
                )

    report["ok"] = not report["issues"]
    if not report["ok"]:
        raise ProbeValidationError("\n".join(str(issue) for issue in report["issues"]))
    return report


def _build_mock_events(config: Mapping[str, Any], schema: CellSchema) -> list[dict[str, Any]]:
    sample_count = int(config["backend"]["mock"]["sample_count"])
    vector_size = int(config["backend"]["mock"]["vector_size"])
    seed = int(config["run"]["seed"])
    max_rank = max(schema.candidate_ranks)
    events: list[dict[str, Any]] = []

    for sample_index in range(sample_count):
        for layer in schema.layer_catalog:
            for step_index, timestep_value in enumerate(schema.timestep_values):
                reference_output = _stable_vector(
                    seed=seed,
                    sample_index=sample_index,
                    layer_id=layer.layer_id,
                    step_index=step_index,
                    size=vector_size,
                )
                raw_delta = _stable_vector(
                    seed=seed + 17,
                    sample_index=sample_index,
                    layer_id=layer.layer_id,
                    step_index=step_index,
                    size=vector_size,
                )
                base_scale = 0.45 + (0.25 * _stable_float(seed, layer.layer_id, step_index, sample_index, "scale"))
                candidate_outputs: dict[int, list[float]] = {}
                for rank in schema.candidate_ranks:
                    rank_fraction = float(rank) / float(max_rank) if max_rank > 0 else 0.0
                    attenuation = max(0.1, 1.0 - (0.82 * rank_fraction))
                    candidate_outputs[rank] = [
                        float(reference_output[index]) + (float(raw_delta[index]) * base_scale * attenuation)
                        for index in range(vector_size)
                    ]
                events.append(
                    {
                        "sample_index": sample_index,
                        "layer_id": layer.layer_id,
                        "layer_kind": layer.kind,
                        "step_index": step_index,
                        "timestep": timestep_value,
                        "reference_output": reference_output,
                        "candidate_outputs": candidate_outputs,
                    }
                )
    return events


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def run_probe(
    config: Mapping[str, Any],
    *,
    resolved_config_path: Path,
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    preflight = run_preflight(config)
    schema = build_probe_schema(config)
    events = _build_mock_events(config, schema)

    cell_schema_path = output_dir / "cell_schema.json"
    utility_records_json_path = output_dir / "utility_records.json"
    utility_records_csv_path = output_dir / "utility_records.csv"

    save_json(cell_schema_path, schema.to_dict())

    cell_to_events: dict[str, list[dict[str, Any]]] = {cell.cell_id: [] for cell in schema.cells}
    for event in events:
        cell = schema.locate_cell(layer_id=str(event["layer_id"]), step_index=int(event["step_index"]))
        cell_to_events[cell.cell_id].append(event)

    utility_records: list[dict[str, Any]] = []
    for cell in schema.cells:
        cell_payload = {
            "cell_id": cell.cell_id,
            "layer_group_id": cell.layer_group.group_id,
            "timestep_band_id": cell.timestep_band.band_id,
            "is_attention_cell": cell.is_attention_cell,
            "layer_count": len(cell.layer_group.layer_ids),
            "step_count": len(cell.timestep_band.step_indices),
        }
        utility_records.extend(
            build_cell_utility_records(
                cell=cell_payload,
                events=cell_to_events[cell.cell_id],
                candidate_ranks=schema.candidate_ranks,
                include_attention_output_drift=bool(config["features"]["include_attention_output_drift"]),
            )
        )

    save_json(
        utility_records_json_path,
        {
            "schema_version": 1,
            "record_count": len(utility_records),
            "records": utility_records,
        },
    )
    _write_csv(
        utility_records_csv_path,
        utility_records,
        fieldnames=UTILITY_RECORD_FIELDNAMES,
    )

    summary = {
        "schema_version": 1,
        "backend": str(config["backend"]["kind"]),
        "resolved_config": display_path(resolved_config_path, repo_root),
        "cell_schema": display_path(cell_schema_path, repo_root),
        "utility_records_json": display_path(utility_records_json_path, repo_root),
        "utility_records_csv": display_path(utility_records_csv_path, repo_root),
        "cell_count": len(schema.cells),
        "candidate_rank_count": len(schema.candidate_ranks),
        "record_count": len(utility_records),
        "event_count": len(events),
        "preflight": preflight,
        "completed_successfully": True,
        "issues": [],
    }
    summary_path = output_dir / "probe_summary.json"
    save_json(summary_path, summary)
    summary["summary_path"] = display_path(summary_path, repo_root)
    return summary


def validate_probe_outputs(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "probe_summary.json"
    if not summary_path.is_file():
        raise ProbeValidationError(f"Missing probe summary: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ProbeValidationError("probe_summary.json must parse to a mapping")

    required_summary_keys = {
        "schema_version",
        "backend",
        "resolved_config",
        "cell_schema",
        "utility_records_json",
        "utility_records_csv",
        "cell_count",
        "candidate_rank_count",
        "record_count",
        "event_count",
        "preflight",
        "completed_successfully",
        "issues",
    }
    missing_summary = sorted(required_summary_keys - set(summary))
    if missing_summary:
        raise ProbeValidationError(f"probe_summary.json is missing keys: {missing_summary}")

    cell_schema_path = output_dir / "cell_schema.json"
    if not cell_schema_path.is_file():
        raise ProbeValidationError(f"Missing cell schema JSON: {cell_schema_path}")
    cell_schema = json.loads(cell_schema_path.read_text(encoding="utf-8"))
    if cell_schema.get("layer_group_count") != DEFAULT_LAYER_GROUP_COUNT:
        raise ProbeValidationError("cell_schema.json layer_group_count does not match the default schema")
    if cell_schema.get("timestep_band_count") != DEFAULT_TIMESTEP_BAND_COUNT:
        raise ProbeValidationError("cell_schema.json timestep_band_count does not match the default schema")
    if list(cell_schema.get("candidate_ranks", [])) != list(DEFAULT_CANDIDATE_RANKS):
        raise ProbeValidationError("cell_schema.json candidate_ranks do not match the default schema")
    if len(cell_schema.get("cells", [])) != DEFAULT_LAYER_GROUP_COUNT * DEFAULT_TIMESTEP_BAND_COUNT:
        raise ProbeValidationError("cell_schema.json does not contain 24 cells")

    utility_json_path = output_dir / "utility_records.json"
    utility_csv_path = output_dir / "utility_records.csv"
    if not utility_json_path.is_file():
        raise ProbeValidationError(f"Missing utility JSON: {utility_json_path}")
    if not utility_csv_path.is_file():
        raise ProbeValidationError(f"Missing utility CSV: {utility_csv_path}")

    utility_payload = json.loads(utility_json_path.read_text(encoding="utf-8"))
    if sorted(utility_payload) != ["record_count", "records", "schema_version"]:
        raise ProbeValidationError("utility_records.json must contain schema_version, record_count, and records")

    with utility_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != UTILITY_RECORD_FIELDNAMES:
            raise ProbeValidationError("utility_records.csv has an unexpected header schema")
        csv_rows = list(reader)

    json_records = utility_payload.get("records", [])
    if len(csv_rows) != len(json_records):
        raise ProbeValidationError("utility JSON/CSV record counts do not match")
    expected_record_count = DEFAULT_LAYER_GROUP_COUNT * DEFAULT_TIMESTEP_BAND_COUNT * len(DEFAULT_CANDIDATE_RANKS)
    if len(csv_rows) != expected_record_count:
        raise ProbeValidationError("utility record count does not match 24 cells x 5 candidate ranks")

    return {
        "ok": True,
        "summary_path": str(summary_path),
        "cell_schema_path": str(cell_schema_path),
        "utility_json_path": str(utility_json_path),
        "utility_csv_path": str(utility_csv_path),
        "record_count": len(csv_rows),
    }


__all__ = [
    "DEFAULT_ACCELERATE_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_PROBE_CONFIG_PATH",
    "ProbeValidationError",
    "build_probe_schema",
    "default_probe_config",
    "resolve_probe_config",
    "run_preflight",
    "run_probe",
    "validate_probe_outputs",
]
