from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rd_lora.cells import CellSchema, build_cell_schema, validate_cell_targets
from rd_lora.runtime.allocation_manifest import load_allocation_manifest, validate_allocation_manifest
from rd_lora.training.timestep_routing import (
    build_adapter_routing_table,
    build_timestep_band_routes,
    describe_routing_table,
)


ALLOWED_TARGET_MODULES = ("to_k", "to_q", "to_v", "to_out.0")
EXPECTED_BACKENDS = ("uniform", "layer_only", "timestep_only", "proposed")
REQUIRED_PROBE_ROW_KEYS = (
    "task",
    "cell_id",
    "layer_group",
    "timestep_band",
    "candidate_rank",
    "pre_loss",
    "post_loss",
    "utility",
    "optimizer_steps",
    "train_batch_count",
    "val_batch_count",
)
REQUIRED_PROBE_SUMMARY_KEYS = (
    "task",
    "cell_count",
    "candidate_ranks",
    "row_count",
)
REQUIRED_SURROGATE_SUMMARY_KEYS = (
    "held_out_spearman_rho",
    "top_5_precision",
    "train_rows",
    "val_rows",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
_REAL_PROBE_ARTIFACTS = (
    (
        "subject_personalization",
        REPO_ROOT / "outputs/rd_lora/probe/task1_real/cell_utility.json",
        REPO_ROOT / "outputs/rd_lora/probe/task1_real/probe_summary.json",
    ),
    (
        "style_domain",
        REPO_ROOT / "outputs/rd_lora/probe/task2_real/cell_utility.json",
        REPO_ROOT / "outputs/rd_lora/probe/task2_real/probe_summary.json",
    ),
)
_REAL_SURROGATE_SUMMARY = REPO_ROOT / "outputs/rd_lora/surrogate/gate_real/surrogate_summary.json"
_REAL_SURROGATE_MODEL = REPO_ROOT / "outputs/rd_lora/surrogate/gate_real/surrogate_model.pkl"
_REAL_ALLOCATION_SUMMARY = REPO_ROOT / "outputs/rd_lora/allocation/gate_real/allocation_summary.json"
_REAL_ALLOCATION_MANIFESTS = tuple(
    REPO_ROOT / f"outputs/rd_lora/allocation/gate_real/{backend_name}.json"
    for backend_name in EXPECTED_BACKENDS
)


class AdapterFactoryError(ValueError):
    """Raised when contract-bound backend planning cannot proceed."""


def _discover_alternative(path: Path) -> str | None:
    if not path.parent.exists():
        return None
    for candidate in sorted(path.parent.glob(path.name)):
        return str(candidate)
    return None


def _require_existing_path(path: Path) -> Path:
    if not path.exists():
        alternative = _discover_alternative(path)
        raise AdapterFactoryError(
            f"Missing required artifact {path}; discovered_alternative={alternative or '<none>'}"
        )
    return path


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(_require_existing_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AdapterFactoryError(f"{path} must parse to a mapping")
    return dict(payload)


def _normalize_target_modules(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ALLOWED_TARGET_MODULES
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise AdapterFactoryError("training.target_modules must be a list")
    normalized = tuple(str(value).strip() for value in raw_value if str(value).strip())
    if normalized != ALLOWED_TARGET_MODULES:
        raise AdapterFactoryError(
            f"LoRA target_modules must remain {ALLOWED_TARGET_MODULES!r}; received={normalized!r}"
        )
    return normalized


def _normalize_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(config or {})
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        raise AdapterFactoryError("config.training must be a mapping")
    use_rslora = bool(training.get("use_rslora", True))
    target_modules = _normalize_target_modules(training.get("target_modules", ALLOWED_TARGET_MODULES))
    return {
        "use_rslora": use_rslora,
        "target_modules": target_modules,
    }


def _require_mapping_keys(
    payload: Mapping[str, Any],
    required_keys: Sequence[str],
    *,
    path: Path,
    context: str,
    discovered_keys: Sequence[str],
) -> None:
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise AdapterFactoryError(
            f"{path} missing {context} keys {missing}; discovered_keys={list(discovered_keys)}"
        )


def _load_probe_contract(
    *,
    task_name: str,
    utility_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    utility_payload = _read_json_mapping(utility_path)
    utility_keys = sorted(utility_payload)
    if "rows" not in utility_payload:
        raise AdapterFactoryError(
            f"{utility_path} must expose top_level_key 'rows'; discovered_keys={utility_keys}"
        )
    rows = utility_payload["rows"]
    if not isinstance(rows, list):
        raise AdapterFactoryError(f"{utility_path} rows must be a list")
    if not rows:
        raise AdapterFactoryError(f"{utility_path} rows must not be empty")
    for row_index, row_value in enumerate(rows):
        if not isinstance(row_value, Mapping):
            raise AdapterFactoryError(f"{utility_path} rows[{row_index}] must be a mapping")
        row = dict(row_value)
        _require_mapping_keys(
            row,
            REQUIRED_PROBE_ROW_KEYS,
            path=utility_path,
            context=f"probe row {row_index}",
            discovered_keys=sorted(row),
        )

    summary_payload = _read_json_mapping(summary_path)
    _require_mapping_keys(
        summary_payload,
        REQUIRED_PROBE_SUMMARY_KEYS,
        path=summary_path,
        context="probe summary",
        discovered_keys=sorted(summary_payload),
    )
    summary_task = str(summary_payload["task"]).strip()
    if summary_task != task_name:
        raise AdapterFactoryError(
            f"{summary_path} task mismatch: expected={task_name!r} discovered={summary_task!r}"
        )

    return {
        "task": task_name,
        "utility_path": str(utility_path),
        "utility_top_level_keys": utility_keys,
        "row_source": "rows",
        "row_count": len(rows),
        "row_keys": sorted(rows[0]),
        "summary_path": str(summary_path),
        "summary_keys": sorted(summary_payload),
    }


def _load_surrogate_contract() -> dict[str, Any]:
    summary_payload = _read_json_mapping(_REAL_SURROGATE_SUMMARY)
    _require_mapping_keys(
        summary_payload,
        REQUIRED_SURROGATE_SUMMARY_KEYS,
        path=_REAL_SURROGATE_SUMMARY,
        context="surrogate summary",
        discovered_keys=sorted(summary_payload),
    )
    model_bytes = _require_existing_path(_REAL_SURROGATE_MODEL).read_bytes()
    return {
        "summary_path": str(_REAL_SURROGATE_SUMMARY),
        "summary_keys": sorted(summary_payload),
        "model_path": str(_REAL_SURROGATE_MODEL),
        "model_size_bytes": len(model_bytes),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
    }


def _load_allocation_contract() -> dict[str, Any]:
    summary_payload = _read_json_mapping(_REAL_ALLOCATION_SUMMARY)
    backends: list[str] = []
    manifests: list[dict[str, Any]] = []
    for path in _REAL_ALLOCATION_MANIFESTS:
        manifest = load_allocation_manifest(_require_existing_path(path))
        backends.append(str(manifest["backend"]))
        manifests.append(manifest)
    if tuple(backends) != EXPECTED_BACKENDS:
        raise AdapterFactoryError(
            f"Allocation manifest backends must be {EXPECTED_BACKENDS!r}; discovered={tuple(backends)!r}"
        )
    return {
        "summary_path": str(_REAL_ALLOCATION_SUMMARY),
        "summary_keys": sorted(summary_payload),
        "backends": backends,
        "manifests": {
            manifest["backend"]: {
                "path": str(path),
                "layer_groups": list(manifest["layer_groups"]),
                "timestep_bands": list(manifest["timestep_bands"]),
                "cell_count": len(manifest["cells"]),
            }
            for path, manifest in zip(_REAL_ALLOCATION_MANIFESTS, manifests, strict=True)
        },
    }


def load_actual_contract_binding() -> dict[str, Any]:
    return {
        "probe": {
            task_name: _load_probe_contract(
                task_name=task_name,
                utility_path=utility_path,
                summary_path=summary_path,
            )
            for task_name, utility_path, summary_path in _REAL_PROBE_ARTIFACTS
        },
        "surrogate": _load_surrogate_contract(),
        "allocation": _load_allocation_contract(),
    }


def _schema_layer_group_module_map(
    schema: CellSchema,
    *,
    target_modules: Sequence[str],
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for group in schema.layer_groups:
        module_names: list[str] = []
        for layer_id in group.layer_ids:
            for target_module in target_modules:
                module_names.append(f"{layer_id}.{target_module}")
        mapping[group.group_id] = module_names
    return mapping


def _schema_cell_lookup(schema: CellSchema) -> dict[str, dict[str, str]]:
    return {
        cell.cell_id: {
            "layer_group": cell.layer_group.group_id,
            "timestep_band": cell.timestep_band.band_id,
        }
        for cell in schema.cells
    }


def _validate_manifest_binding(
    manifest: Mapping[str, Any],
    *,
    schema: CellSchema,
    target_modules: Sequence[str],
) -> dict[str, dict[str, Any]]:
    expected_layer_groups = [group.group_id for group in schema.layer_groups]
    expected_timestep_bands = [band.band_id for band in schema.timestep_bands]
    if list(manifest["layer_groups"]) != expected_layer_groups:
        raise AdapterFactoryError(
            f"manifest layer_groups mismatch; expected={expected_layer_groups} discovered={list(manifest['layer_groups'])}"
        )
    if list(manifest["timestep_bands"]) != expected_timestep_bands:
        raise AdapterFactoryError(
            "manifest timestep_bands mismatch; "
            f"expected={expected_timestep_bands} discovered={list(manifest['timestep_bands'])}"
        )

    schema_cells = _schema_cell_lookup(schema)
    cells_by_id: dict[str, dict[str, Any]] = {}
    for cell in manifest["cells"]:
        cell_id = str(cell["cell_id"])
        if cell_id not in schema_cells:
            raise AdapterFactoryError(f"Unknown cell_id {cell_id!r} in manifest")
        if tuple(cell["target_modules"]) != tuple(target_modules):
            raise AdapterFactoryError(
                f"cell {cell_id!r} target_modules mismatch; expected={tuple(target_modules)!r} discovered={tuple(cell['target_modules'])!r}"
            )
        expected = schema_cells[cell_id]
        if str(cell["layer_group"]) != expected["layer_group"]:
            raise AdapterFactoryError(
                f"cell {cell_id!r} layer_group mismatch; expected={expected['layer_group']!r} discovered={cell['layer_group']!r}"
            )
        if str(cell["timestep_band"]) != expected["timestep_band"]:
            raise AdapterFactoryError(
                f"cell {cell_id!r} timestep_band mismatch; expected={expected['timestep_band']!r} discovered={cell['timestep_band']!r}"
            )
        cells_by_id[cell_id] = dict(cell)
    if set(cells_by_id) != set(schema_cells):
        missing = sorted(set(schema_cells) - set(cells_by_id))
        extra = sorted(set(cells_by_id) - set(schema_cells))
        raise AdapterFactoryError(
            f"manifest cell inventory mismatch; missing={missing} extra={extra}"
        )
    return cells_by_id


def _group_cell_ids(
    cells_by_id: Mapping[str, Mapping[str, Any]],
    *,
    field_name: str,
    ordered_names: Sequence[str],
) -> dict[str, list[str]]:
    grouped = {str(name): [] for name in ordered_names}
    for cell_id, cell in cells_by_id.items():
        grouped[str(cell[field_name])].append(str(cell_id))
    for name in ordered_names:
        grouped[str(name)].sort()
    return grouped


def _build_timestep_band_metadata(
    cells_by_id: Mapping[str, Mapping[str, Any]],
    *,
    timestep_bands: Sequence[str],
) -> dict[str, dict[str, Any]]:
    cell_ids_by_band = _group_cell_ids(
        cells_by_id,
        field_name="timestep_band",
        ordered_names=timestep_bands,
    )
    metadata: dict[str, dict[str, Any]] = {}
    for timestep_band in timestep_bands:
        band_cell_ids = list(cell_ids_by_band[str(timestep_band)])
        positive_rank_cell_count = sum(
            1
            for cell_id in band_cell_ids
            if int(cells_by_id[str(cell_id)]["rank"]) > 0
        )
        metadata[str(timestep_band)] = {
            "timestep_band": str(timestep_band),
            "cell_ids": band_cell_ids,
            "cell_count": len(band_cell_ids),
            "positive_rank_cell_count": positive_rank_cell_count,
            "zero_rank_cell_count": len(band_cell_ids) - positive_rank_cell_count,
            "has_trainable_params": positive_rank_cell_count > 0,
            "noop": positive_rank_cell_count == 0,
        }
    return metadata


def _module_patterns_for_cell_ids(
    cell_ids: Sequence[str],
    *,
    cells_by_id: Mapping[str, Mapping[str, Any]],
    layer_group_module_map: Mapping[str, Sequence[str]],
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    module_names: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    for cell_id in cell_ids:
        cell = cells_by_id[str(cell_id)]
        for module_name in layer_group_module_map[str(cell["layer_group"])]:
            module_names.append(module_name)
            rank_pattern[module_name] = int(cell["rank"])
            alpha_pattern[module_name] = int(cell["alpha"])
    return sorted(set(module_names)), rank_pattern, alpha_pattern


def _single_value(cells: Sequence[Mapping[str, Any]], field_name: str, *, backend: str, scope: str) -> int:
    values = {int(cell[field_name]) for cell in cells}
    if len(values) != 1:
        raise AdapterFactoryError(
            f"{backend} backend requires one {field_name} for {scope}; discovered={sorted(values)}"
        )
    return next(iter(values))


def build_backend_plan(
    *,
    manifest: Mapping[str, Any],
    task: str,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized_manifest = validate_allocation_manifest(manifest)
    normalized_config = _normalize_config(config)
    backend = str(normalized_manifest["backend"])
    schema = build_cell_schema(target_modules=normalized_config["target_modules"])
    cells_by_id = _validate_manifest_binding(
        normalized_manifest,
        schema=schema,
        target_modules=normalized_config["target_modules"],
    )
    ordered_cell_ids = [cell.cell_id for cell in schema.cells]
    layer_groups = list(normalized_manifest["layer_groups"])
    timestep_bands = list(normalized_manifest["timestep_bands"])
    layer_group_module_map = _schema_layer_group_module_map(
        schema,
        target_modules=normalized_config["target_modules"],
    )
    timestep_band_routes = build_timestep_band_routes(timestep_bands, schema=schema)
    timestep_band_metadata = _build_timestep_band_metadata(
        cells_by_id,
        timestep_bands=timestep_bands,
    )

    adapter_banks: list[dict[str, Any]] = []
    routing: dict[str, Any]

    if backend == "uniform":
        cells = [cells_by_id[cell_id] for cell_id in ordered_cell_ids]
        rank = _single_value(cells, "rank", backend=backend, scope="all cells")
        alpha = _single_value(cells, "alpha", backend=backend, scope="all cells")
        adapter_names = {str(cell["adapter_name"]) for cell in cells}
        if len(adapter_names) != 1:
            raise AdapterFactoryError(
                f"uniform backend requires one adapter_name; discovered={sorted(adapter_names)}"
            )
        module_names, rank_pattern, alpha_pattern = _module_patterns_for_cell_ids(
            ordered_cell_ids,
            cells_by_id=cells_by_id,
            layer_group_module_map=layer_group_module_map,
        )
        adapter_name = next(iter(adapter_names))
        adapter_banks.append(
            {
                "adapter_name": adapter_name,
                "timestep_band": None,
                "cell_ids": ordered_cell_ids,
                "module_names": module_names,
                "rank": rank,
                "alpha": alpha,
                "rank_pattern": rank_pattern,
                "alpha_pattern": alpha_pattern,
            }
        )
        routing = {
            "mode": "static",
            "default_adapter_name": adapter_name,
        }

    elif backend == "layer_only":
        cell_ids_by_group = _group_cell_ids(
            cells_by_id,
            field_name="layer_group",
            ordered_names=layer_groups,
        )
        collapsed_cell_ids: list[str] = []
        adapter_names = {str(cell["adapter_name"]) for cell in cells_by_id.values()}
        if len(adapter_names) != 1:
            raise AdapterFactoryError(
                f"layer_only backend requires one adapter_name; discovered={sorted(adapter_names)}"
            )
        for layer_group in layer_groups:
            group_cells = [cells_by_id[cell_id] for cell_id in cell_ids_by_group[layer_group]]
            _single_value(group_cells, "rank", backend=backend, scope=layer_group)
            _single_value(group_cells, "alpha", backend=backend, scope=layer_group)
            collapsed_cell_ids.append(cell_ids_by_group[layer_group][0])
        module_names, rank_pattern, alpha_pattern = _module_patterns_for_cell_ids(
            collapsed_cell_ids,
            cells_by_id=cells_by_id,
            layer_group_module_map=layer_group_module_map,
        )
        adapter_name = next(iter(adapter_names))
        adapter_banks.append(
            {
                "adapter_name": adapter_name,
                "timestep_band": None,
                "cell_ids": collapsed_cell_ids,
                "module_names": module_names,
                "rank_pattern": rank_pattern,
                "alpha_pattern": alpha_pattern,
            }
        )
        routing = {
            "mode": "static",
            "default_adapter_name": adapter_name,
        }

    elif backend in {"timestep_only", "proposed"}:
        cell_ids_by_band = {
            str(timestep_band): list(timestep_band_metadata[str(timestep_band)]["cell_ids"])
            for timestep_band in timestep_bands
        }
        band_to_adapter: dict[str, str] = {}
        for timestep_band in timestep_bands:
            band_cell_ids = cell_ids_by_band[timestep_band]
            band_cells = [cells_by_id[cell_id] for cell_id in band_cell_ids]
            adapter_names = {str(cell["adapter_name"]) for cell in band_cells}
            if len(adapter_names) != 1:
                raise AdapterFactoryError(
                    f"{backend} backend requires one adapter_name for {timestep_band!r}; discovered={sorted(adapter_names)}"
                )
            adapter_name = next(iter(adapter_names))
            band_to_adapter[timestep_band] = adapter_name
            module_names, rank_pattern, alpha_pattern = _module_patterns_for_cell_ids(
                band_cell_ids,
                cells_by_id=cells_by_id,
                layer_group_module_map=layer_group_module_map,
            )
            bank: dict[str, Any] = {
                "adapter_name": adapter_name,
                "timestep_band": timestep_band,
                "cell_ids": band_cell_ids,
                "module_names": module_names,
                "rank_pattern": rank_pattern,
                "alpha_pattern": alpha_pattern,
                "has_trainable_params": bool(timestep_band_metadata[timestep_band]["has_trainable_params"]),
                "noop": bool(timestep_band_metadata[timestep_band]["noop"]),
            }
            if backend == "timestep_only":
                bank["rank"] = _single_value(band_cells, "rank", backend=backend, scope=timestep_band)
                bank["alpha"] = _single_value(band_cells, "alpha", backend=backend, scope=timestep_band)
            adapter_banks.append(bank)
        routing_table = build_adapter_routing_table(timestep_band_routes, band_to_adapter=band_to_adapter)
        routing = {
            "mode": "timestep_band",
            "table": routing_table,
            "summary": describe_routing_table(routing_table),
        }

    else:
        raise AdapterFactoryError(f"Unsupported backend {backend!r}")

    return {
        "schema_version": "1.0",
        "task": str(task),
        "backend": backend,
        "use_rslora": bool(normalized_config["use_rslora"]),
        "target_modules": list(normalized_config["target_modules"]),
        "rank_budget_total": int(normalized_manifest["rank_budget_total"]),
        "layer_groups": layer_groups,
        "timestep_bands": timestep_bands,
        "timestep_band_routes": timestep_band_routes,
        "timestep_band_metadata": timestep_band_metadata,
        "cell_ids": ordered_cell_ids,
        "cell_configs": {
            cell_id: dict(cells_by_id[cell_id])
            for cell_id in ordered_cell_ids
        },
        "adapter_banks": adapter_banks,
        "routing": routing,
    }


def _mode_int(values: Sequence[int], *, context: str) -> int:
    normalized = [int(value) for value in values]
    if not normalized:
        raise AdapterFactoryError(f"{context} must not be empty")
    return int(Counter(normalized).most_common(1)[0][0])


def build_concrete_adapter_specs(
    plan: Mapping[str, Any],
    *,
    unet: Any,
) -> list[dict[str, Any]]:
    schema = build_cell_schema(target_modules=plan["target_modules"])
    matches_by_cell = validate_cell_targets(schema, unet, plan["target_modules"])
    cell_configs = {
        str(cell_id): dict(cell)
        for cell_id, cell in dict(plan["cell_configs"]).items()
    }

    concrete_specs: list[dict[str, Any]] = []
    for bank in plan["adapter_banks"]:
        rank_by_module: dict[str, int] = {}
        alpha_by_module: dict[str, int] = {}
        for cell_id in bank["cell_ids"]:
            cell = cell_configs[str(cell_id)]
            for module_name in matches_by_cell[str(cell_id)]:
                rank_by_module[module_name] = int(cell["rank"])
                alpha_by_module[module_name] = int(cell["alpha"])

        positive_modules = sorted(
            module_name
            for module_name, rank in rank_by_module.items()
            if int(rank) > 0
        )
        if not positive_modules:
            concrete_specs.append(
                {
                    "adapter_name": str(bank["adapter_name"]),
                    "timestep_band": bank["timestep_band"],
                    "cell_ids": list(bank["cell_ids"]),
                    "target_modules": [],
                    "rank": 0,
                    "alpha": 0,
                    "rank_pattern": {},
                    "alpha_pattern": {},
                    "noop": True,
                }
            )
            continue

        rank = _mode_int(
            [rank_by_module[module_name] for module_name in positive_modules],
            context=f"{bank['adapter_name']} rank values",
        )
        alpha = _mode_int(
            [alpha_by_module[module_name] for module_name in positive_modules],
            context=f"{bank['adapter_name']} alpha values",
        )
        rank_pattern = {
            module_name: int(rank_by_module[module_name])
            for module_name in positive_modules
            if int(rank_by_module[module_name]) != rank
        }
        alpha_pattern = {
            module_name: int(alpha_by_module[module_name])
            for module_name in positive_modules
            if int(alpha_by_module[module_name]) != alpha
        }
        concrete_specs.append(
            {
                "adapter_name": str(bank["adapter_name"]),
                "timestep_band": bank["timestep_band"],
                "cell_ids": list(bank["cell_ids"]),
                "target_modules": positive_modules,
                "rank": rank,
                "alpha": alpha,
                "rank_pattern": rank_pattern,
                "alpha_pattern": alpha_pattern,
                "noop": False,
            }
        )

    return concrete_specs


def build_backend_adapter_plan(
    backend: str,
    manifest: Mapping[str, Any] | str | Path,
    *,
    use_rslora: bool = True,
    target_modules: Sequence[str] = ALLOWED_TARGET_MODULES,
) -> dict[str, Any]:
    normalized_manifest = (
        load_allocation_manifest(manifest)
        if isinstance(manifest, (str, Path))
        else validate_allocation_manifest(manifest)
    )
    if str(backend).strip() != str(normalized_manifest["backend"]):
        raise AdapterFactoryError(
            f"Requested backend {backend!r} does not match manifest backend {normalized_manifest['backend']!r}"
        )
    return build_backend_plan(
        manifest=normalized_manifest,
        task="unknown",
        config={
            "training": {
                "use_rslora": bool(use_rslora),
                "target_modules": list(target_modules),
            }
        },
    )


def write_backend_plan(plan: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_adapter_plan(plan: Mapping[str, Any], path: str | Path) -> Path:
    return write_backend_plan(plan, path)


__all__ = [
    "ALLOWED_TARGET_MODULES",
    "AdapterFactoryError",
    "EXPECTED_BACKENDS",
    "REQUIRED_PROBE_ROW_KEYS",
    "REQUIRED_PROBE_SUMMARY_KEYS",
    "REQUIRED_SURROGATE_SUMMARY_KEYS",
    "build_backend_adapter_plan",
    "build_concrete_adapter_specs",
    "build_backend_plan",
    "load_actual_contract_binding",
    "write_adapter_plan",
    "write_backend_plan",
]
