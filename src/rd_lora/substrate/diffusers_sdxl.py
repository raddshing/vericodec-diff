from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shlex
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from PIL import Image, ImageOps


DEFAULT_IMAGE_SIZE = 1024
DEFAULT_OFFICIAL_DIFFUSERS_SCRIPT = (
    "baselines/external/huggingface_diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py"
)
DEFAULT_OFFICIAL_DREAMBOOTH_LORA_SCRIPT = (
    "baselines/external/huggingface_diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py"
)
DEFAULT_TASKS_CONFIG = "configs/rdlora_tasks.yaml"
PILOT_MANIFEST_COLUMNS = (
    "task_id",
    "file_name",
    "source_path",
    "prepared_path",
    "caption",
    "width",
    "height",
)
_RESAMPLING = getattr(Image, "Resampling", Image)


class PilotTaskValidationError(ValueError):
    """Raised when the pilot dataset task config is malformed."""


class DiffusersHarnessValidationError(ValueError):
    """Raised when the diffusers SDXL LoRA launch plan is invalid."""


class DiffusersExecutionError(RuntimeError):
    """Raised when the local DreamBooth SDXL execution substrate is unavailable or invalid."""


def validate_official_sdxl_dreambooth_script(
    repo_root: Path,
    raw_path: str | Path | None = None,
) -> Path:
    script_path = resolve_path(
        repo_root,
        raw_path or DEFAULT_OFFICIAL_DREAMBOOTH_LORA_SCRIPT,
    ).resolve()
    if not script_path.is_file():
        raise DiffusersExecutionError(f"Official SDXL DreamBooth LoRA script not found: {script_path}")
    return script_path


def load_official_sdxl_dreambooth_module(script_path: str | Path) -> Any:
    path = Path(script_path).expanduser().resolve()
    if not path.is_file():
        raise DiffusersExecutionError(f"Official SDXL DreamBooth LoRA script not found: {path}")
    spec = importlib.util.spec_from_file_location("rd_lora_official_dreambooth_sdxl", path)
    if spec is None or spec.loader is None:
        raise DiffusersExecutionError(f"Unable to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_sdxl_components(
    *,
    pretrained_model_name_or_path: str,
    revision: str | None,
    variant: str | None,
    pretrained_vae_model_name_or_path: str | None,
    script_path: str | Path,
) -> dict[str, Any]:
    module = load_official_sdxl_dreambooth_module(script_path)

    tokenizer = module.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
        use_fast=False,
    )
    tokenizer_2 = module.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=revision,
        use_fast=False,
    )
    text_encoder_cls = module.import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path,
        revision,
    )
    text_encoder_cls_2 = module.import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path,
        revision,
        subfolder="text_encoder_2",
    )
    scheduler_type = module.determine_scheduler_type(pretrained_model_name_or_path, revision)
    if "EDM" in scheduler_type:
        scheduler = module.EDMEulerScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )
    else:
        scheduler = module.DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )

    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
    )
    text_encoder_2 = text_encoder_cls_2.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
    )
    vae_source = pretrained_model_name_or_path if pretrained_vae_model_name_or_path in (None, "") else str(
        pretrained_vae_model_name_or_path
    )
    vae = module.AutoencoderKL.from_pretrained(
        vae_source,
        subfolder="vae" if pretrained_vae_model_name_or_path in (None, "") else None,
        revision=revision,
        variant=variant,
    )
    unet = module.UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
        variant=variant,
    )
    return {
        "module": module,
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "vae": vae,
        "unet": unet,
        "scheduler": scheduler,
        "scheduler_type": scheduler_type,
    }


def describe_loaded_sdxl_components(components: Mapping[str, Any]) -> list[str]:
    ordered = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "vae",
        "unet",
        "scheduler",
    ]
    return [name for name in ordered if name in components]


def resolve_path(repo_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def deep_update(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in dict(overrides).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must parse to a mapping")
    return dict(data)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_shell_command(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n")
        handle.write(f"{render_shell_command(command)}\n")


def render_shell_command(command: Sequence[str]) -> str:
    return shlex.join([str(item) for item in command])


def _slug_token(raw_value: str, *, fallback: str) -> str:
    characters: list[str] = []
    previous_underscore = False
    for character in raw_value.lower():
        if character.isalnum():
            characters.append(character)
            previous_underscore = False
            continue
        if not previous_underscore:
            characters.append("_")
            previous_underscore = True
    token = "".join(characters).strip("_")
    return token or fallback


def _validate_repo_local_path(repo_root: Path, path: Path, *, name: str) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{name} must resolve inside repo_root for stable local paths") from exc


def _validate_single_path_token(raw_value: str, *, name: str) -> str:
    token = raw_value.strip()
    if not token:
        raise ValueError(f"{name} must be a non-empty path token")
    token_path = Path(token)
    if len(token_path.parts) != 1 or token_path.name != token:
        raise ValueError(f"{name} must be a single path component")
    if token in {".", ".."}:
        raise ValueError(f"{name} must not be '.' or '..'")
    return token


def _parse_positive_int(raw_value: Any, *, name: str, allow_none: bool = False) -> int | None:
    if raw_value in (None, ""):
        if allow_none:
            return None
        raise ValueError(f"{name} is required")
    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_non_negative_int(raw_value: Any, *, name: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def set_nested_value(container: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = [item.strip() for item in dotted_key.split(".") if item.strip()]
    if not keys:
        raise ValueError("Override key must be a non-empty dotted path")
    cursor = container
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            existing = {}
            cursor[key] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Override key {dotted_key!r} collides with non-mapping path {key!r}")
        cursor = existing
    cursor[keys[-1]] = value


def build_cli_overrides(items: Sequence[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"Override {item!r} must use key=value form")
        key, raw_value = item.split("=", 1)
        parsed_value = yaml.safe_load(raw_value)
        set_nested_value(overrides, key.strip(), parsed_value)
    return overrides


def default_vanilla_lora_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "official_script": DEFAULT_OFFICIAL_DIFFUSERS_SCRIPT,
            "tasks_config": DEFAULT_TASKS_CONFIG,
            "prepared_data_root": "outputs/rd_lora/pilot_data",
            "run_root": "outputs/rd_lora/vanilla",
            "smoke_root": "outputs/rd_lora/smoke",
            "cache_dir": None,
        },
        "accelerate": {
            "executable": "accelerate",
            "launch": {
                "config_file": None,
                "num_processes": 1,
            },
        },
        "model": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "pretrained_vae_model_name_or_path": None,
            "revision": None,
            "variant": None,
        },
        "dataset": {
            "image_column": "image",
            "caption_column": "text",
        },
        "run": {
            "name": "vanilla_lora",
            "validation_prompt": None,
        },
        "training": {
            "resolution": DEFAULT_IMAGE_SIZE,
            "center_crop": True,
            "random_flip": False,
            "train_text_encoder": False,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_train_steps": 100,
            "checkpointing_steps": 25,
            "checkpoints_total_limit": 2,
            "learning_rate": 1.0e-4,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "gradient_checkpointing": False,
            "scale_lr": False,
            "allow_tf32": False,
            "rank": 8,
            "seed": 20260408,
            "report_to": "tensorboard",
            "mixed_precision": "fp16",
            "num_validation_images": 1,
            "validation_epochs": 1,
            "dataloader_num_workers": 0,
            "max_train_samples": None,
            "use_8bit_adam": False,
            "enable_xformers_memory_efficient_attention": False,
        },
        "smoke": {
            "task_id": "smoke_portrait",
            "run_name": "smoke",
            "validation_prompt": None,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_train_steps": 1,
            "checkpointing_steps": 1,
            "checkpoints_total_limit": 1,
            "gradient_checkpointing": False,
            "mixed_precision": "no",
            "num_validation_images": 1,
            "dataloader_num_workers": 0,
            "report_to": "tensorboard",
        },
    }


def _resolve_cache_dir(repo_root: Path, raw_value: Any) -> str | None:
    if raw_value in (None, ""):
        return None
    cache_dir = resolve_path(repo_root, str(raw_value)).resolve()
    _validate_repo_local_path(repo_root, cache_dir, name="paths.cache_dir")
    return str(cache_dir)


def _resolve_training_config(training: Mapping[str, Any]) -> dict[str, Any]:
    resolution = _parse_positive_int(training.get("resolution"), name="training.resolution")
    if resolution != DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"RD-LoRA-Diff is locked to {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE} for the SDXL LoRA harness"
        )

    mixed_precision = str(training.get("mixed_precision", "fp16")).strip()
    if mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision must be one of {'no', 'fp16', 'bf16'}")

    report_to = str(training.get("report_to", "tensorboard")).strip()
    if not report_to:
        raise ValueError("training.report_to must be non-empty")

    lr_scheduler = str(training.get("lr_scheduler", "constant")).strip()
    if not lr_scheduler:
        raise ValueError("training.lr_scheduler must be non-empty")

    max_train_samples = training.get("max_train_samples")
    if max_train_samples not in (None, ""):
        max_train_samples = _parse_positive_int(
            max_train_samples,
            name="training.max_train_samples",
        )
    else:
        max_train_samples = None

    seed = training.get("seed")
    if seed in (None, ""):
        resolved_seed = None
    else:
        resolved_seed = int(seed)

    return {
        "resolution": resolution,
        "center_crop": bool(training.get("center_crop", True)),
        "random_flip": bool(training.get("random_flip", False)),
        "train_text_encoder": bool(training.get("train_text_encoder", False)),
        "train_batch_size": _parse_positive_int(
            training.get("train_batch_size"),
            name="training.train_batch_size",
        ),
        "gradient_accumulation_steps": _parse_positive_int(
            training.get("gradient_accumulation_steps"),
            name="training.gradient_accumulation_steps",
        ),
        "max_train_steps": _parse_positive_int(
            training.get("max_train_steps"),
            name="training.max_train_steps",
        ),
        "checkpointing_steps": _parse_positive_int(
            training.get("checkpointing_steps"),
            name="training.checkpointing_steps",
        ),
        "checkpoints_total_limit": _parse_positive_int(
            training.get("checkpoints_total_limit"),
            name="training.checkpoints_total_limit",
            allow_none=True,
        ),
        "learning_rate": float(training.get("learning_rate", 1.0e-4)),
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": _parse_non_negative_int(
            training.get("lr_warmup_steps", 0),
            name="training.lr_warmup_steps",
        ),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", False)),
        "scale_lr": bool(training.get("scale_lr", False)),
        "allow_tf32": bool(training.get("allow_tf32", False)),
        "rank": _parse_positive_int(training.get("rank"), name="training.rank"),
        "seed": resolved_seed,
        "report_to": report_to,
        "mixed_precision": mixed_precision,
        "num_validation_images": _parse_positive_int(
            training.get("num_validation_images"),
            name="training.num_validation_images",
        ),
        "validation_epochs": _parse_positive_int(
            training.get("validation_epochs"),
            name="training.validation_epochs",
        ),
        "dataloader_num_workers": _parse_non_negative_int(
            training.get("dataloader_num_workers", 0),
            name="training.dataloader_num_workers",
        ),
        "max_train_samples": max_train_samples,
        "use_8bit_adam": bool(training.get("use_8bit_adam", False)),
        "enable_xformers_memory_efficient_attention": bool(
            training.get("enable_xformers_memory_efficient_attention", False)
        ),
    }


def resolve_vanilla_lora_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_vanilla_lora_config(), raw_config)
    paths = dict(config.get("paths", {}))
    accelerate = dict(config.get("accelerate", {}))
    model = dict(config.get("model", {}))
    dataset = dict(config.get("dataset", {}))
    run = dict(config.get("run", {}))
    smoke = dict(config.get("smoke", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    official_script = resolve_path(
        resolved_repo_root,
        str(paths.get("official_script", DEFAULT_OFFICIAL_DIFFUSERS_SCRIPT)),
    ).resolve()
    tasks_config = resolve_path(
        resolved_repo_root,
        str(paths.get("tasks_config", DEFAULT_TASKS_CONFIG)),
    ).resolve()
    prepared_data_root = resolve_path(
        resolved_repo_root,
        str(paths.get("prepared_data_root", "outputs/rd_lora/pilot_data")),
    ).resolve()
    run_root = resolve_path(
        resolved_repo_root,
        str(paths.get("run_root", "outputs/rd_lora/vanilla")),
    ).resolve()
    smoke_root = resolve_path(
        resolved_repo_root,
        str(paths.get("smoke_root", "outputs/rd_lora/smoke")),
    ).resolve()

    if not official_script.is_file():
        raise FileNotFoundError(f"Official diffusers SDXL LoRA script not found: {official_script}")
    if not tasks_config.is_file():
        raise FileNotFoundError(f"RD-LoRA task config not found: {tasks_config}")

    for name, path in {
        "paths.prepared_data_root": prepared_data_root,
        "paths.run_root": run_root,
        "paths.smoke_root": smoke_root,
    }.items():
        _validate_repo_local_path(resolved_repo_root, path, name=name)

    accelerate_executable = str(accelerate.get("executable", "accelerate")).strip() or "accelerate"
    launch = dict(accelerate.get("launch", {}))
    launch_config_file = launch.get("config_file")
    resolved_launch_config = None
    if launch_config_file not in (None, ""):
        resolved_launch_config = resolve_path(resolved_repo_root, str(launch_config_file)).resolve()
        if not resolved_launch_config.is_file():
            raise FileNotFoundError(f"accelerate launch config not found: {resolved_launch_config}")
    num_processes = _parse_positive_int(
        launch.get("num_processes", 1),
        name="accelerate.launch.num_processes",
    )

    pretrained_model_name_or_path = str(model.get("pretrained_model_name_or_path") or "").strip()
    if not pretrained_model_name_or_path:
        raise ValueError("model.pretrained_model_name_or_path is required")

    image_column = str(dataset.get("image_column", "image")).strip() or "image"
    caption_column = str(dataset.get("caption_column", "text")).strip() or "text"
    if image_column != "image":
        raise ValueError("dataset.image_column must remain 'image' for diffusers imagefolder")
    if not caption_column:
        raise ValueError("dataset.caption_column must be non-empty")

    run_name = _validate_single_path_token(str(run.get("name", "vanilla_lora")), name="run.name")
    run_validation_prompt = run.get("validation_prompt")
    if run_validation_prompt not in (None, ""):
        run_validation_prompt = str(run_validation_prompt)
    else:
        run_validation_prompt = None

    smoke_task_id = str(smoke.get("task_id", "smoke_portrait")).strip() or "smoke_portrait"
    smoke_run_name = _validate_single_path_token(
        str(smoke.get("run_name", "smoke")),
        name="smoke.run_name",
    )
    smoke_validation_prompt = smoke.get("validation_prompt")
    if smoke_validation_prompt not in (None, ""):
        smoke_validation_prompt = str(smoke_validation_prompt)
    else:
        smoke_validation_prompt = None

    training = _resolve_training_config(dict(config.get("training", {})))

    resolved_smoke = {
        "task_id": smoke_task_id,
        "run_name": smoke_run_name,
        "validation_prompt": smoke_validation_prompt,
    }
    for key in training.keys():
        if key in smoke:
            resolved_smoke[key] = smoke.get(key)

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "official_script": str(official_script),
            "tasks_config": str(tasks_config),
            "prepared_data_root": str(prepared_data_root),
            "run_root": str(run_root),
            "smoke_root": str(smoke_root),
            "cache_dir": _resolve_cache_dir(resolved_repo_root, paths.get("cache_dir")),
        },
        "accelerate": {
            "executable": accelerate_executable,
            "launch": {
                "config_file": str(resolved_launch_config) if resolved_launch_config is not None else None,
                "num_processes": num_processes,
            },
        },
        "model": {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "pretrained_vae_model_name_or_path": (
                None
                if model.get("pretrained_vae_model_name_or_path") in (None, "")
                else str(model.get("pretrained_vae_model_name_or_path"))
            ),
            "revision": None if model.get("revision") in (None, "") else str(model.get("revision")),
            "variant": None if model.get("variant") in (None, "") else str(model.get("variant")),
        },
        "dataset": {
            "image_column": image_column,
            "caption_column": caption_column,
        },
        "run": {
            "name": run_name,
            "validation_prompt": run_validation_prompt,
        },
        "training": training,
        "smoke": resolved_smoke,
    }


def load_rdlora_tasks(path: Path | str) -> dict[str, Any]:
    tasks_path = Path(path)
    data = load_yaml_mapping(tasks_path)
    defaults = data.get("defaults") or {}
    tasks = data.get("tasks") or {}
    if not isinstance(defaults, dict):
        raise PilotTaskValidationError(f"{tasks_path}: defaults must be a mapping")
    if not isinstance(tasks, dict):
        raise PilotTaskValidationError(f"{tasks_path}: tasks must be a mapping")
    if not tasks:
        raise PilotTaskValidationError(f"{tasks_path}: tasks must define at least one pilot task")
    return {
        "version": int(data.get("version", 1)),
        "defaults": deepcopy(defaults),
        "tasks": deepcopy(tasks),
        "path": str(tasks_path),
    }


def _resolve_source_image_path(repo_root: Path, source_root: Path | None, raw_path: str) -> Path:
    raw = Path(raw_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    if source_root is not None:
        return (source_root / raw).resolve()
    return (repo_root / raw).resolve()


def validate_pilot_task(repo_root: Path, tasks_bundle: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    task_mapping = tasks_bundle.get("tasks") or {}
    if task_id not in task_mapping:
        available = ", ".join(sorted(task_mapping))
        raise PilotTaskValidationError(f"Task {task_id!r} not found in {tasks_bundle['path']}; available: {available}")

    defaults = dict(tasks_bundle.get("defaults") or {})
    raw_task = task_mapping[task_id]
    if not isinstance(raw_task, Mapping):
        raise PilotTaskValidationError(f"Task {task_id!r} must map to a task definition")

    concept_name = str(raw_task.get("concept_name") or "").strip()
    if not concept_name:
        raise PilotTaskValidationError(f"Task {task_id!r} must define concept_name")

    validation_prompt = str(raw_task.get("validation_prompt") or "").strip()
    if not validation_prompt:
        raise PilotTaskValidationError(f"Task {task_id!r} must define validation_prompt")

    image_size = int(raw_task.get("image_size") or defaults.get("image_size") or DEFAULT_IMAGE_SIZE)
    if image_size != DEFAULT_IMAGE_SIZE:
        raise PilotTaskValidationError(
            f"Task {task_id!r} must stay at {DEFAULT_IMAGE_SIZE} image_size for the SDXL substrate"
        )

    image_column = str(raw_task.get("image_column") or defaults.get("image_column") or "image").strip() or "image"
    caption_column = str(raw_task.get("caption_column") or defaults.get("caption_column") or "text").strip() or "text"
    if image_column != "image":
        raise PilotTaskValidationError(
            f"Task {task_id!r} must use image_column='image' for diffusers imagefolder"
        )
    if not caption_column:
        raise PilotTaskValidationError(f"Task {task_id!r} must define a non-empty caption_column")

    source_root = None
    source_root_raw = raw_task.get("source_root", defaults.get("source_root"))
    if source_root_raw not in (None, ""):
        source_root = resolve_path(repo_root, str(source_root_raw)).resolve()
        if not source_root.exists():
            raise PilotTaskValidationError(f"Task {task_id!r} source_root not found: {source_root}")

    prepared_root_raw = raw_task.get("prepared_root", defaults.get("prepared_root"))

    raw_records = raw_task.get("train_records")
    if not isinstance(raw_records, list) or not raw_records:
        raise PilotTaskValidationError(f"Task {task_id!r} must define a non-empty train_records list")

    validated_records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records, start=1):
        if not isinstance(raw_record, Mapping):
            raise PilotTaskValidationError(f"Task {task_id!r} record {index} must be a mapping")
        raw_image = str(raw_record.get("image") or "").strip()
        if not raw_image:
            raise PilotTaskValidationError(f"Task {task_id!r} record {index} must define image")
        caption = str(raw_record.get("caption") or "").strip()
        if not caption:
            raise PilotTaskValidationError(f"Task {task_id!r} record {index} must define caption")

        source_path = _resolve_source_image_path(repo_root, source_root, raw_image)
        if not source_path.is_file():
            raise PilotTaskValidationError(
                f"Task {task_id!r} record {index} source image not found: {source_path}"
            )

        validated_records.append(
            {
                "index": index,
                "image": raw_image,
                "caption": caption,
                "source_path": str(source_path),
            }
        )

    return {
        "task_id": task_id,
        "concept_name": concept_name,
        "validation_prompt": validation_prompt,
        "image_size": image_size,
        "image_column": image_column,
        "caption_column": caption_column,
        "prepared_root_raw": prepared_root_raw,
        "records": validated_records,
    }


def resolve_task_layout(
    config: Mapping[str, Any],
    tasks_bundle: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    task = validate_pilot_task(repo_root, tasks_bundle, task_id)
    prepared_root_raw = (
        task["prepared_root_raw"]
        if task["prepared_root_raw"] not in (None, "")
        else config["paths"]["prepared_data_root"]
    )
    prepared_root = resolve_path(repo_root, str(prepared_root_raw)).resolve()
    _validate_repo_local_path(repo_root, prepared_root, name="pilot prepared_root")

    task_root = (prepared_root / task_id).resolve()
    train_data_dir = (task_root / "imagefolder").resolve()
    manifest_path = (task_root / "pilot_manifest.csv").resolve()
    metadata_path = (train_data_dir / "metadata.jsonl").resolve()
    _validate_repo_local_path(repo_root, task_root, name="pilot task_root")

    return {
        "task": task,
        "task_root": str(task_root),
        "train_data_dir": str(train_data_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
    }


def _normalize_image(image: Image.Image, *, image_size: int) -> Image.Image:
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(
        prepared,
        (image_size, image_size),
        method=_RESAMPLING.LANCZOS,
        centering=(0.5, 0.5),
    )


def _build_prepared_file_name(task_id: str, source_path: Path, index: int) -> str:
    stem = _slug_token(source_path.stem, fallback="image")
    digest = hashlib.blake2b(
        f"{task_id}|{source_path}".encode("utf-8"),
        digest_size=4,
    ).hexdigest()
    return f"{index:04d}_{stem}_{digest}.png"


def write_pilot_manifest(rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in PILOT_MANIFEST_COLUMNS})


def write_metadata_jsonl(
    rows: Sequence[Mapping[str, Any]],
    metadata_path: Path,
    *,
    caption_column: str,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "file_name": row["file_name"],
                caption_column: row["caption"],
            }
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")


def load_pilot_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != PILOT_MANIFEST_COLUMNS:
            raise PilotTaskValidationError(
                f"{path.name}: expected columns {PILOT_MANIFEST_COLUMNS}, found {fieldnames}"
            )
        return [dict(row) for row in reader]


def validate_pilot_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    verify_files: bool = False,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, Any]:
    rows = load_pilot_manifest(manifest_path)
    if not rows:
        raise PilotTaskValidationError(f"{manifest_path.name}: manifest must contain at least one row")

    seen_files: set[str] = set()
    task_counts: dict[str, int] = {}
    for row_number, row in enumerate(rows, start=2):
        for column in PILOT_MANIFEST_COLUMNS:
            value = str(row.get(column) or "").strip()
            if not value:
                raise PilotTaskValidationError(f"{manifest_path.name}: row {row_number} has empty {column}")
        file_name = row["file_name"]
        if file_name in seen_files:
            raise PilotTaskValidationError(
                f"{manifest_path.name}: duplicate file_name {file_name!r} at row {row_number}"
            )
        seen_files.add(file_name)
        task_counts[row["task_id"]] = task_counts.get(row["task_id"], 0) + 1

        if verify_files:
            prepared_path = resolve_path(repo_root, row["prepared_path"]).resolve()
            if not prepared_path.is_file():
                raise PilotTaskValidationError(
                    f"{manifest_path.name}: prepared image missing at row {row_number}: {prepared_path}"
                )
            with Image.open(prepared_path) as image:
                if image.size != (image_size, image_size):
                    raise PilotTaskValidationError(
                        f"{manifest_path.name}: row {row_number} prepared image must be "
                        f"{image_size}x{image_size}, found {image.size}"
                    )

    return {
        "total": len(rows),
        "task_counts": task_counts,
        "manifest_path": display_path(manifest_path, repo_root),
    }


def prepare_pilot_imagefolder(
    config: Mapping[str, Any],
    tasks_bundle: Mapping[str, Any],
    task_id: str,
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    layout = resolve_task_layout(config, tasks_bundle, task_id)
    task = layout["task"]
    task_root = Path(layout["task_root"])
    train_data_dir = Path(layout["train_data_dir"])
    manifest_path = Path(layout["manifest_path"])
    metadata_path = Path(layout["metadata_path"])

    train_data_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen_file_names: set[str] = set()
    for record in task["records"]:
        source_path = Path(record["source_path"])
        file_name = _build_prepared_file_name(task_id, source_path, int(record["index"]))
        if file_name in seen_file_names:
            raise PilotTaskValidationError(
                f"Task {task_id!r} produced duplicate prepared file_name {file_name!r}"
            )
        seen_file_names.add(file_name)
        prepared_path = train_data_dir / file_name

        with Image.open(source_path) as image:
            normalized = _normalize_image(image, image_size=int(task["image_size"]))
            width, height = normalized.size
            normalized.save(prepared_path, format="PNG")

        rows.append(
            {
                "task_id": task_id,
                "file_name": file_name,
                "source_path": display_path(source_path, repo_root),
                "prepared_path": display_path(prepared_path, repo_root),
                "caption": record["caption"],
                "width": str(width),
                "height": str(height),
            }
        )

    write_pilot_manifest(rows, manifest_path)
    write_metadata_jsonl(rows, metadata_path, caption_column=str(task["caption_column"]))
    manifest_summary = validate_pilot_manifest(
        manifest_path,
        repo_root=repo_root,
        verify_files=True,
        image_size=int(task["image_size"]),
    )

    return {
        "task_id": task_id,
        "concept_name": task["concept_name"],
        "validation_prompt": task["validation_prompt"],
        "caption_column": task["caption_column"],
        "image_column": task["image_column"],
        "task_root": str(task_root),
        "train_data_dir": str(train_data_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "image_count": len(rows),
        "manifest_summary": manifest_summary,
    }


def resolve_training_options(config: Mapping[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    training = deepcopy(dict(config["training"]))
    if smoke:
        smoke_config = dict(config.get("smoke", {}))
        for key in tuple(training.keys()):
            if smoke_config.get(key) not in (None, ""):
                training[key] = smoke_config[key]
    return _resolve_training_config(training)


def _append_optional_flag(command: list[str], flag: str, value: Any) -> None:
    if value in (None, ""):
        return
    command.extend([flag, str(value)])


def build_diffusers_sdxl_lora_command(
    config: Mapping[str, Any],
    *,
    task_id: str,
    run_name: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    tasks_bundle = load_rdlora_tasks(config["paths"]["tasks_config"])
    layout = resolve_task_layout(config, tasks_bundle, task_id)
    task = layout["task"]

    train_data_dir = Path(layout["train_data_dir"])
    metadata_path = Path(layout["metadata_path"])
    manifest_path = Path(layout["manifest_path"])
    if not metadata_path.is_file():
        raise DiffusersHarnessValidationError(
            f"Prepared metadata.jsonl is missing for task {task_id!r}: {metadata_path}. "
            "Run scripts/prepare_rdlora_pilot_data.py first."
        )
    if not manifest_path.is_file():
        raise DiffusersHarnessValidationError(
            f"Prepared pilot_manifest.csv is missing for task {task_id!r}: {manifest_path}. "
            "Run scripts/prepare_rdlora_pilot_data.py first."
        )

    training = resolve_training_options(config, smoke=smoke)
    requested_run_name = (
        run_name
        if run_name is not None
        else str(config["smoke"]["run_name"] if smoke else config["run"]["name"])
    )
    resolved_run_name = _validate_single_path_token(requested_run_name, name="run_name")
    output_root = Path(config["paths"]["smoke_root"] if smoke else config["paths"]["run_root"])
    output_dir = (output_root / task_id / resolved_run_name).resolve()
    logging_dir = (output_dir / "logs").resolve()
    _validate_repo_local_path(repo_root, output_dir, name="output_dir")

    validation_prompt = (
        config["smoke"].get("validation_prompt")
        if smoke and config["smoke"].get("validation_prompt") not in (None, "")
        else config["run"].get("validation_prompt")
    )
    if validation_prompt in (None, ""):
        validation_prompt = task["validation_prompt"]

    official_script = Path(config["paths"]["official_script"])
    launch = dict(config["accelerate"]["launch"])
    command: list[str] = [str(config["accelerate"]["executable"]), "launch"]
    _append_optional_flag(command, "--config_file", launch.get("config_file"))
    _append_optional_flag(command, "--num_processes", launch.get("num_processes"))
    command.append(str(official_script))
    command.extend(
        [
            "--pretrained_model_name_or_path",
            str(config["model"]["pretrained_model_name_or_path"]),
            "--train_data_dir",
            str(train_data_dir),
            "--image_column",
            str(task["image_column"]),
            "--caption_column",
            str(task["caption_column"]),
            "--validation_prompt",
            str(validation_prompt),
            "--num_validation_images",
            str(training["num_validation_images"]),
            "--validation_epochs",
            str(training["validation_epochs"]),
            "--output_dir",
            str(output_dir),
            "--logging_dir",
            str(logging_dir),
            "--resolution",
            str(training["resolution"]),
            "--train_batch_size",
            str(training["train_batch_size"]),
            "--gradient_accumulation_steps",
            str(training["gradient_accumulation_steps"]),
            "--max_train_steps",
            str(training["max_train_steps"]),
            "--checkpointing_steps",
            str(training["checkpointing_steps"]),
            "--learning_rate",
            str(training["learning_rate"]),
            "--lr_scheduler",
            str(training["lr_scheduler"]),
            "--lr_warmup_steps",
            str(training["lr_warmup_steps"]),
            "--rank",
            str(training["rank"]),
            "--report_to",
            str(training["report_to"]),
            "--mixed_precision",
            str(training["mixed_precision"]),
            "--dataloader_num_workers",
            str(training["dataloader_num_workers"]),
        ]
    )

    _append_optional_flag(
        command,
        "--pretrained_vae_model_name_or_path",
        config["model"]["pretrained_vae_model_name_or_path"],
    )
    _append_optional_flag(command, "--revision", config["model"]["revision"])
    _append_optional_flag(command, "--variant", config["model"]["variant"])
    _append_optional_flag(command, "--checkpoints_total_limit", training["checkpoints_total_limit"])
    _append_optional_flag(command, "--max_train_samples", training["max_train_samples"])
    _append_optional_flag(command, "--cache_dir", config["paths"]["cache_dir"])
    if training["seed"] is not None:
        command.extend(["--seed", str(training["seed"])])
    if training["center_crop"]:
        command.append("--center_crop")
    if training["random_flip"]:
        command.append("--random_flip")
    if training["train_text_encoder"]:
        command.append("--train_text_encoder")
    if training["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")
    if training["scale_lr"]:
        command.append("--scale_lr")
    if training["allow_tf32"]:
        command.append("--allow_tf32")
    if training["use_8bit_adam"]:
        command.append("--use_8bit_adam")
    if training["enable_xformers_memory_efficient_attention"]:
        command.append("--enable_xformers_memory_efficient_attention")

    return {
        "task_id": task_id,
        "task": task,
        "smoke": smoke,
        "run_name": resolved_run_name,
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "train_data_dir": str(train_data_dir),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path),
        "official_script": str(official_script),
        "command": command,
        "manual_command": render_shell_command(command),
        "training": training,
    }


def validate_launch_plan(plan: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    command = [str(item) for item in plan["command"]]
    if len(command) < 4:
        raise DiffusersHarnessValidationError("accelerate launch command is incomplete")
    if command[1] != "launch":
        raise DiffusersHarnessValidationError("generated command must use 'accelerate launch'")

    official_script = Path(plan["official_script"]).resolve()
    if not official_script.is_file():
        raise DiffusersHarnessValidationError(f"Official diffusers script not found: {official_script}")
    if str(official_script) not in command:
        raise DiffusersHarnessValidationError("generated command must target the official diffusers script path")

    train_data_dir = Path(plan["train_data_dir"]).resolve()
    metadata_path = Path(plan["metadata_path"]).resolve()
    if not train_data_dir.is_dir():
        raise DiffusersHarnessValidationError(f"train_data_dir does not exist: {train_data_dir}")
    if not metadata_path.is_file():
        raise DiffusersHarnessValidationError(f"metadata.jsonl does not exist: {metadata_path}")

    output_dir = Path(plan["output_dir"]).resolve()
    _validate_repo_local_path(repo_root, output_dir, name="launch output_dir")

    required_flags = (
        "--pretrained_model_name_or_path",
        "--train_data_dir",
        "--output_dir",
        "--resolution",
        "--train_batch_size",
        "--gradient_accumulation_steps",
        "--max_train_steps",
        "--learning_rate",
        "--rank",
        "--mixed_precision",
    )
    missing_flags = [flag for flag in required_flags if flag not in command]
    if missing_flags:
        raise DiffusersHarnessValidationError(
            f"generated command is missing required flags: {', '.join(missing_flags)}"
        )

    accelerate_executable = command[0]
    accelerate_found = shutil.which(accelerate_executable) is not None if Path(accelerate_executable).name == accelerate_executable else Path(accelerate_executable).exists()

    return {
        "wrapped_official_script": True,
        "official_script": display_path(official_script, repo_root),
        "accelerate_executable": accelerate_executable,
        "accelerate_found": bool(accelerate_found),
        "train_data_dir": display_path(train_data_dir, repo_root),
        "metadata_path": display_path(metadata_path, repo_root),
        "output_dir": display_path(output_dir, repo_root),
        "manual_command": plan["manual_command"],
    }
