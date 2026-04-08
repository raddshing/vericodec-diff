from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rd_lora.substrate.diffusers_sdxl import (
    build_cli_overrides,
    deep_update,
    display_path,
    load_yaml_mapping,
    render_shell_command,
    resolve_path,
    save_json,
    write_shell_command,
)


DEFAULT_REGISTRY_LOCK = "baselines/registry.lock.json"
TLORA_BASELINE_ID = "t_lora"
INTLORA_BASELINE_ID = "int_lora"


class BaselineWrapperValidationError(ValueError):
    """Raised when a baseline wrapper config or launch plan is invalid."""


def _validate_repo_local_path(repo_root: Path, path: Path, *, name: str) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise BaselineWrapperValidationError(
            f"{name} must resolve inside repo_root for deterministic paths"
        ) from exc


def _validate_single_path_token(raw_value: str, *, name: str) -> str:
    token = raw_value.strip()
    if not token:
        raise BaselineWrapperValidationError(f"{name} must be a non-empty path token")
    token_path = Path(token)
    if len(token_path.parts) != 1 or token_path.name != token or token in {".", ".."}:
        raise BaselineWrapperValidationError(f"{name} must be a single path component")
    return token


def _append_optional_flag(command: list[str], flag: str, value: Any) -> None:
    if value in (None, ""):
        return
    command.extend([flag, str(value)])


def _run_git_text(checkout_path: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def load_registry_lock(path: Path | str) -> dict[str, Any]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BaselineWrapperValidationError(f"{registry_path} must parse to a mapping")
    baselines = payload.get("baselines")
    if not isinstance(baselines, list):
        raise BaselineWrapperValidationError(f"{registry_path} must define a baselines list")
    return payload


def get_registry_record(registry_payload: Mapping[str, Any], baseline_id: str) -> dict[str, Any]:
    for item in registry_payload.get("baselines", []):
        if isinstance(item, dict) and item.get("id") == baseline_id:
            return dict(item)
    raise BaselineWrapperValidationError(f"Baseline id {baseline_id!r} is missing from registry")


def inspect_registered_checkout(checkout_path: Path) -> dict[str, str]:
    if not checkout_path.is_dir():
        raise FileNotFoundError(f"Baseline checkout path does not exist: {checkout_path}")
    return {
        "head_commit": _run_git_text(checkout_path, ["rev-parse", "HEAD"]),
        "origin_url": _run_git_text(checkout_path, ["remote", "get-url", "origin"]),
    }


def _load_accelerate_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise BaselineWrapperValidationError(f"{path} must parse to a mapping")
    return dict(data)


def default_tlora_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "registry_lock": DEFAULT_REGISTRY_LOCK,
            "baseline_root": "baselines/external/controlgenai_t-lora",
            "train_script": "baselines/external/controlgenai_t-lora/train.py",
            "output_root": "outputs/baselines/t_lora",
            "accelerate_config": "configs/accelerate/single_gpu_fp16.yaml",
        },
        "accelerate": {
            "executable": "accelerate",
        },
        "model": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "revision": None,
        },
        "dataset": {
            "train_data_dir": "baselines/external/controlgenai_t-lora/dog_example",
            "class_data_dir": None,
            "class_name": "dog",
            "placeholder_token": "sks",
            "validation_prompts": (
                "a {0} lying in the bed#"
                "a {0} swimming#"
                "a {0} dressed as a ballerina"
            ),
            "num_val_imgs_per_prompt": 3,
            "with_prior_preservation": False,
            "prior_loss_weight": 1.0,
            "one_image": "02.jpg",
        },
        "run": {
            "name": "dog_example",
            "wandb_api_key_env": "WANDB_API_KEY",
            "pass_wandb_api_key": False,
        },
        "training": {
            "trainer_type": "ortho_lora",
            "trainer_class": "sdxl_tlora",
            "mixed_precision": "fp16",
            "num_train_epochs": 800,
            "checkpointing_steps": 100,
            "resolution": 1024,
            "train_batch_size": 1,
            "dataloader_num_workers": 1,
            "seed": 0,
            "lora_rank": 64,
            "min_rank": 32,
            "sig_type": "last",
            "alpha_rank_scale": 1.0,
            "learning_rate": 1.0e-4,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 1.0e-4,
            "adam_epsilon": 1.0e-8,
        },
    }


def default_intlora_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "registry_lock": DEFAULT_REGISTRY_LOCK,
            "baseline_root": "baselines/external/csguoh_intlora",
            "train_script": "baselines/external/csguoh_intlora/train_dreambooth_quant.py",
            "output_root": "outputs/baselines/int_lora",
        },
        "python": {
            "executable": "python",
        },
        "model": {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "revision": None,
        },
        "dataset": {
            "instance_data_dir": "tests/fixtures/rdlora_pilot",
            "class_data_dir": None,
            "instance_prompt": "a photo of qwe subject",
            "class_prompt": None,
            "validation_prompt": "a qwe subject in studio light",
            "test_prompt": "a subject in studio light",
            "with_prior_preservation": False,
            "prior_loss_weight": 1.0,
            "num_class_images": 200,
        },
        "run": {
            "name": "sd15_appendix_smoke",
        },
        "training": {
            "rank": 4,
            "intlora": "MUL",
            "nbits": 8,
            "use_activation_quant": True,
            "act_nbits": 8,
            "resolution": 512,
            "center_crop": False,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "checkpointing_steps": 5000,
            "learning_rate": 6.0e-5,
            "report_to": "none",
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "max_train_steps": 2000,
            "num_validation_images": 5,
            "validation_epochs": 1,
            "seed": 0,
            "name": "appendix_subject",
            "gradient_checkpointing": False,
        },
    }


def _resolve_common_baseline_paths(
    repo_root: Path,
    raw_paths: Mapping[str, Any],
    *,
    default_baseline_root: str,
    default_train_script: str,
    default_output_root: str,
) -> dict[str, str]:
    resolved_repo_root = resolve_path(repo_root, str(raw_paths.get("repo_root", "."))).resolve()
    registry_lock = resolve_path(
        resolved_repo_root,
        str(raw_paths.get("registry_lock", DEFAULT_REGISTRY_LOCK)),
    ).resolve()
    baseline_root = resolve_path(
        resolved_repo_root,
        str(raw_paths.get("baseline_root", default_baseline_root)),
    ).resolve()
    train_script = resolve_path(
        resolved_repo_root,
        str(raw_paths.get("train_script", default_train_script)),
    ).resolve()
    output_root = resolve_path(
        resolved_repo_root,
        str(raw_paths.get("output_root", default_output_root)),
    ).resolve()

    if not registry_lock.is_file():
        raise FileNotFoundError(f"Registry lock not found: {registry_lock}")
    if not baseline_root.is_dir():
        raise FileNotFoundError(f"Baseline root not found: {baseline_root}")
    if not train_script.is_file():
        raise FileNotFoundError(f"Baseline train script not found: {train_script}")

    for name, path in {
        "paths.baseline_root": baseline_root,
        "paths.output_root": output_root,
    }.items():
        _validate_repo_local_path(resolved_repo_root, path, name=name)

    return {
        "repo_root": str(resolved_repo_root),
        "registry_lock": str(registry_lock),
        "baseline_root": str(baseline_root),
        "train_script": str(train_script),
        "output_root": str(output_root),
    }


def resolve_tlora_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_tlora_config(), raw_config)
    paths = _resolve_common_baseline_paths(
        repo_root,
        dict(config.get("paths", {})),
        default_baseline_root="baselines/external/controlgenai_t-lora",
        default_train_script="baselines/external/controlgenai_t-lora/train.py",
        default_output_root="outputs/baselines/t_lora",
    )
    accelerate = dict(config.get("accelerate", {}))
    model = dict(config.get("model", {}))
    dataset = dict(config.get("dataset", {}))
    run = dict(config.get("run", {}))
    training = dict(config.get("training", {}))

    resolved_repo_root = Path(paths["repo_root"])
    accelerate_config = resolve_path(
        resolved_repo_root,
        str(
            dict(config.get("paths", {})).get(
                "accelerate_config",
                "configs/accelerate/single_gpu_fp16.yaml",
            )
        ),
    ).resolve()
    if not accelerate_config.is_file():
        raise FileNotFoundError(f"T-LoRA accelerate config not found: {accelerate_config}")

    train_data_dir = resolve_path(
        resolved_repo_root,
        str(dataset.get("train_data_dir", "baselines/external/controlgenai_t-lora/dog_example")),
    ).resolve()
    class_data_dir = dataset.get("class_data_dir")
    resolved_class_data_dir = None
    if class_data_dir not in (None, ""):
        resolved_class_data_dir = resolve_path(resolved_repo_root, str(class_data_dir)).resolve()
    if not train_data_dir.is_dir():
        raise FileNotFoundError(f"dataset.train_data_dir does not exist: {train_data_dir}")

    output_root = Path(paths["output_root"])
    run_name = _validate_single_path_token(str(run.get("name", "dog_example")), name="run.name")
    pretrained_model_name_or_path = str(model.get("pretrained_model_name_or_path") or "").strip()
    if not pretrained_model_name_or_path:
        raise BaselineWrapperValidationError("model.pretrained_model_name_or_path is required")

    resolution = int(training.get("resolution", 1024))
    if resolution != 1024:
        raise BaselineWrapperValidationError("T-LoRA wrapper is locked to SDXL 1024 resolution")

    return {
        "baseline": {
            "id": TLORA_BASELINE_ID,
            "critical_path": True,
        },
        "paths": {
            **paths,
            "accelerate_config": str(accelerate_config),
            "train_data_dir": str(train_data_dir),
            "class_data_dir": str(resolved_class_data_dir) if resolved_class_data_dir else None,
        },
        "accelerate": {
            "executable": str(accelerate.get("executable", "accelerate")).strip() or "accelerate",
        },
        "model": {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "revision": None if model.get("revision") in (None, "") else str(model.get("revision")),
        },
        "dataset": {
            "class_name": str(dataset.get("class_name") or "").strip(),
            "placeholder_token": str(dataset.get("placeholder_token") or "").strip(),
            "validation_prompts": None
            if dataset.get("validation_prompts") in (None, "")
            else str(dataset.get("validation_prompts")),
            "num_val_imgs_per_prompt": int(dataset.get("num_val_imgs_per_prompt", 3)),
            "with_prior_preservation": bool(dataset.get("with_prior_preservation", False)),
            "prior_loss_weight": float(dataset.get("prior_loss_weight", 1.0)),
            "one_image": None if dataset.get("one_image") in (None, "") else str(dataset.get("one_image")),
        },
        "run": {
            "name": run_name,
            "wandb_api_key_env": str(run.get("wandb_api_key_env", "WANDB_API_KEY")).strip(),
            "pass_wandb_api_key": bool(run.get("pass_wandb_api_key", False)),
        },
        "training": {
            "trainer_type": str(training.get("trainer_type", "ortho_lora")).strip(),
            "trainer_class": str(training.get("trainer_class", "sdxl_tlora")).strip(),
            "mixed_precision": str(training.get("mixed_precision", "fp16")).strip(),
            "num_train_epochs": int(training.get("num_train_epochs", 800)),
            "checkpointing_steps": int(training.get("checkpointing_steps", 100)),
            "resolution": resolution,
            "train_batch_size": int(training.get("train_batch_size", 1)),
            "dataloader_num_workers": int(training.get("dataloader_num_workers", 1)),
            "seed": int(training.get("seed", 0)),
            "lora_rank": int(training.get("lora_rank", 64)),
            "min_rank": int(training.get("min_rank", 32)),
            "sig_type": str(training.get("sig_type", "last")).strip(),
            "alpha_rank_scale": float(training.get("alpha_rank_scale", 1.0)),
            "learning_rate": float(training.get("learning_rate", 1.0e-4)),
            "adam_beta1": float(training.get("adam_beta1", 0.9)),
            "adam_beta2": float(training.get("adam_beta2", 0.999)),
            "adam_weight_decay": float(training.get("adam_weight_decay", 1.0e-4)),
            "adam_epsilon": float(training.get("adam_epsilon", 1.0e-8)),
        },
    }


def resolve_intlora_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_intlora_config(), raw_config)
    paths = _resolve_common_baseline_paths(
        repo_root,
        dict(config.get("paths", {})),
        default_baseline_root="baselines/external/csguoh_intlora",
        default_train_script="baselines/external/csguoh_intlora/train_dreambooth_quant.py",
        default_output_root="outputs/baselines/int_lora",
    )
    python_config = dict(config.get("python", {}))
    model = dict(config.get("model", {}))
    dataset = dict(config.get("dataset", {}))
    run = dict(config.get("run", {}))
    training = dict(config.get("training", {}))

    resolved_repo_root = Path(paths["repo_root"])
    instance_data_dir = resolve_path(
        resolved_repo_root,
        str(dataset.get("instance_data_dir", "tests/fixtures/rdlora_pilot")),
    ).resolve()
    class_data_dir = dataset.get("class_data_dir")
    resolved_class_data_dir = None
    if class_data_dir not in (None, ""):
        resolved_class_data_dir = resolve_path(resolved_repo_root, str(class_data_dir)).resolve()
    if not instance_data_dir.is_dir():
        raise FileNotFoundError(f"dataset.instance_data_dir does not exist: {instance_data_dir}")

    output_root = Path(paths["output_root"])
    run_name = _validate_single_path_token(
        str(run.get("name", "sd15_appendix_smoke")),
        name="run.name",
    )
    pretrained_model_name_or_path = str(model.get("pretrained_model_name_or_path") or "").strip()
    if not pretrained_model_name_or_path:
        raise BaselineWrapperValidationError("model.pretrained_model_name_or_path is required")

    resolution = int(training.get("resolution", 512))
    if resolution != 512:
        raise BaselineWrapperValidationError("IntLoRA appendix wrapper is locked to the official 512 SD1.5 path")

    return {
        "baseline": {
            "id": INTLORA_BASELINE_ID,
            "critical_path": False,
        },
        "paths": {
            **paths,
            "instance_data_dir": str(instance_data_dir),
            "class_data_dir": str(resolved_class_data_dir) if resolved_class_data_dir else None,
        },
        "python": {
            "executable": str(python_config.get("executable", "python")).strip() or "python",
        },
        "model": {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "revision": None if model.get("revision") in (None, "") else str(model.get("revision")),
        },
        "dataset": {
            "instance_prompt": str(dataset.get("instance_prompt") or "").strip(),
            "class_prompt": None if dataset.get("class_prompt") in (None, "") else str(dataset.get("class_prompt")),
            "validation_prompt": None
            if dataset.get("validation_prompt") in (None, "")
            else str(dataset.get("validation_prompt")),
            "test_prompt": None if dataset.get("test_prompt") in (None, "") else str(dataset.get("test_prompt")),
            "with_prior_preservation": bool(dataset.get("with_prior_preservation", False)),
            "prior_loss_weight": float(dataset.get("prior_loss_weight", 1.0)),
            "num_class_images": int(dataset.get("num_class_images", 200)),
        },
        "run": {
            "name": run_name,
        },
        "training": {
            "rank": int(training.get("rank", 4)),
            "intlora": str(training.get("intlora", "MUL")).strip(),
            "nbits": int(training.get("nbits", 8)),
            "use_activation_quant": bool(training.get("use_activation_quant", True)),
            "act_nbits": int(training.get("act_nbits", 8)),
            "resolution": resolution,
            "center_crop": bool(training.get("center_crop", False)),
            "train_batch_size": int(training.get("train_batch_size", 1)),
            "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
            "checkpointing_steps": int(training.get("checkpointing_steps", 5000)),
            "learning_rate": float(training.get("learning_rate", 6.0e-5)),
            "report_to": str(training.get("report_to", "none")).strip(),
            "lr_scheduler": str(training.get("lr_scheduler", "constant")).strip(),
            "lr_warmup_steps": int(training.get("lr_warmup_steps", 0)),
            "max_train_steps": int(training.get("max_train_steps", 2000)),
            "num_validation_images": int(training.get("num_validation_images", 5)),
            "validation_epochs": int(training.get("validation_epochs", 1)),
            "seed": int(training.get("seed", 0)),
            "name": str(training.get("name", "appendix_subject")).strip(),
            "gradient_checkpointing": bool(training.get("gradient_checkpointing", False)),
        },
    }


def verify_registered_baseline(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    registry_path = Path(config["paths"]["registry_lock"])
    baseline_root = Path(config["paths"]["baseline_root"])
    registry_payload = load_registry_lock(registry_path)
    record = get_registry_record(registry_payload, str(config["baseline"]["id"]))
    checkout = inspect_registered_checkout(baseline_root)

    expected_path = str(record.get("local_checkout_path"))
    actual_relative_path = display_path(baseline_root, repo_root)
    issues: list[str] = []

    if actual_relative_path != expected_path:
        issues.append(
            f"local checkout mismatch: registry={expected_path!r} actual={actual_relative_path!r}"
        )
    if checkout["origin_url"] != str(record.get("upstream_url")):
        issues.append(
            "origin remote mismatch: "
            f"registry={record.get('upstream_url')!r} actual={checkout['origin_url']!r}"
        )
    if checkout["head_commit"] != str(record.get("pinned_commit")):
        issues.append(
            "pinned commit mismatch: "
            f"registry={record.get('pinned_commit')!r} actual={checkout['head_commit']!r}"
        )

    expected_critical_path = bool(config["baseline"]["critical_path"])
    if bool(record.get("critical_path")) != expected_critical_path:
        issues.append(
            "critical_path mismatch: "
            f"registry={bool(record.get('critical_path'))} expected={expected_critical_path}"
        )

    if expected_critical_path:
        if str(record.get("status")) != "available":
            issues.append("critical-path baseline must have status='available'")
    else:
        if str(record.get("status")) != "appendix-only":
            issues.append("appendix baseline must have status='appendix-only'")
        if not str(record.get("reason") or "").strip():
            issues.append("appendix baseline must record a non-critical-path reason")

    return {
        "ok": not issues,
        "issues": issues,
        "registry_path": display_path(registry_path, repo_root),
        "baseline_id": str(record.get("id")),
        "baseline_name": str(record.get("name")),
        "status": str(record.get("status")),
        "critical_path": bool(record.get("critical_path")),
        "official_repo": str(record.get("upstream_url")),
        "local_checkout_path": expected_path,
        "pinned_commit": str(record.get("pinned_commit")),
        "actual_checkout": checkout,
        "record": record,
    }


def build_tlora_command(config: Mapping[str, Any], *, run_name: str | None = None) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    output_root = Path(config["paths"]["output_root"])
    resolved_run_name = _validate_single_path_token(
        run_name or str(config["run"]["name"]),
        name="run_name",
    )
    output_dir = (output_root / resolved_run_name).resolve()
    logging_dir = (output_dir / "logs").resolve()
    _validate_repo_local_path(repo_root, output_dir, name="output_dir")

    command: list[str] = [str(config["accelerate"]["executable"]), "launch"]
    _append_optional_flag(command, "--config_file", config["paths"]["accelerate_config"])
    command.append(str(config["paths"]["train_script"]))
    command.extend(
        [
            "--trainer_type",
            str(config["training"]["trainer_type"]),
            "--trainer_class",
            str(config["training"]["trainer_class"]),
            "--pretrained_model_name_or_path",
            str(config["model"]["pretrained_model_name_or_path"]),
            "--mixed_precision",
            str(config["training"]["mixed_precision"]),
            "--num_train_epochs",
            str(config["training"]["num_train_epochs"]),
            "--checkpointing_steps",
            str(config["training"]["checkpointing_steps"]),
            "--train_data_dir",
            str(config["paths"]["train_data_dir"]),
            "--train_batch_size",
            str(config["training"]["train_batch_size"]),
            "--dataloader_num_workers",
            str(config["training"]["dataloader_num_workers"]),
            "--resolution",
            str(config["training"]["resolution"]),
            "--output_dir",
            str(output_dir),
            "--class_name",
            str(config["dataset"]["class_name"]),
            "--placeholder_token",
            str(config["dataset"]["placeholder_token"]),
            "--num_val_imgs_per_prompt",
            str(config["dataset"]["num_val_imgs_per_prompt"]),
            "--lora_rank",
            str(config["training"]["lora_rank"]),
            "--min_rank",
            str(config["training"]["min_rank"]),
            "--sig_type",
            str(config["training"]["sig_type"]),
            "--alpha_rank_scale",
            str(config["training"]["alpha_rank_scale"]),
            "--learning_rate",
            str(config["training"]["learning_rate"]),
            "--adam_beta1",
            str(config["training"]["adam_beta1"]),
            "--adam_beta2",
            str(config["training"]["adam_beta2"]),
            "--adam_weight_decay",
            str(config["training"]["adam_weight_decay"]),
            "--adam_epsilon",
            str(config["training"]["adam_epsilon"]),
            "--seed",
            str(config["training"]["seed"]),
        ]
    )
    _append_optional_flag(command, "--revision", config["model"]["revision"])
    _append_optional_flag(command, "--class_data_dir", config["paths"]["class_data_dir"])
    _append_optional_flag(command, "--validation_prompts", config["dataset"]["validation_prompts"])
    _append_optional_flag(command, "--one_image", config["dataset"]["one_image"])

    if config["dataset"]["with_prior_preservation"]:
        command.append("--with_prior_preservation")
        command.extend(["--prior_loss_weight", str(config["dataset"]["prior_loss_weight"])])

    return {
        "baseline_id": TLORA_BASELINE_ID,
        "run_name": resolved_run_name,
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "train_script": str(config["paths"]["train_script"]),
        "command": command,
        "manual_command": render_shell_command(command),
    }


def build_intlora_command(config: Mapping[str, Any], *, run_name: str | None = None) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    output_root = Path(config["paths"]["output_root"])
    resolved_run_name = _validate_single_path_token(
        run_name or str(config["run"]["name"]),
        name="run_name",
    )
    output_dir = (output_root / resolved_run_name).resolve()
    _validate_repo_local_path(repo_root, output_dir, name="output_dir")

    command: list[str] = [
        str(config["python"]["executable"]),
        str(config["paths"]["train_script"]),
        "--pretrained_model_name_or_path",
        str(config["model"]["pretrained_model_name_or_path"]),
        "--instance_data_dir",
        str(config["paths"]["instance_data_dir"]),
        "--rank",
        str(config["training"]["rank"]),
        "--intlora",
        str(config["training"]["intlora"]),
        "--nbits",
        str(config["training"]["nbits"]),
        "--act_nbits",
        str(config["training"]["act_nbits"]),
        "--output_dir",
        str(output_dir),
        "--instance_prompt",
        str(config["dataset"]["instance_prompt"]),
        "--resolution",
        str(config["training"]["resolution"]),
        "--train_batch_size",
        str(config["training"]["train_batch_size"]),
        "--gradient_accumulation_steps",
        str(config["training"]["gradient_accumulation_steps"]),
        "--checkpointing_steps",
        str(config["training"]["checkpointing_steps"]),
        "--learning_rate",
        str(config["training"]["learning_rate"]),
        "--report_to",
        str(config["training"]["report_to"]),
        "--lr_scheduler",
        str(config["training"]["lr_scheduler"]),
        "--lr_warmup_steps",
        str(config["training"]["lr_warmup_steps"]),
        "--max_train_steps",
        str(config["training"]["max_train_steps"]),
        "--num_validation_images",
        str(config["training"]["num_validation_images"]),
        "--validation_epochs",
        str(config["training"]["validation_epochs"]),
        "--seed",
        str(config["training"]["seed"]),
        "--name",
        str(config["training"]["name"]),
    ]
    _append_optional_flag(command, "--revision", config["model"]["revision"])
    _append_optional_flag(command, "--class_data_dir", config["paths"]["class_data_dir"])
    _append_optional_flag(command, "--class_prompt", config["dataset"]["class_prompt"])
    _append_optional_flag(command, "--validation_prompt", config["dataset"]["validation_prompt"])
    _append_optional_flag(command, "--test_prompt", config["dataset"]["test_prompt"])
    if config["dataset"]["with_prior_preservation"]:
        command.append("--with_prior_preservation")
        command.extend(["--prior_loss_weight", str(config["dataset"]["prior_loss_weight"])])
        command.extend(["--num_class_images", str(config["dataset"]["num_class_images"])])
    if config["training"]["use_activation_quant"]:
        command.append("--use_activation_quant")
    if config["training"]["center_crop"]:
        command.append("--center_crop")
    if config["training"]["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")

    return {
        "baseline_id": INTLORA_BASELINE_ID,
        "run_name": resolved_run_name,
        "output_dir": str(output_dir),
        "train_script": str(config["paths"]["train_script"]),
        "command": command,
        "manual_command": render_shell_command(command),
    }


def validate_tlora_launch_plan(plan: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    command = [str(item) for item in plan["command"]]
    if len(command) < 4:
        raise BaselineWrapperValidationError("generated T-LoRA command is incomplete")
    if command[1] != "launch":
        raise BaselineWrapperValidationError("T-LoRA command must use 'accelerate launch'")

    train_script = Path(plan["train_script"]).resolve()
    if not train_script.is_file():
        raise BaselineWrapperValidationError(f"T-LoRA train.py not found: {train_script}")
    if str(train_script) not in command:
        raise BaselineWrapperValidationError("T-LoRA command must target the official train.py path")

    output_dir = Path(plan["output_dir"]).resolve()
    _validate_repo_local_path(repo_root, output_dir, name="launch output_dir")

    accelerate_executable = command[0]
    accelerate_found = (
        shutil.which(accelerate_executable) is not None
        if Path(accelerate_executable).name == accelerate_executable
        else Path(accelerate_executable).exists()
    )
    accelerate_config_path = None
    if "--config_file" in command:
        accelerate_config_path = Path(command[command.index("--config_file") + 1]).resolve()
        if not accelerate_config_path.is_file():
            raise BaselineWrapperValidationError(
                f"T-LoRA accelerate config does not exist: {accelerate_config_path}"
            )
    accelerate_config = _load_accelerate_config(accelerate_config_path)

    required_flags = (
        "--trainer_type",
        "--trainer_class",
        "--pretrained_model_name_or_path",
        "--train_data_dir",
        "--output_dir",
        "--resolution",
        "--class_name",
        "--placeholder_token",
        "--lora_rank",
        "--min_rank",
    )
    missing_flags = [flag for flag in required_flags if flag not in command]
    if missing_flags:
        raise BaselineWrapperValidationError(
            f"T-LoRA command is missing required flags: {', '.join(missing_flags)}"
        )

    return {
        "accelerate_executable": accelerate_executable,
        "accelerate_found": bool(accelerate_found),
        "accelerate_config": (
            display_path(accelerate_config_path, repo_root) if accelerate_config_path else None
        ),
        "accelerate_uses_cpu": bool(accelerate_config.get("use_cpu", False))
        if accelerate_config
        else None,
        "train_script": display_path(train_script, repo_root),
        "output_dir": display_path(output_dir, repo_root),
        "manual_command": str(plan["manual_command"]),
    }


def validate_intlora_launch_plan(plan: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    command = [str(item) for item in plan["command"]]
    if len(command) < 3:
        raise BaselineWrapperValidationError("generated IntLoRA command is incomplete")

    train_script = Path(plan["train_script"]).resolve()
    if not train_script.is_file():
        raise BaselineWrapperValidationError(f"IntLoRA train_dreambooth_quant.py not found: {train_script}")
    if str(train_script) not in command:
        raise BaselineWrapperValidationError(
            "IntLoRA command must target the official train_dreambooth_quant.py path"
        )

    output_dir = Path(plan["output_dir"]).resolve()
    _validate_repo_local_path(repo_root, output_dir, name="launch output_dir")

    python_executable = command[0]
    python_found = (
        shutil.which(python_executable) is not None
        if Path(python_executable).name == python_executable
        else Path(python_executable).exists()
    )

    required_flags = (
        "--pretrained_model_name_or_path",
        "--instance_data_dir",
        "--rank",
        "--intlora",
        "--nbits",
        "--act_nbits",
        "--output_dir",
        "--instance_prompt",
        "--resolution",
        "--max_train_steps",
    )
    missing_flags = [flag for flag in required_flags if flag not in command]
    if missing_flags:
        raise BaselineWrapperValidationError(
            f"IntLoRA command is missing required flags: {', '.join(missing_flags)}"
        )

    return {
        "python_executable": python_executable,
        "python_found": bool(python_found),
        "train_script": display_path(train_script, repo_root),
        "output_dir": display_path(output_dir, repo_root),
        "manual_command": str(plan["manual_command"]),
    }


def write_wrapper_artifacts(
    output_dir: Path,
    *,
    resolved_config: Mapping[str, Any],
    resolved_config_path: Path,
    launch_script_name: str,
    command: Sequence[str],
    summary_path: Path,
    summary_payload: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary_path, summary_payload)
    write_shell_command(output_dir / launch_script_name, command)
    save_json(
        output_dir / "resolved_config_snapshot.json",
        {
            "resolved_config_path": str(resolved_config_path),
            "baseline_id": resolved_config["baseline"]["id"],
            "run_name": resolved_config["run"]["name"],
        },
    )


__all__ = [
    "BaselineWrapperValidationError",
    "INTLORA_BASELINE_ID",
    "TLORA_BASELINE_ID",
    "build_cli_overrides",
    "build_intlora_command",
    "build_tlora_command",
    "default_intlora_config",
    "default_tlora_config",
    "deep_update",
    "display_path",
    "get_registry_record",
    "inspect_registered_checkout",
    "load_registry_lock",
    "load_yaml_mapping",
    "resolve_intlora_config",
    "resolve_tlora_config",
    "save_json",
    "validate_intlora_launch_plan",
    "validate_tlora_launch_plan",
    "verify_registered_baseline",
    "write_shell_command",
    "write_wrapper_artifacts",
]
