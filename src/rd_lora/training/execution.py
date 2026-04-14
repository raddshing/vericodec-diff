from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from rd_lora.runtime.allocation_manifest import load_allocation_manifest
from rd_lora.runtime.provenance import (
    assert_real_gpu_training_provenance,
    collect_runtime_metadata,
)
from rd_lora.substrate.diffusers_sdxl import (
    describe_loaded_sdxl_components,
    load_official_sdxl_components,
    resolve_path,
    save_json,
    validate_official_sdxl_dreambooth_script,
)
from rd_lora.training.adapter_factory import (
    build_backend_plan,
    build_concrete_adapter_specs,
)
from rd_lora.training.timestep_routing import (
    TimestepRoutingError,
    build_timestep_band_routes,
    resolve_deterministic_timestep_for_band,
    resolve_timestep_band_name_for_training_timesteps,
)


TRAINING_SUCCESS_STATUS = "completed"
REQUIRED_SUCCESS_FILES = (
    "run_provenance.json",
    "train_summary.json",
    "metrics.json",
    "checkpoint_info.json",
)
REQUIRED_SUMMARY_KEYS = (
    "status",
    "backend",
    "task",
    "allocation_manifest",
    "train_steps",
    "checkpoint_path",
)
REQUIRED_CHECKPOINT_INFO_KEYS = (
    "checkpoint_path",
    "checkpoint_exists",
)
REQUIRED_METRIC_KEYS = (
    "global_step",
    "train_loss_last",
    "train_loss_mean",
    "skipped_noop_band_batches",
    "skipped_noop_band_fraction",
    "active_band_optimizer_steps",
)


class TrainingExecutionError(RuntimeError):
    """Raised when the real RD-LoRA training path cannot complete successfully."""


def load_torch() -> Any:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise TrainingExecutionError(f"Unable to import torch: {type(exc).__name__}: {exc}") from exc
    return torch


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingExecutionError(f"{name} must be a mapping")
    return dict(value)


def _require_real_gpu(torch_module: Any) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        raise TrainingExecutionError("torch.cuda is unavailable for a real_gpu run")
    if not bool(cuda.is_available()):
        raise TrainingExecutionError("CUDA must be available for a real_gpu run")
    if int(cuda.device_count()) < 1:
        raise TrainingExecutionError("torch.cuda.device_count() must be at least 1 for a real_gpu run")


def _resolve_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model = _require_mapping(config.get("model", {}), name="config.model")
    pretrained_model_name_or_path = str(model.get("pretrained_model_name_or_path") or "").strip()
    if not pretrained_model_name_or_path:
        raise TrainingExecutionError("model.pretrained_model_name_or_path is required")
    return {
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "pretrained_vae_model_name_or_path": (
            None
            if model.get("pretrained_vae_model_name_or_path") in (None, "")
            else str(model.get("pretrained_vae_model_name_or_path"))
        ),
        "revision": None if model.get("revision") in (None, "") else str(model.get("revision")),
        "variant": None if model.get("variant") in (None, "") else str(model.get("variant")),
    }


def _resolve_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    training = _require_mapping(config.get("training", {}), name="config.training")
    return {
        "resolution": int(training.get("resolution", 1024)),
        "train_batch_size": int(training.get("train_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "max_train_steps": int(training.get("max_train_steps", 1)),
        "checkpointing_steps": int(training.get("checkpointing_steps", 1)),
        "checkpoints_total_limit": (
            None
            if training.get("checkpoints_total_limit") in (None, "")
            else int(training.get("checkpoints_total_limit"))
        ),
        "learning_rate": float(training.get("learning_rate", 1.0e-4)),
        "lr_scheduler": str(training.get("lr_scheduler", "constant")),
        "lr_warmup_steps": int(training.get("lr_warmup_steps", 0)),
        "lr_num_cycles": int(training.get("lr_num_cycles", 1)),
        "lr_power": float(training.get("lr_power", 1.0)),
        "report_to": str(training.get("report_to", "tensorboard")),
        "mixed_precision": str(training.get("mixed_precision", "fp16")),
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 0)),
        "num_validation_images": int(training.get("num_validation_images", 1)),
        "validation_epochs": int(training.get("validation_epochs", 1)),
        "center_crop": bool(training.get("center_crop", True)),
        "random_flip": bool(training.get("random_flip", False)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", False)),
        "scale_lr": bool(training.get("scale_lr", False)),
        "allow_tf32": bool(training.get("allow_tf32", False)),
        "use_8bit_adam": bool(training.get("use_8bit_adam", False)),
        "enable_xformers_memory_efficient_attention": bool(
            training.get("enable_xformers_memory_efficient_attention", False)
        ),
        "optimizer": str(training.get("optimizer", "AdamW")),
        "adam_beta1": float(training.get("adam_beta1", 0.9)),
        "adam_beta2": float(training.get("adam_beta2", 0.999)),
        "adam_weight_decay": float(training.get("adam_weight_decay", 1.0e-4)),
        "adam_weight_decay_text_encoder": float(training.get("adam_weight_decay_text_encoder", 1.0e-3)),
        "adam_epsilon": float(training.get("adam_epsilon", 1.0e-8)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "lora_dropout": float(training.get("lora_dropout", 0.0)),
        "repeats": int(training.get("repeats", 1)),
        "seed": None if training.get("seed") in (None, "") else int(training.get("seed")),
        "use_rslora": bool(training.get("use_rslora", True)),
        "target_modules": list(training.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"])),
    }


def _resolve_task_config(
    repo_root: Path,
    config: Mapping[str, Any],
    task: str,
) -> dict[str, Any]:
    tasks = _require_mapping(config.get("tasks", {}), name="config.tasks")
    if task not in tasks:
        raise TrainingExecutionError(f"config.tasks is missing {task!r}")
    payload = _require_mapping(tasks[task], name=f"config.tasks.{task}")
    train_data_dir = resolve_path(repo_root, str(payload.get("train_data_dir", ""))).resolve()
    if not train_data_dir.is_dir():
        raise TrainingExecutionError(f"Task image directory not found: {train_data_dir}")
    validation_prompt = str(payload.get("validation_prompt") or "").strip()
    if not validation_prompt:
        raise TrainingExecutionError(f"config.tasks.{task}.validation_prompt must be non-empty")
    instance_prompt = str(payload.get("instance_prompt") or "").strip()
    if not instance_prompt:
        raise TrainingExecutionError(f"config.tasks.{task}.instance_prompt must be non-empty")
    return {
        "train_data_dir": str(train_data_dir),
        "validation_prompt": validation_prompt,
        "instance_prompt": instance_prompt,
    }


def build_training_args(
    *,
    repo_root: Path,
    config: Mapping[str, Any],
    task: str,
    output_dir: Path,
) -> SimpleNamespace:
    paths = _require_mapping(config.get("paths", {}), name="config.paths")
    model = _resolve_model_config(config)
    training = _resolve_training_config(config)
    task_config = _resolve_task_config(repo_root, config, task)
    accelerate_config = resolve_path(repo_root, str(paths.get("accelerate_config", ""))).resolve()
    if not accelerate_config.is_file():
        raise TrainingExecutionError(f"Accelerate config not found: {accelerate_config}")

    return SimpleNamespace(
        pretrained_model_name_or_path=model["pretrained_model_name_or_path"],
        pretrained_vae_model_name_or_path=model["pretrained_vae_model_name_or_path"],
        revision=model["revision"],
        variant=model["variant"],
        dataset_name=None,
        dataset_config_name=None,
        instance_data_dir=task_config["train_data_dir"],
        cache_dir=None,
        image_column="image",
        caption_column=None,
        repeats=training["repeats"],
        class_data_dir=None,
        instance_prompt=task_config["instance_prompt"],
        class_prompt=None,
        validation_prompt=task_config["validation_prompt"],
        num_validation_images=training["num_validation_images"],
        validation_epochs=training["validation_epochs"],
        do_edm_style_training=False,
        with_prior_preservation=False,
        prior_loss_weight=1.0,
        num_class_images=0,
        output_dir=str(output_dir),
        output_kohya_format=False,
        seed=training["seed"],
        resolution=training["resolution"],
        center_crop=training["center_crop"],
        random_flip=training["random_flip"],
        train_text_encoder=False,
        train_batch_size=training["train_batch_size"],
        sample_batch_size=training["train_batch_size"],
        num_train_epochs=1,
        max_train_steps=training["max_train_steps"],
        checkpointing_steps=training["checkpointing_steps"],
        checkpoints_total_limit=training["checkpoints_total_limit"],
        resume_from_checkpoint=None,
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        gradient_checkpointing=training["gradient_checkpointing"],
        learning_rate=training["learning_rate"],
        text_encoder_lr=5.0e-6,
        scale_lr=training["scale_lr"],
        lr_scheduler=training["lr_scheduler"],
        snr_gamma=None,
        lr_warmup_steps=training["lr_warmup_steps"],
        lr_num_cycles=training["lr_num_cycles"],
        lr_power=training["lr_power"],
        dataloader_num_workers=training["dataloader_num_workers"],
        optimizer=training["optimizer"],
        use_8bit_adam=training["use_8bit_adam"],
        adam_beta1=training["adam_beta1"],
        adam_beta2=training["adam_beta2"],
        prodigy_beta3=None,
        prodigy_decouple=True,
        adam_weight_decay=training["adam_weight_decay"],
        adam_weight_decay_text_encoder=training["adam_weight_decay_text_encoder"],
        adam_epsilon=training["adam_epsilon"],
        prodigy_use_bias_correction=True,
        prodigy_safeguard_warmup=True,
        max_grad_norm=training["max_grad_norm"],
        push_to_hub=False,
        hub_token=None,
        hub_model_id=None,
        logging_dir=str((output_dir / "logs").resolve()),
        allow_tf32=training["allow_tf32"],
        report_to=training["report_to"],
        mixed_precision=training["mixed_precision"],
        prior_generation_precision=None,
        local_rank=-1,
        enable_xformers_memory_efficient_attention=training["enable_xformers_memory_efficient_attention"],
        rank=1,
        lora_dropout=training["lora_dropout"],
        use_dora=False,
        image_interpolation_mode="lanczos",
        accelerate_config_file=str(accelerate_config),
        expected_diffusers_substring=str(paths.get("expected_diffusers_substring", "huggingface_diffusers")),
    )


def create_accelerator(training_args: SimpleNamespace) -> Any:
    try:
        from accelerate import Accelerator  # type: ignore
        from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration  # type: ignore
    except Exception as exc:
        raise TrainingExecutionError(f"Unable to import accelerate: {type(exc).__name__}: {exc}") from exc

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_config = ProjectConfiguration(
        project_dir=training_args.output_dir,
        logging_dir=training_args.logging_dir,
    )
    return Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=training_args.mixed_precision,
        log_with=training_args.report_to,
        project_config=project_config,
        kwargs_handlers=[kwargs],
    )


def create_train_dataloader(
    *,
    torch_module: Any,
    official_module: Any,
    training_args: SimpleNamespace,
) -> Any:
    official_module.args = training_args
    train_dataset = official_module.DreamBoothDataset(
        instance_data_root=training_args.instance_data_dir,
        instance_prompt=training_args.instance_prompt,
        class_prompt=training_args.class_prompt,
        class_data_root=None,
        class_num=None,
        size=training_args.resolution,
        repeats=training_args.repeats,
        center_crop=training_args.center_crop,
    )
    return torch_module.utils.data.DataLoader(
        train_dataset,
        batch_size=training_args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: official_module.collate_fn(examples, training_args.with_prior_preservation),
        num_workers=training_args.dataloader_num_workers,
    )


def _weight_dtype(torch_module: Any, mixed_precision: str) -> Any:
    if mixed_precision == "fp16":
        return torch_module.float16
    if mixed_precision == "bf16":
        return torch_module.bfloat16
    return torch_module.float32


def _collect_latent_stats(torch_module: Any, vae: Any) -> tuple[Any | None, Any | None]:
    latents_mean = None
    latents_std = None
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        latents_mean = torch_module.tensor(vae.config.latents_mean).view(1, 4, 1, 1)
    if hasattr(vae.config, "latents_std") and vae.config.latents_std is not None:
        latents_std = torch_module.tensor(vae.config.latents_std).view(1, 4, 1, 1)
    return latents_mean, latents_std


def _configure_components_for_training(
    *,
    torch_module: Any,
    accelerator: Any,
    official_module: Any,
    training_args: SimpleNamespace,
    components: Mapping[str, Any],
) -> dict[str, Any]:
    weight_dtype = _weight_dtype(torch_module, str(getattr(accelerator, "mixed_precision", training_args.mixed_precision)))
    text_encoder = components["text_encoder"]
    text_encoder_2 = components["text_encoder_2"]
    vae = components["vae"]
    unet = components["unet"]

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch_module.float32)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    if training_args.gradient_checkpointing and hasattr(unet, "enable_gradient_checkpointing"):
        unet.enable_gradient_checkpointing()

    latents_mean, latents_std = _collect_latent_stats(torch_module, vae)
    return {
        "weight_dtype": weight_dtype,
        "latents_mean": latents_mean,
        "latents_std": latents_std,
    }


def apply_adapter_specs(
    *,
    official_module: Any,
    unet: Any,
    adapter_specs: Sequence[Mapping[str, Any]],
    use_rslora: bool,
    lora_dropout: float,
) -> list[str]:
    created_any = False
    created_adapter_names: list[str] = []
    for spec in adapter_specs:
        if bool(spec["noop"]):
            continue
        adapter_config = official_module.LoraConfig(
            r=int(spec["rank"]),
            lora_alpha=int(spec["alpha"]),
            target_modules=list(spec["target_modules"]),
            rank_pattern=dict(spec["rank_pattern"]),
            alpha_pattern=dict(spec["alpha_pattern"]),
            lora_dropout=float(lora_dropout),
            init_lora_weights="gaussian",
            use_rslora=bool(use_rslora),
        )
        unet.add_adapter(adapter_config, adapter_name=str(spec["adapter_name"]))
        created_adapter_names.append(str(spec["adapter_name"]))
        created_any = True

    if not created_any:
        raise TrainingExecutionError("Adapter construction produced no trainable LoRA banks")
    return created_adapter_names


def assert_adapter_param_counts(unet: Any, adapter_names: list[str] | None) -> None:
    if not adapter_names:
        return
    for adapter_name in adapter_names:
        count = sum(p.numel() for n, p in unet.named_parameters() if adapter_name in n)
        if count == 0:
            raise TrainingExecutionError(
                f"Adapter {adapter_name!r} has zero parameters; bad rank_pattern/target mapping"
            )


def upcast_trainable_params_to_fp32(torch_module: Any, trainable_parameters: Sequence[Any]) -> None:
    seen_parameter_ids: set[int] = set()
    for parameter in trainable_parameters:
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        is_parameter_floating_point = getattr(parameter, "is_floating_point", None)
        if callable(is_parameter_floating_point) and is_parameter_floating_point() and parameter.dtype != torch_module.float32:
            parameter.data = parameter.data.to(dtype=torch_module.float32)
        grad = getattr(parameter, "grad", None)
        is_grad_floating_point = getattr(grad, "is_floating_point", None)
        if grad is not None and callable(is_grad_floating_point) and is_grad_floating_point():
            parameter.grad = parameter.grad.to(dtype=torch_module.float32)


def assert_all_params_fp32(torch_module: Any, trainable_parameters: Sequence[Any]) -> None:
    seen_parameter_ids: set[int] = set()
    for parameter in trainable_parameters:
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        is_parameter_floating_point = getattr(parameter, "is_floating_point", None)
        if callable(is_parameter_floating_point) and is_parameter_floating_point() and parameter.dtype != torch_module.float32:
            raise TrainingExecutionError(
                "All floating-point trainable parameters must be float32 before optimizer creation"
            )


def collect_trainable_parameters(unet: Any, adapter_names: Sequence[str] | None = None) -> list[Any]:
    if adapter_names is not None and len(adapter_names) > 0:
        trainable = [
            param
            for name, param in unet.named_parameters()
            if any(adapter_name in name for adapter_name in adapter_names)
        ]
    else:
        trainable = [parameter for parameter in unet.parameters() if bool(getattr(parameter, "requires_grad", False))]
    if not trainable:
        raise TrainingExecutionError("No trainable LoRA parameters were found after adapter construction")
    return trainable


def create_optimizer(
    *,
    torch_module: Any,
    trainable_parameters: Sequence[Any],
    training_args: SimpleNamespace,
) -> Any:
    if str(training_args.optimizer).lower() != "adamw":
        raise TrainingExecutionError(f"Unsupported optimizer {training_args.optimizer!r}; expected 'AdamW'")
    optimizer_class = torch_module.optim.AdamW
    return optimizer_class(
        [{"params": list(trainable_parameters), "lr": float(training_args.learning_rate)}],
        betas=(float(training_args.adam_beta1), float(training_args.adam_beta2)),
        weight_decay=float(training_args.adam_weight_decay),
        eps=float(training_args.adam_epsilon),
    )


def create_lr_scheduler(
    *,
    official_module: Any,
    optimizer: Any,
    accelerator: Any,
    train_dataloader: Any,
    training_args: SimpleNamespace,
) -> Any:
    num_update_steps_per_epoch = max(1, math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps))
    if training_args.max_train_steps is None:
        num_training_steps = int(training_args.num_train_epochs) * num_update_steps_per_epoch
    else:
        num_training_steps = int(training_args.max_train_steps) * int(getattr(accelerator, "num_processes", 1))
    return official_module.get_scheduler(
        training_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(training_args.lr_warmup_steps) * int(getattr(accelerator, "num_processes", 1)),
        num_training_steps=num_training_steps,
        num_cycles=int(training_args.lr_num_cycles),
        power=float(training_args.lr_power),
    )


def _plan_timestep_band_routes(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    routes = plan.get("timestep_band_routes")
    if isinstance(routes, Mapping) and routes:
        return {
            str(timestep_band): dict(route)
            for timestep_band, route in routes.items()
        }

    routing = _require_mapping(plan.get("routing", {}), name="plan.routing")
    if str(routing.get("mode")) == "timestep_band":
        table = routing.get("table")
        if not isinstance(table, Mapping) or not table:
            raise TrainingExecutionError("plan.routing.table must be a non-empty mapping")
        return {
            str(timestep_band): dict(route)
            for timestep_band, route in table.items()
        }

    timestep_bands = plan.get("timestep_bands")
    if not isinstance(timestep_bands, Sequence) or isinstance(timestep_bands, (str, bytes)):
        raise TrainingExecutionError("plan.timestep_bands must be a sequence")
    try:
        return build_timestep_band_routes([str(timestep_band) for timestep_band in timestep_bands])
    except TimestepRoutingError as exc:
        raise TrainingExecutionError(f"Unable to synthesize timestep band routes: {exc}") from exc


def _synthesize_timestep_band_metadata(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    timestep_bands = plan.get("timestep_bands")
    if not isinstance(timestep_bands, Sequence) or isinstance(timestep_bands, (str, bytes)):
        raise TrainingExecutionError("plan.timestep_bands must be a sequence")
    cell_configs = plan.get("cell_configs")
    if not isinstance(cell_configs, Mapping):
        raise TrainingExecutionError("plan must expose timestep_band_metadata or cell_configs")

    grouped_cells = {str(timestep_band): [] for timestep_band in timestep_bands}
    for raw_cell_id, raw_cell in cell_configs.items():
        if not isinstance(raw_cell, Mapping):
            raise TrainingExecutionError(f"plan.cell_configs[{raw_cell_id!r}] must be a mapping")
        cell = dict(raw_cell)
        timestep_band = str(cell.get("timestep_band", "")).strip()
        if timestep_band not in grouped_cells:
            raise TrainingExecutionError(
                f"plan.cell_configs[{raw_cell_id!r}] references unknown timestep_band {timestep_band!r}"
            )
        grouped_cells[timestep_band].append((str(raw_cell_id), cell))

    metadata: dict[str, dict[str, Any]] = {}
    for timestep_band in timestep_bands:
        ordered_cells = sorted(grouped_cells[str(timestep_band)], key=lambda item: item[0])
        positive_rank_cell_count = sum(1 for _cell_id, cell in ordered_cells if int(cell["rank"]) > 0)
        metadata[str(timestep_band)] = {
            "timestep_band": str(timestep_band),
            "cell_ids": [cell_id for cell_id, _cell in ordered_cells],
            "cell_count": len(ordered_cells),
            "positive_rank_cell_count": positive_rank_cell_count,
            "zero_rank_cell_count": len(ordered_cells) - positive_rank_cell_count,
            "has_trainable_params": positive_rank_cell_count > 0,
            "noop": positive_rank_cell_count == 0,
        }
    return metadata


def _plan_timestep_band_metadata(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = plan.get("timestep_band_metadata")
    if isinstance(metadata, Mapping) and metadata:
        return {
            str(timestep_band): dict(payload)
            for timestep_band, payload in metadata.items()
        }
    return _synthesize_timestep_band_metadata(plan)


def _normalize_forced_timestep_band_sequence(
    value: Sequence[str] | None,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TrainingExecutionError("forced_timestep_band_sequence must be a sequence of band names")

    normalized: list[str] = []
    for index, raw_band_name in enumerate(value):
        band_name = str(raw_band_name).strip()
        if not band_name:
            raise TrainingExecutionError(
                f"forced_timestep_band_sequence[{index}] must be a non-empty band name"
            )
        normalized.append(band_name)
    if not normalized:
        raise TrainingExecutionError("forced_timestep_band_sequence must not be empty")
    return normalized


def _set_active_timestep_band_marker(unet: Any, timestep_band: str | None) -> None:
    try:
        setattr(unet, "_rd_lora_active_timestep_band", "" if timestep_band is None else str(timestep_band))
    except Exception:
        return None


def _active_timestep_band_marker(unet: Any) -> str:
    value = getattr(unet, "_rd_lora_active_timestep_band", "")
    return str(value).strip()


def _adapter_spec_lookup(adapter_specs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(spec["adapter_name"]): dict(spec)
        for spec in adapter_specs
    }


def _normalize_optional_adapter_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _activate_adapter(unet: Any, *, adapter_name: str, adapter_spec_by_name: Mapping[str, Mapping[str, Any]]) -> None:
    if adapter_name not in adapter_spec_by_name:
        raise TrainingExecutionError(f"Unknown adapter_name {adapter_name!r}")
    spec = adapter_spec_by_name[adapter_name]
    if bool(spec["noop"]):
        raise TrainingExecutionError(f"Cannot activate noop adapter spec {adapter_name!r}")
    if hasattr(unet, "enable_adapters"):
        unet.enable_adapters()
    unet.set_adapter(adapter_name)


def _optimizer_parameters(optimizer: Any) -> list[Any]:
    param_groups = getattr(optimizer, "param_groups", None)
    if isinstance(param_groups, Sequence):
        parameters: list[Any] = []
        for group_index, group in enumerate(param_groups):
            if not isinstance(group, Mapping):
                raise TrainingExecutionError(f"optimizer.param_groups[{group_index}] must be a mapping")
            group_params = group.get("params")
            if not isinstance(group_params, Sequence):
                raise TrainingExecutionError(f"optimizer.param_groups[{group_index}].params must be a sequence")
            parameters.extend(list(group_params))
        return parameters
    parameters = getattr(optimizer, "parameters", None)
    if isinstance(parameters, Sequence):
        return list(parameters)
    raise TrainingExecutionError("optimizer must expose param_groups or parameters for coverage checks")


def _assert_no_noop_adapter_specs(adapter_spec_by_name: Mapping[str, Mapping[str, Any]]) -> None:
    noop_adapter_names = sorted(
        adapter_name
        for adapter_name, spec in adapter_spec_by_name.items()
        if bool(spec.get("noop", False))
    )
    if noop_adapter_names:
        raise TrainingExecutionError(
            f"Noop adapter specs must not exist in adapter_spec_by_name; discovered={noop_adapter_names}"
        )


def _assert_optimizer_parameter_coverage(
    *,
    trainable_parameters: Sequence[Any],
    optimizer: Any,
) -> list[str]:
    optimizer_parameters = _optimizer_parameters(optimizer)
    optimizer_counts: dict[int, int] = {}
    for parameter in optimizer_parameters:
        parameter_id = id(parameter)
        optimizer_counts[parameter_id] = optimizer_counts.get(parameter_id, 0) + 1
    missing = [parameter for parameter in trainable_parameters if optimizer_counts.get(id(parameter), 0) == 0]
    duplicated = [parameter for parameter in trainable_parameters if optimizer_counts.get(id(parameter), 0) > 1]
    if missing or duplicated:
        raise TrainingExecutionError(
            "Optimizer parameter groups must cover each trainable adapter parameter exactly once; "
            f"missing={len(missing)} duplicated={len(duplicated)}"
        )


def _assert_optimizer_has_grad(optimizer: Any) -> None:
    if not any(getattr(parameter, "grad", None) is not None for parameter in _optimizer_parameters(optimizer)):
        raise TrainingExecutionError("Backward pass did not produce gradients for optimizer parameters")


def select_active_adapter_name(
    *,
    plan: Mapping[str, Any],
    timesteps: Sequence[int],
    num_train_timesteps: int,
    active_timestep_band: str | None = None,
) -> str | None:
    if active_timestep_band is None:
        active_timestep_band = select_active_timestep_band_name(
            plan=plan,
            timesteps=timesteps,
            num_train_timesteps=num_train_timesteps,
        )
    routing = dict(plan["routing"])
    if routing["mode"] == "static":
        adapter_name = _normalize_optional_adapter_name(routing.get("default_adapter_name"))
    elif routing["mode"] == "timestep_band":
        adapter_name = _normalize_optional_adapter_name(dict(routing["table"][active_timestep_band]).get("adapter_name"))
    else:
        raise TrainingExecutionError(f"Unsupported routing mode {routing['mode']!r}")
    if routing["mode"] == "static" and adapter_name is None:
        raise TrainingExecutionError(f"Routing entry for {active_timestep_band!r} is missing adapter_name")
    return adapter_name


def select_active_timestep_band_name(
    *,
    plan: Mapping[str, Any],
    timesteps: Sequence[int],
    num_train_timesteps: int,
) -> str:
    try:
        return resolve_timestep_band_name_for_training_timesteps(
            _plan_timestep_band_routes(plan),
            timesteps=timesteps,
            num_train_timesteps=num_train_timesteps,
        )
    except TimestepRoutingError as exc:
        raise TrainingExecutionError(f"Timestep routing failed: {exc}") from exc


def _sample_timestep_band_training_timesteps(
    *,
    torch_module: Any,
    noise_scheduler: Any,
    batch: Mapping[str, Any],
) -> Any:
    pixel_values = batch.get("pixel_values")
    if pixel_values is None:
        raise TrainingExecutionError("batch must include pixel_values for timestep-band routing")
    shape = getattr(pixel_values, "shape", None)
    if shape is None or len(shape) < 1:
        raise TrainingExecutionError("batch.pixel_values must expose a batch dimension for timestep-band routing")
    batch_size = int(shape[0])
    if batch_size <= 0:
        raise TrainingExecutionError("batch.pixel_values must include at least one example")
    randint_kwargs: dict[str, Any] = {}
    device = getattr(pixel_values, "device", None)
    if device is not None:
        randint_kwargs["device"] = device
    return torch_module.randint(
        0,
        int(noise_scheduler.config.num_train_timesteps),
        (batch_size,),
        **randint_kwargs,
    ).long()


def _plan_timestep_band_batch_route(
    *,
    torch_module: Any,
    plan: Mapping[str, Any],
    components: Mapping[str, Any],
    batch: Mapping[str, Any],
    timestep_band_metadata: Mapping[str, Mapping[str, Any]],
    forced_timestep: int | None = None,
) -> dict[str, Any]:
    noise_scheduler = components["scheduler"]
    if forced_timestep is None:
        timesteps = _sample_timestep_band_training_timesteps(
            torch_module=torch_module,
            noise_scheduler=noise_scheduler,
            batch=batch,
        )
    else:
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            raise TrainingExecutionError(
                "batch must include pixel_values for forced timestep-band routing"
            )
        shape = getattr(pixel_values, "shape", None)
        if shape is None or len(shape) < 1:
            raise TrainingExecutionError(
                "batch.pixel_values must expose a batch dimension for forced timestep-band routing"
            )
        batch_size = int(shape[0])
        if batch_size <= 0:
            raise TrainingExecutionError("batch.pixel_values must include at least one example")
        full_kwargs: dict[str, Any] = {}
        device = getattr(pixel_values, "device", None)
        if device is not None:
            full_kwargs["device"] = device
        long_dtype = getattr(torch_module, "long", None)
        if long_dtype is not None:
            full_kwargs["dtype"] = long_dtype
        timesteps = torch_module.full((batch_size,), int(forced_timestep), **full_kwargs)
        if hasattr(timesteps, "long"):
            timesteps = timesteps.long()
    timestep_values = [int(value) for value in timesteps.detach().flatten().tolist()]
    active_timestep_band = select_active_timestep_band_name(
        plan=plan,
        timesteps=timestep_values,
        num_train_timesteps=int(noise_scheduler.config.num_train_timesteps),
    )
    if active_timestep_band not in timestep_band_metadata:
        raise TrainingExecutionError(f"Unknown routed timestep band {active_timestep_band!r}")
    routing = dict(plan["routing"])
    if str(routing.get("mode")) != "timestep_band":
        raise TrainingExecutionError("Timestep-band batch routing requires plan.routing.mode='timestep_band'")
    routing_entry = dict(routing["table"][active_timestep_band])
    routed_timestep_band = dict(timestep_band_metadata[active_timestep_band])
    noop = bool(routing_entry.get("noop", False))
    if noop != bool(routed_timestep_band["noop"]):
        raise TrainingExecutionError(
            f"Routing noop mismatch for {active_timestep_band!r}: route={noop} metadata={routed_timestep_band['noop']}"
        )
    adapter_name = _normalize_optional_adapter_name(routing_entry.get("adapter_name"))
    return {
        "timesteps": timesteps,
        "active_timestep_band": active_timestep_band,
        "adapter_name": adapter_name,
        "noop": noop,
    }


def _resolve_routed_timestep_band_metadata(
    *,
    plan: Mapping[str, Any],
    unet: Any,
    active_adapter_name: str,
    timestep_band_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    active_timestep_band = _active_timestep_band_marker(unet)
    if active_timestep_band:
        if active_timestep_band not in timestep_band_metadata:
            raise TrainingExecutionError(f"Unknown routed timestep band {active_timestep_band!r}")
        return dict(timestep_band_metadata[active_timestep_band])

    routing = dict(plan["routing"])
    if routing["mode"] == "static":
        noop_values = {bool(dict(metadata).get("noop", False)) for metadata in timestep_band_metadata.values()}
        if len(noop_values) != 1:
            raise TrainingExecutionError(
                "Static routing cannot infer routed timestep-band noop state from mixed per-band metadata"
            )
        first_band_name = sorted(str(timestep_band) for timestep_band in timestep_band_metadata)[0]
        return dict(timestep_band_metadata[first_band_name])
    if routing["mode"] != "timestep_band":
        raise TrainingExecutionError(f"Unsupported routing mode {routing['mode']!r}")

    matches = [
        str(timestep_band)
        for timestep_band, route in dict(routing["table"]).items()
        if str(dict(route).get("adapter_name", "")).strip() == str(active_adapter_name).strip()
    ]
    if len(matches) != 1:
        raise TrainingExecutionError(
            f"Unable to resolve timestep band from adapter_name {active_adapter_name!r}; matches={matches}"
        )
    return dict(timestep_band_metadata[matches[0]])


def _compute_time_ids(
    *,
    torch_module: Any,
    accelerator: Any,
    weight_dtype: Any,
    resolution: int,
    original_size: Sequence[int],
    crop_top_left: Sequence[int],
) -> Any:
    add_time_ids = list(original_size) + list(crop_top_left) + [int(resolution), int(resolution)]
    return torch_module.tensor([add_time_ids]).to(accelerator.device, dtype=weight_dtype)


def compute_step_loss(
    *,
    torch_module: Any,
    official_module: Any,
    accelerator: Any,
    training_args: SimpleNamespace,
    plan: Mapping[str, Any],
    adapter_spec_by_name: Mapping[str, Mapping[str, Any]],
    components: Mapping[str, Any],
    runtime_state: Mapping[str, Any],
    batch: Mapping[str, Any],
    timesteps: Any | None = None,
    active_timestep_band: str | None = None,
    active_adapter_name: str | None = None,
) -> Any:
    vae = components["vae"]
    unet = components["unet"]
    noise_scheduler = components["scheduler"]
    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
    model_input = vae.encode(pixel_values).latent_dist.sample()
    latents_mean = runtime_state["latents_mean"]
    latents_std = runtime_state["latents_std"]
    if latents_mean is None and latents_std is None:
        model_input = model_input * vae.config.scaling_factor
        if training_args.pretrained_vae_model_name_or_path is None:
            model_input = model_input.to(runtime_state["weight_dtype"])
    else:
        latents_mean = latents_mean.to(device=model_input.device, dtype=model_input.dtype)
        latents_std = latents_std.to(device=model_input.device, dtype=model_input.dtype)
        model_input = (model_input - latents_mean) * vae.config.scaling_factor / latents_std
        model_input = model_input.to(dtype=runtime_state["weight_dtype"])

    noise = torch_module.randn_like(model_input)
    batch_size = model_input.shape[0]
    if timesteps is None:
        timesteps = torch_module.randint(
            0,
            int(noise_scheduler.config.num_train_timesteps),
            (batch_size,),
            device=model_input.device,
        ).long()
    else:
        timesteps = timesteps.to(device=model_input.device).long()

    timestep_values: list[int] | None = None
    if active_timestep_band is None or active_adapter_name is None:
        timestep_values = [int(value) for value in timesteps.detach().flatten().tolist()]
    if active_timestep_band is None:
        assert timestep_values is not None
        active_timestep_band = select_active_timestep_band_name(
            plan=plan,
            timesteps=timestep_values,
            num_train_timesteps=int(noise_scheduler.config.num_train_timesteps),
        )
    if active_adapter_name is None:
        assert timestep_values is not None
        active_adapter_name = select_active_adapter_name(
            plan=plan,
            timesteps=timestep_values,
            num_train_timesteps=int(noise_scheduler.config.num_train_timesteps),
            active_timestep_band=active_timestep_band,
        )
    if active_adapter_name is None:
        raise TrainingExecutionError(
            f"compute_step_loss cannot run without a concrete adapter_name for timestep band {active_timestep_band!r}"
        )

    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
    prompt_embeds, pooled_prompt_embeds = official_module.encode_prompt(
        [components["text_encoder"], components["text_encoder_2"]],
        [components["tokenizer"], components["tokenizer_2"]],
        training_args.instance_prompt,
    )
    prompt_embeds = prompt_embeds.to(accelerator.device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
    add_time_ids = torch_module.cat(
        [
            _compute_time_ids(
                torch_module=torch_module,
                accelerator=accelerator,
                weight_dtype=runtime_state["weight_dtype"],
                resolution=training_args.resolution,
                original_size=original_size,
                crop_top_left=crop_top_left,
            )
            for original_size, crop_top_left in zip(batch["original_sizes"], batch["crop_top_lefts"])
        ]
    )
    model_pred = unet(
        noisy_model_input,
        timesteps,
        prompt_embeds.repeat(batch_size, 1, 1),
        added_cond_kwargs={
            "time_ids": add_time_ids,
            "text_embeds": pooled_prompt_embeds.repeat(batch_size, 1),
        },
        return_dict=False,
    )[0]

    prediction_type = str(noise_scheduler.config.prediction_type)
    if prediction_type == "epsilon":
        target = noise
    elif prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
    else:
        raise TrainingExecutionError(f"Unknown prediction type {prediction_type!r}")

    loss = official_module.F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss


def save_checkpoint(*, accelerator: Any, output_dir: Path, global_step: int) -> Path:
    checkpoint_path = (output_dir / f"checkpoint-{global_step}").resolve()
    accelerator.save_state(str(checkpoint_path))
    if not checkpoint_path.exists():
        raise TrainingExecutionError(f"Checkpoint save did not create {checkpoint_path}")
    return checkpoint_path


def _peak_vram_mib_from_torch(torch_module: Any) -> float | None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return None
    max_memory_allocated = getattr(cuda, "max_memory_allocated", None)
    if not callable(max_memory_allocated):
        return None
    raw_bytes = float(max_memory_allocated())
    if raw_bytes <= 0.0:
        return None
    return round(raw_bytes / float(1024**2), 3)


def _validate_metrics_payload(metrics: Mapping[str, Any]) -> None:
    for key in REQUIRED_METRIC_KEYS:
        if key not in metrics:
            raise TrainingExecutionError(f"metrics.json is missing key {key!r}")
    if int(metrics["global_step"]) <= 0:
        raise TrainingExecutionError("metrics.global_step must be positive")
    if int(metrics["skipped_noop_band_batches"]) < 0:
        raise TrainingExecutionError("metrics.skipped_noop_band_batches must be non-negative")
    skipped_noop_band_fraction = float(metrics["skipped_noop_band_fraction"])
    if skipped_noop_band_fraction < 0.0 or skipped_noop_band_fraction > 1.0:
        raise TrainingExecutionError("metrics.skipped_noop_band_fraction must be in [0.0, 1.0]")
    if int(metrics["active_band_optimizer_steps"]) != int(metrics["global_step"]):
        raise TrainingExecutionError("metrics.active_band_optimizer_steps must equal metrics.global_step")


def write_success_artifacts(
    *,
    output_dir: Path,
    run_provenance: Mapping[str, Any],
    train_summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    checkpoint_info: Mapping[str, Any],
) -> list[str]:
    assert_real_gpu_training_provenance(run_provenance)
    if str(train_summary.get("status")) != TRAINING_SUCCESS_STATUS:
        raise TrainingExecutionError(
            f"Successful train_summary.json must use status={TRAINING_SUCCESS_STATUS!r}"
        )
    checkpoint_path = Path(str(checkpoint_info["checkpoint_path"])).expanduser().resolve()
    if not bool(checkpoint_info["checkpoint_exists"]) or not checkpoint_path.exists():
        raise TrainingExecutionError("Success artifacts require a real checkpoint path on disk")
    _validate_metrics_payload(metrics)

    save_json(output_dir / "checkpoint_info.json", checkpoint_info)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "run_provenance.json", run_provenance)
    save_json(output_dir / "train_summary.json", train_summary)


def build_run_provenance_payload(
    *,
    repo_root: Path,
    training_args: SimpleNamespace,
    backend: str,
    task: str,
    allocation_manifest_path: Path,
    peak_vram_mib: float,
    official_script: Path,
    components: Mapping[str, Any],
    active_adapter_name: str,
) -> dict[str, Any]:
    runtime = collect_runtime_metadata(
        repo_root=repo_root,
        accelerate_config_file=training_args.accelerate_config_file,
        require_cuda=True,
        expected_diffusers_substring=training_args.expected_diffusers_substring,
    )
    payload = {
        "schema_version": "1.0",
        "run_mode": "real_gpu",
        "used_gpu": True,
        "python": runtime["python"],
        "torch": runtime["torch"],
        "torch_cuda_is_available": runtime["torch_cuda_is_available"],
        "torch_cuda_version": runtime["torch_cuda_version"],
        "torch_device_count": runtime["torch_device_count"],
        "gpu_names": runtime["gpu_names"],
        "diffusers": runtime["diffusers"],
        "diffusers_file": runtime["diffusers_file"],
        "accelerate_config_file": runtime["accelerate_config_file"],
        "git_commit": runtime["git_commit"],
        "backend": str(backend),
        "task": str(task),
        "allocation_manifest": str(allocation_manifest_path),
        "peak_vram_mib": round(float(peak_vram_mib), 3),
        "timestamp_utc": runtime["timestamp_utc"],
        "official_substrate_script": str(official_script),
        "loaded_components": describe_loaded_sdxl_components(components),
        "active_adapter_last": str(active_adapter_name),
    }
    assert_real_gpu_training_provenance(payload)
    return payload


def validate_training_output_artifacts(output_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(output_dir).expanduser().resolve()
    missing_files = [name for name in REQUIRED_SUCCESS_FILES if not (run_dir / name).is_file()]
    if missing_files:
        raise TrainingExecutionError(f"{run_dir} is missing required success artifacts: {missing_files}")

    provenance = json.loads((run_dir / "run_provenance.json").read_text(encoding="utf-8"))
    train_summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    checkpoint_info = json.loads((run_dir / "checkpoint_info.json").read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not isinstance(train_summary, dict):
        raise TrainingExecutionError(f"{run_dir} must contain JSON object artifacts")
    if not isinstance(metrics, dict) or not isinstance(checkpoint_info, dict):
        raise TrainingExecutionError(f"{run_dir} must contain JSON object artifacts")

    assert_real_gpu_training_provenance(provenance)
    missing_summary = [key for key in REQUIRED_SUMMARY_KEYS if key not in train_summary]
    if missing_summary:
        raise TrainingExecutionError(f"train_summary.json is missing keys: {missing_summary}")
    if str(train_summary["status"]).strip() == "planning_only":
        raise TrainingExecutionError("train_summary.json cannot use planning_only as a successful status")
    if str(train_summary["status"]).strip() != TRAINING_SUCCESS_STATUS:
        raise TrainingExecutionError(
            f"train_summary.json status must be {TRAINING_SUCCESS_STATUS!r}"
        )
    missing_checkpoint_info = [key for key in REQUIRED_CHECKPOINT_INFO_KEYS if key not in checkpoint_info]
    if missing_checkpoint_info:
        raise TrainingExecutionError(f"checkpoint_info.json is missing keys: {missing_checkpoint_info}")
    _validate_metrics_payload(metrics)

    checkpoint_path = Path(str(checkpoint_info["checkpoint_path"])).expanduser().resolve()
    if str(train_summary["checkpoint_path"]) != str(checkpoint_path):
        raise TrainingExecutionError("train_summary.checkpoint_path must match checkpoint_info.checkpoint_path")
    if not bool(checkpoint_info["checkpoint_exists"]):
        raise TrainingExecutionError("checkpoint_info.checkpoint_exists must be true for a successful run")
    if not checkpoint_path.exists():
        raise TrainingExecutionError(f"checkpoint_path does not exist on disk: {checkpoint_path}")
    return {
        "run_provenance": provenance,
        "train_summary": train_summary,
        "metrics": metrics,
        "checkpoint_info": checkpoint_info,
    }


def execute_training_run(
    *,
    repo_root: str | Path,
    config: Mapping[str, Any],
    task: str,
    backend: str,
    allocation_path: str | Path,
    output_dir: str | Path,
    run_mode: str,
    forced_timestep_band_sequence: Sequence[str] | None = None,
) -> dict[str, Any]:
    if str(run_mode) != "real_gpu":
        raise TrainingExecutionError("run_mode must be 'real_gpu'")

    repo_root_path = Path(repo_root).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    allocation_manifest_path = Path(allocation_path).expanduser().resolve()
    manifest = load_allocation_manifest(allocation_manifest_path)
    if str(backend) != str(manifest["backend"]):
        raise TrainingExecutionError(
            f"--backend {backend!r} does not match manifest backend {manifest['backend']!r}"
        )

    training_args = build_training_args(
        repo_root=repo_root_path,
        config=config,
        task=task,
        output_dir=output_dir_path,
    )
    paths = _require_mapping(config.get("paths", {}), name="config.paths")
    official_script = validate_official_sdxl_dreambooth_script(
        repo_root_path,
        paths.get("official_diffusers_script"),
    )

    torch_module = load_torch()
    _require_real_gpu(torch_module)
    torch_module.cuda.reset_peak_memory_stats()

    plan = build_backend_plan(
        manifest=manifest,
        task=task,
        config=config,
    )
    accelerator = create_accelerator(training_args)
    components = load_official_sdxl_components(
        pretrained_model_name_or_path=training_args.pretrained_model_name_or_path,
        revision=training_args.revision,
        variant=training_args.variant,
        pretrained_vae_model_name_or_path=training_args.pretrained_vae_model_name_or_path,
        script_path=official_script,
    )
    official_module = components["module"]
    runtime_state = _configure_components_for_training(
        torch_module=torch_module,
        accelerator=accelerator,
        official_module=official_module,
        training_args=training_args,
        components=components,
    )
    adapter_specs = build_concrete_adapter_specs(plan, unet=components["unet"])
    created_adapter_names = apply_adapter_specs(
        official_module=official_module,
        unet=components["unet"],
        adapter_specs=adapter_specs,
        use_rslora=bool(plan["use_rslora"]),
        lora_dropout=float(training_args.lora_dropout),
    )
    assert_adapter_param_counts(components["unet"], created_adapter_names)
    adapter_spec_by_name = _adapter_spec_lookup(adapter_specs)
    _assert_no_noop_adapter_specs(adapter_spec_by_name)
    trainable_parameters = collect_trainable_parameters(components["unet"], adapter_names=created_adapter_names)
    if training_args.mixed_precision == "fp16":
        upcast_trainable_params_to_fp32(torch_module, trainable_parameters)
    assert_all_params_fp32(torch_module, trainable_parameters)
    optimizer = create_optimizer(
        torch_module=torch_module,
        trainable_parameters=trainable_parameters,
        training_args=training_args,
    )
    _assert_optimizer_parameter_coverage(
        trainable_parameters=trainable_parameters,
        optimizer=optimizer,
    )
    train_dataloader = create_train_dataloader(
        torch_module=torch_module,
        official_module=official_module,
        training_args=training_args,
    )
    lr_scheduler = create_lr_scheduler(
        official_module=official_module,
        optimizer=optimizer,
        accelerator=accelerator,
        train_dataloader=train_dataloader,
        training_args=training_args,
    )

    components["unet"], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        components["unet"],
        optimizer,
        train_dataloader,
        lr_scheduler,
    )
    timestep_band_metadata = _plan_timestep_band_metadata(plan)
    routing_table = _plan_timestep_band_routes(plan)
    static_adapter_name = _normalize_optional_adapter_name(dict(plan["routing"]).get("default_adapter_name"))
    requested_forced_timestep_band_sequence = (
        forced_timestep_band_sequence
        if forced_timestep_band_sequence is not None
        else plan.get("forced_timestep_band_sequence")
    )
    normalized_forced_timestep_band_sequence = _normalize_forced_timestep_band_sequence(
        requested_forced_timestep_band_sequence
    )
    if dict(plan["routing"])["mode"] == "static":
        if static_adapter_name is None:
            raise TrainingExecutionError("Static routing requires a concrete default_adapter_name")
        _activate_adapter(
            components["unet"],
            adapter_name=static_adapter_name,
            adapter_spec_by_name=adapter_spec_by_name,
        )

    if getattr(accelerator, "is_main_process", True) and hasattr(accelerator, "init_trackers"):
        accelerator.init_trackers("dreambooth-lora-sd-xl", config={"task": task, "backend": backend})

    global_step = 0
    skipped_noop_band_batches = 0
    total_batches_processed = 0
    recorded_losses: list[float] = []
    last_checkpoint_path: Path | None = None
    last_active_adapter_name = static_adapter_name
    max_train_steps = int(training_args.max_train_steps)
    routing_mode = str(dict(plan["routing"]).get("mode", ""))
    if normalized_forced_timestep_band_sequence is not None and routing_mode != "timestep_band":
        raise TrainingExecutionError(
            "forced_timestep_band_sequence requires plan.routing.mode='timestep_band'"
        )
    forced_timestep_band_index = 0
    forced_timestep_band_sequence_exhausted = False

    while normalized_forced_timestep_band_sequence is not None or global_step < max_train_steps:
        components["unet"].train()
        batches_in_epoch = 0
        for batch in train_dataloader:
            forced_timestep: int | None = None
            if normalized_forced_timestep_band_sequence is not None:
                if forced_timestep_band_index >= len(normalized_forced_timestep_band_sequence):
                    forced_timestep_band_sequence_exhausted = True
                    break
                forced_timestep_band_name = normalized_forced_timestep_band_sequence[
                    forced_timestep_band_index
                ]
                forced_timestep_band_index += 1
                try:
                    forced_timestep = resolve_deterministic_timestep_for_band(
                        routing_table,
                        forced_timestep_band_name,
                        num_train_timesteps=int(components["scheduler"].config.num_train_timesteps),
                    )
                except TimestepRoutingError as exc:
                    raise TrainingExecutionError(f"Forced timestep routing failed: {exc}") from exc
            batches_in_epoch += 1
            total_batches_processed += 1
            loss: Any | None = None
            optimizer_step_completed = False
            step_timesteps: Any | None = None
            active_timestep_band: str | None = None
            adapter_name: str | None = None
            _set_active_timestep_band_marker(components["unet"], None)
            if routing_mode == "timestep_band":
                route = _plan_timestep_band_batch_route(
                    torch_module=torch_module,
                    plan=plan,
                    components=components,
                    batch=batch,
                    timestep_band_metadata=timestep_band_metadata,
                    forced_timestep=forced_timestep,
                )
                step_timesteps = route["timesteps"]
                active_timestep_band = str(route["active_timestep_band"])
                adapter_name = route["adapter_name"]
                if bool(route["noop"]):
                    if adapter_name is not None:
                        raise TrainingExecutionError(
                            f"Routed noop timestep band {active_timestep_band!r} must not expose an adapter_name"
                        )
                    skipped_noop_band_batches += 1
                    continue
                if adapter_name is None:
                    raise TrainingExecutionError(
                        f"Trainable timestep band {active_timestep_band!r} must expose a concrete adapter_name"
                    )
                if adapter_name not in adapter_spec_by_name:
                    raise TrainingExecutionError(f"Unknown routed adapter_name {adapter_name!r}")
                spec = adapter_spec_by_name[adapter_name]
                if bool(spec["noop"]):
                    raise TrainingExecutionError(f"Routed adapter_name {adapter_name!r} cannot resolve to a noop spec")
                if hasattr(components["unet"], "enable_adapters"):
                    components["unet"].enable_adapters()
                components["unet"].set_adapter(adapter_name)
                last_active_adapter_name = adapter_name
                _set_active_timestep_band_marker(components["unet"], active_timestep_band)
            else:
                adapter_name = static_adapter_name
            with accelerator.accumulate(components["unet"]) if hasattr(accelerator, "accumulate") else nullcontext():
                loss = compute_step_loss(
                    torch_module=torch_module,
                    official_module=official_module,
                    accelerator=accelerator,
                    training_args=training_args,
                    plan=plan,
                    adapter_spec_by_name=adapter_spec_by_name,
                    components=components,
                    runtime_state=runtime_state,
                    batch=batch,
                    timesteps=step_timesteps,
                    active_timestep_band=active_timestep_band,
                    active_adapter_name=adapter_name,
                )
                if isinstance(loss, tuple):
                    reported_loss, reported_adapter_name = loss
                    normalized_reported_adapter_name = _normalize_optional_adapter_name(reported_adapter_name)
                    if normalized_reported_adapter_name != adapter_name:
                        raise TrainingExecutionError(
                            "compute_step_loss reported a mismatched adapter_name; "
                            f"expected={adapter_name!r} reported={normalized_reported_adapter_name!r}"
                        )
                    loss = reported_loss
                accelerator.backward(loss)
                _assert_optimizer_has_grad(optimizer)
                if getattr(accelerator, "sync_gradients", True) and hasattr(accelerator, "clip_grad_norm_"):
                    accelerator.clip_grad_norm_(trainable_parameters, training_args.max_grad_norm)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_completed = bool(getattr(accelerator, "sync_gradients", True))

            if optimizer_step_completed:
                assert loss is not None
                global_step += 1
                loss_value = float(loss.detach().item())
                recorded_losses.append(loss_value)
                if hasattr(accelerator, "log"):
                    accelerator.log({"loss": loss_value}, step=global_step)
                if global_step % int(training_args.checkpointing_steps) == 0:
                    last_checkpoint_path = save_checkpoint(
                        accelerator=accelerator,
                        output_dir=output_dir_path,
                        global_step=global_step,
                    )
            if normalized_forced_timestep_band_sequence is None and global_step >= max_train_steps:
                break
        if forced_timestep_band_sequence_exhausted:
            break
        if batches_in_epoch <= 0:
            raise TrainingExecutionError("train_dataloader must yield at least one batch")

    if global_step <= 0:
        raise TrainingExecutionError("Training exited without any optimizer steps")
    if last_checkpoint_path is None:
        last_checkpoint_path = save_checkpoint(
            accelerator=accelerator,
            output_dir=output_dir_path,
            global_step=global_step,
        )
    if not last_checkpoint_path.exists():
        raise TrainingExecutionError(f"Checkpoint path does not exist after save: {last_checkpoint_path}")

    peak_vram_mib = _peak_vram_mib_from_torch(torch_module)
    if peak_vram_mib is None or float(peak_vram_mib) <= 0.0:
        raise TrainingExecutionError("peak_vram_mib must be measured from CUDA stats and be > 0 in real_gpu mode")

    run_provenance = build_run_provenance_payload(
        repo_root=repo_root_path,
        training_args=training_args,
        backend=backend,
        task=task,
        allocation_manifest_path=allocation_manifest_path,
        peak_vram_mib=peak_vram_mib,
        official_script=official_script,
        components=components,
        active_adapter_name=str(last_active_adapter_name),
    )
    checkpoint_info = {
        "checkpoint_path": str(last_checkpoint_path),
        "checkpoint_exists": bool(last_checkpoint_path.exists()),
    }
    metrics = {
        "global_step": int(global_step),
        "train_loss_last": float(recorded_losses[-1]),
        "train_loss_mean": round(sum(recorded_losses) / len(recorded_losses), 6),
        "learning_rate_last": float(lr_scheduler.get_last_lr()[0]) if lr_scheduler is not None else None,
        "skipped_noop_band_batches": int(skipped_noop_band_batches),
        "skipped_noop_band_fraction": round(
            float(skipped_noop_band_batches) / float(total_batches_processed),
            6,
        ),
        "active_band_optimizer_steps": int(global_step),
    }
    train_summary = {
        "status": TRAINING_SUCCESS_STATUS,
        "backend": str(backend),
        "task": str(task),
        "allocation_manifest": str(allocation_manifest_path),
        "train_steps": int(global_step),
        "checkpoint_path": str(last_checkpoint_path),
    }
    write_success_artifacts(
        output_dir=output_dir_path,
        run_provenance=run_provenance,
        train_summary=train_summary,
        metrics=metrics,
        checkpoint_info=checkpoint_info,
    )
    validate_training_output_artifacts(output_dir_path)
    return {
        "run_provenance": run_provenance,
        "train_summary": train_summary,
        "metrics": metrics,
        "checkpoint_info": checkpoint_info,
    }


__all__ = [
    "REQUIRED_SUCCESS_FILES",
    "TRAINING_SUCCESS_STATUS",
    "TrainingExecutionError",
    "assert_all_params_fp32",
    "build_training_args",
    "assert_adapter_param_counts",
    "collect_trainable_parameters",
    "build_run_provenance_payload",
    "create_accelerator",
    "create_lr_scheduler",
    "create_optimizer",
    "create_train_dataloader",
    "execute_training_run",
    "load_torch",
    "save_checkpoint",
    "upcast_trainable_params_to_fp32",
    "validate_training_output_artifacts",
    "write_success_artifacts",
]
