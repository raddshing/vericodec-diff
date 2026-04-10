from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps

from rd_lora.cells import (
    DEFAULT_CANDIDATE_RANKS,
    DEFAULT_LAYER_GROUP_COUNT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_TARGET_MODULES,
    DEFAULT_TIMESTEP_BAND_COUNT,
    CellSchema,
    ProbeCell,
    build_cell_schema,
    parse_candidate_ranks_argument,
)
from rd_lora.features import (
    UTILITY_RECORD_FIELDNAMES,
    build_probe_row,
    build_probe_summary,
    collect_run_provenance,
    validate_probe_payloads,
)
from rd_lora.substrate.diffusers_sdxl import deep_update, save_json


DEFAULT_IMAGE_SIZE = 1024
SUPPORTED_IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".webp")
RESAMPLING = getattr(Image, "Resampling", Image)


class ProbeValidationError(ValueError):
    """Raised when the probe configuration or runtime is invalid."""


@dataclass(frozen=True)
class EncodedExample:
    image_path: str
    latents: Any
    prompt_embeds: Any
    pooled_prompt_embeds: Any
    add_time_ids: Any


def default_probe_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "output_root": "outputs/rd_lora/probe",
            "accelerate_config": "configs/accelerate/single_gpu_fp16.yaml",
            "expected_diffusers_substring": "huggingface_diffusers",
        },
        "model": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "pretrained_vae_model_name_or_path": None,
            "revision": None,
            "variant": None,
            "mixed_precision": "fp16",
        },
        "schema": {
            "layer_group_count": DEFAULT_LAYER_GROUP_COUNT,
            "timestep_band_count": DEFAULT_TIMESTEP_BAND_COUNT,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "candidate_ranks": list(DEFAULT_CANDIDATE_RANKS),
        },
        "probe": {
            "candidate_ranks": list(DEFAULT_CANDIDATE_RANKS),
            "probe_inner_steps": 1,
            "max_train_batches": 1,
            "max_val_batches": 1,
            "learning_rate": 1.0e-4,
            "target_modules": list(DEFAULT_TARGET_MODULES),
            "use_rslora": True,
        },
        "tasks": {
            "subject_personalization": {
                "train_data_dir": "tests/fixtures/rdlora_pilot",
                "validation_prompt": "A studio portrait of the same subject.",
            },
            "style_domain": {
                "train_data_dir": "tests/fixtures/rdlora_pilot",
                "validation_prompt": "A poster in the target style domain.",
            },
        },
    }


def resolve_probe_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_probe_config(), raw_config)
    paths = dict(config.get("paths", {}))
    schema = dict(config.get("schema", {}))
    probe = dict(config.get("probe", {}))
    model = dict(config.get("model", {}))
    tasks = dict(config.get("tasks", {}))

    target_modules = tuple(str(value).strip() for value in probe.get("target_modules", DEFAULT_TARGET_MODULES))
    if target_modules != DEFAULT_TARGET_MODULES:
        raise ProbeValidationError(f"probe.target_modules must be exactly {DEFAULT_TARGET_MODULES!r}")

    candidate_ranks = parse_candidate_ranks_argument(
        probe.get("candidate_ranks", schema.get("candidate_ranks", DEFAULT_CANDIDATE_RANKS))
    )
    num_inference_steps = int(schema.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    layer_group_count = int(schema.get("layer_group_count", DEFAULT_LAYER_GROUP_COUNT))
    timestep_band_count = int(schema.get("timestep_band_count", DEFAULT_TIMESTEP_BAND_COUNT))
    probe_inner_steps = int(probe.get("probe_inner_steps", 1))
    max_train_batches = int(probe.get("max_train_batches", 1))
    max_val_batches = int(probe.get("max_val_batches", 1))
    if layer_group_count != DEFAULT_LAYER_GROUP_COUNT:
        raise ProbeValidationError("The RD-LoRA probe is locked to exactly 6 layer groups")
    if timestep_band_count != DEFAULT_TIMESTEP_BAND_COUNT:
        raise ProbeValidationError("The RD-LoRA probe is locked to exactly 4 timestep bands")
    if num_inference_steps <= 0:
        raise ProbeValidationError("schema.num_inference_steps must be positive")
    if probe_inner_steps <= 0:
        raise ProbeValidationError("probe.probe_inner_steps must be positive")
    if max_train_batches <= 0:
        raise ProbeValidationError("probe.max_train_batches must be positive")
    if max_val_batches <= 0:
        raise ProbeValidationError("probe.max_val_batches must be positive")

    resolved_tasks: dict[str, dict[str, Any]] = {}
    for task_name, task_config in tasks.items():
        task_payload = dict(task_config)
        train_data_dir = (repo_root / str(task_payload.get("train_data_dir", ""))).resolve()
        if not str(task_payload.get("validation_prompt", "")).strip():
            raise ProbeValidationError(f"tasks.{task_name}.validation_prompt must be non-empty")
        resolved_tasks[str(task_name)] = {
            "train_data_dir": str(train_data_dir),
            "validation_prompt": str(task_payload["validation_prompt"]),
        }

    return {
        "paths": {
            "repo_root": str(repo_root.resolve()),
            "output_root": str((repo_root / str(paths.get("output_root", "outputs/rd_lora/probe"))).resolve()),
            "accelerate_config": str((repo_root / str(paths.get("accelerate_config"))).resolve()),
            "expected_diffusers_substring": str(paths.get("expected_diffusers_substring", "huggingface_diffusers")),
        },
        "model": {
            "pretrained_model_name_or_path": str(model["pretrained_model_name_or_path"]),
            "pretrained_vae_model_name_or_path": model.get("pretrained_vae_model_name_or_path"),
            "revision": model.get("revision"),
            "variant": model.get("variant"),
            "mixed_precision": str(model.get("mixed_precision", "fp16")),
        },
        "schema": {
            "layer_group_count": layer_group_count,
            "timestep_band_count": timestep_band_count,
            "num_inference_steps": num_inference_steps,
            "candidate_ranks": list(candidate_ranks),
        },
        "probe": {
            "candidate_ranks": list(candidate_ranks),
            "probe_inner_steps": probe_inner_steps,
            "max_train_batches": max_train_batches,
            "max_val_batches": max_val_batches,
            "learning_rate": float(probe.get("learning_rate", 1.0e-4)),
            "target_modules": list(target_modules),
            "use_rslora": bool(probe.get("use_rslora", True)),
        },
        "tasks": resolved_tasks,
    }


def build_probe_schema(config: Mapping[str, Any], *, timestep_values: Sequence[int] | None = None) -> CellSchema:
    schema = dict(config["schema"])
    return build_cell_schema(
        layer_group_count=int(schema["layer_group_count"]),
        timestep_band_count=int(schema["timestep_band_count"]),
        num_inference_steps=int(schema["num_inference_steps"]),
        timestep_values=timestep_values,
        candidate_ranks=schema["candidate_ranks"],
    )


def build_run_request(
    *,
    config: Mapping[str, Any],
    task: str,
    run_mode: str,
    output_dir: Path,
    max_cells: int | None = None,
    candidate_ranks: Sequence[int] | None = None,
    probe_inner_steps: int | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict[str, Any]:
    if task not in config["tasks"]:
        raise ProbeValidationError(f"Unknown task {task!r}")
    if run_mode not in {"real_gpu", "cpu_mock"}:
        raise ProbeValidationError("run_mode must be one of {'real_gpu', 'cpu_mock'}")

    resolved_candidate_ranks = parse_candidate_ranks_argument(candidate_ranks, default=config["probe"]["candidate_ranks"])
    request = deep_update(
        dict(config),
        {
            "task": str(task),
            "run_mode": str(run_mode),
            "paths": {
                "output_dir": str(output_dir.resolve()),
            },
            "probe": {
                "candidate_ranks": list(resolved_candidate_ranks),
            },
        },
    )
    if max_cells is not None:
        request["max_cells"] = int(max_cells)
    if probe_inner_steps is not None:
        request["probe"]["probe_inner_steps"] = int(probe_inner_steps)
    if max_train_batches is not None:
        request["probe"]["max_train_batches"] = int(max_train_batches)
    if max_val_batches is not None:
        request["probe"]["max_val_batches"] = int(max_val_batches)
    request["schema"]["candidate_ranks"] = list(resolved_candidate_ranks)
    return request


def ensure_real_gpu_available(torch_module: Any | None = None) -> None:
    if torch_module is None:
        import torch as torch_module  # type: ignore

    if not bool(torch_module.cuda.is_available()):
        raise ProbeValidationError("run_mode=real_gpu requires torch.cuda.is_available() == True")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(UTILITY_RECORD_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in UTILITY_RECORD_FIELDNAMES})


def _iter_image_paths(train_data_dir: Path) -> list[Path]:
    if not train_data_dir.is_dir():
        raise ProbeValidationError(f"Task image directory not found: {train_data_dir}")
    paths = [path for path in sorted(train_data_dir.iterdir()) if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    if not paths:
        raise ProbeValidationError(f"No supported images found in {train_data_dir}")
    return paths


def _fit_image(path: Path, *, resolution: int) -> Image.Image:
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    return ImageOps.fit(image, (resolution, resolution), method=RESAMPLING.LANCZOS, centering=(0.5, 0.5))


def _import_text_encoder_class(
    *,
    pretrained_model_name_or_path: str,
    revision: str | None,
    subfolder: str,
) -> Any:
    from transformers import PretrainedConfig  # type: ignore

    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        revision=revision,
        subfolder=subfolder,
    )
    architecture = str(config.architectures[0])
    if architecture == "CLIPTextModel":
        from transformers import CLIPTextModel  # type: ignore

        return CLIPTextModel
    if architecture == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection  # type: ignore

        return CLIPTextModelWithProjection
    raise ProbeValidationError(f"Unsupported SDXL text encoder architecture {architecture!r}")


def _resolve_weight_dtype(torch_module: Any, mixed_precision: str) -> Any:
    if mixed_precision == "fp16":
        return torch_module.float16
    if mixed_precision == "bf16":
        return torch_module.bfloat16
    return torch_module.float32


def load_real_sdxl_components(config: Mapping[str, Any]) -> tuple[dict[str, Any], CellSchema]:
    ensure_real_gpu_available()

    import torch  # type: ignore
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    model = dict(config["model"])
    device = torch.device("cuda")
    weight_dtype = _resolve_weight_dtype(torch, str(model.get("mixed_precision", "fp16")))

    text_encoder_cls = _import_text_encoder_class(
        pretrained_model_name_or_path=str(model["pretrained_model_name_or_path"]),
        revision=model.get("revision"),
        subfolder="text_encoder",
    )
    text_encoder_2_cls = _import_text_encoder_class(
        pretrained_model_name_or_path=str(model["pretrained_model_name_or_path"]),
        revision=model.get("revision"),
        subfolder="text_encoder_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model["pretrained_model_name_or_path"]),
        subfolder="tokenizer",
        revision=model.get("revision"),
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        str(model["pretrained_model_name_or_path"]),
        subfolder="tokenizer_2",
        revision=model.get("revision"),
        use_fast=False,
    )
    text_encoder = text_encoder_cls.from_pretrained(
        str(model["pretrained_model_name_or_path"]),
        subfolder="text_encoder",
        revision=model.get("revision"),
        variant=model.get("variant"),
    )
    text_encoder_2 = text_encoder_2_cls.from_pretrained(
        str(model["pretrained_model_name_or_path"]),
        subfolder="text_encoder_2",
        revision=model.get("revision"),
        variant=model.get("variant"),
    )
    scheduler = DDPMScheduler.from_pretrained(
        str(model["pretrained_model_name_or_path"]),
        subfolder="scheduler",
    )
    vae_path = model.get("pretrained_vae_model_name_or_path") or model["pretrained_model_name_or_path"]
    vae = AutoencoderKL.from_pretrained(
        str(vae_path),
        subfolder=None if model.get("pretrained_vae_model_name_or_path") else "vae",
        revision=model.get("revision"),
        variant=model.get("variant"),
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(model["pretrained_model_name_or_path"]),
        subfolder="unet",
        revision=model.get("revision"),
        variant=model.get("variant"),
    )

    tokenizer.model_max_length = int(tokenizer.model_max_length)
    tokenizer_2.model_max_length = int(tokenizer_2.model_max_length)
    text_encoder.requires_grad_(False).eval().to(device=device, dtype=weight_dtype)
    text_encoder_2.requires_grad_(False).eval().to(device=device, dtype=weight_dtype)
    vae.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    unet.requires_grad_(False).eval().to(device=device, dtype=weight_dtype)

    scheduler.set_timesteps(int(config["schema"]["num_inference_steps"]), device=device)
    schema = build_probe_schema(config, timestep_values=[int(value) for value in scheduler.timesteps.tolist()])
    return (
        {
            "device": device,
            "weight_dtype": weight_dtype,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "vae": vae,
            "unet": unet,
            "scheduler": scheduler,
            "loaded_components": [
                "tokenizer",
                "tokenizer_2",
                "text_encoder",
                "text_encoder_2",
                "vae",
                "unet",
                "scheduler",
            ],
        },
        schema,
    )


def _encode_prompt_embeddings(components: Mapping[str, Any], prompt: str) -> tuple[Any, Any, Any]:
    import torch  # type: ignore

    tokenizer = components["tokenizer"]
    tokenizer_2 = components["tokenizer_2"]
    text_encoder = components["text_encoder"]
    text_encoder_2 = components["text_encoder_2"]
    device = components["device"]

    ids_1 = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    ids_2 = tokenizer_2(
        [prompt],
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        encoded_1 = text_encoder(ids_1, output_hidden_states=True, return_dict=False)
        encoded_2 = text_encoder_2(ids_2, output_hidden_states=True, return_dict=False)
        prompt_embeds = torch.concat([encoded_1[-1][-2], encoded_2[-1][-2]], dim=-1)
        pooled_prompt_embeds = encoded_2[0].view(prompt_embeds.shape[0], -1)
        add_time_ids = torch.tensor(
            [[DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 0, 0, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE]],
            device=device,
            dtype=components["weight_dtype"],
        )
    return prompt_embeds.detach().cpu(), pooled_prompt_embeds.detach().cpu(), add_time_ids.detach().cpu()


def load_task_examples(config: Mapping[str, Any], task: str, components: Mapping[str, Any]) -> list[EncodedExample]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    task_config = dict(config["tasks"][task])
    image_paths = _iter_image_paths(Path(task_config["train_data_dir"]))
    if len(image_paths) < 2:
        raise ProbeValidationError("The real GPU probe requires at least two images so validation is held out")

    prompt_embeds, pooled_prompt_embeds, add_time_ids = _encode_prompt_embeddings(
        components,
        str(task_config["validation_prompt"]),
    )
    vae = components["vae"]
    device = components["device"]

    examples: list[EncodedExample] = []
    for image_path in image_paths:
        image = _fit_image(image_path, resolution=DEFAULT_IMAGE_SIZE)
        image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).contiguous().float()
        image_tensor = ((image_tensor / 127.5) - 1.0).unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            latent_dist = vae.encode(image_tensor).latent_dist
            latents = latent_dist.sample() * float(vae.config.scaling_factor)
        examples.append(
            EncodedExample(
                image_path=str(image_path),
                latents=latents.detach().cpu(),
                prompt_embeds=prompt_embeds.clone(),
                pooled_prompt_embeds=pooled_prompt_embeds.clone(),
                add_time_ids=add_time_ids.clone(),
            )
        )
    return examples


def _build_seed(*, base_seed: int, cell_index: int, candidate_rank: int, split_offset: int, step_index: int) -> int:
    return (
        int(base_seed)
        + (cell_index * 10_000)
        + (candidate_rank * 100)
        + (split_offset * 10)
        + int(step_index)
    )


def _repeat_examples(examples: Sequence[EncodedExample], *, count: int) -> list[EncodedExample]:
    return [examples[index % len(examples)] for index in range(count)]


def _sample_band_inputs(
    *,
    scheduler: Any,
    batch: Sequence[EncodedExample],
    band_timestep_values: Sequence[int],
    seed: int,
    device: Any,
    latent_dtype: Any,
) -> dict[str, Any]:
    import torch  # type: ignore

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    latents = torch.cat([example.latents for example in batch], dim=0).to(device=device, dtype=latent_dtype)
    prompt_embeds = torch.cat([example.prompt_embeds for example in batch], dim=0).to(device=device, dtype=latent_dtype)
    pooled_prompt_embeds = torch.cat([example.pooled_prompt_embeds for example in batch], dim=0).to(
        device=device,
        dtype=latent_dtype,
    )
    add_time_ids = torch.cat([example.add_time_ids for example in batch], dim=0).to(device=device, dtype=latent_dtype)
    timestep_indices = torch.randint(
        low=0,
        high=len(band_timestep_values),
        size=(latents.shape[0],),
        generator=generator,
    )
    timesteps = torch.tensor(
        [int(band_timestep_values[index]) for index in timestep_indices.tolist()],
        device=device,
        dtype=torch.long,
    )
    noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32).to(device=device, dtype=latent_dtype)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    return {
        "latents": latents,
        "timesteps": timesteps,
        "noise": noise,
        "noisy_latents": noisy_latents,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "add_time_ids": add_time_ids,
    }


def _compute_loss(
    *,
    components: Mapping[str, Any],
    batch: Sequence[EncodedExample],
    band_timestep_values: Sequence[int],
    seed: int,
) -> Any:
    import torch.nn.functional as F  # type: ignore

    scheduler = components["scheduler"]
    unet = components["unet"]
    sampled = _sample_band_inputs(
        scheduler=scheduler,
        batch=batch,
        band_timestep_values=band_timestep_values,
        seed=seed,
        device=components["device"],
        latent_dtype=components["weight_dtype"],
    )

    model_pred = unet(
        sampled["noisy_latents"],
        sampled["timesteps"],
        sampled["prompt_embeds"],
        added_cond_kwargs={
            "time_ids": sampled["add_time_ids"],
            "text_embeds": sampled["pooled_prompt_embeds"],
        },
        return_dict=False,
    )[0]

    prediction_type = str(scheduler.config.prediction_type)
    if prediction_type == "epsilon":
        target = sampled["noise"]
    elif prediction_type == "v_prediction":
        target = scheduler.get_velocity(sampled["latents"], sampled["noise"], sampled["timesteps"])
    else:
        raise ProbeValidationError(f"Unsupported scheduler prediction_type {prediction_type!r}")
    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def _measure_loss(
    *,
    components: Mapping[str, Any],
    batches: Sequence[Sequence[EncodedExample]],
    cell: ProbeCell,
    base_seed: int,
    candidate_rank: int,
    split_offset: int,
) -> float:
    import torch  # type: ignore

    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            loss = _compute_loss(
                components=components,
                batch=batch,
                band_timestep_values=cell.timestep_band.timestep_values,
                seed=_build_seed(
                    base_seed=base_seed,
                    cell_index=cell.cell_index,
                    candidate_rank=candidate_rank,
                    split_offset=split_offset,
                    step_index=batch_index,
                ),
            )
            losses.append(float(loss.detach().item()))
    if not losses:
        raise ProbeValidationError("At least one batch is required to measure probe loss")
    return sum(losses) / float(len(losses))


def _cell_target_module_names(cell: ProbeCell, unet: Any, target_modules: Sequence[str]) -> list[str]:
    matched: list[str] = []
    prefixes = tuple(str(layer_id) for layer_id in cell.layer_group.layer_ids)
    suffixes = tuple(str(target_module) for target_module in target_modules)
    for module_name, _module in unet.named_modules():
        if any(module_name.startswith(prefix) for prefix in prefixes) and any(
            module_name.endswith(f".{suffix}") or module_name == suffix for suffix in suffixes
        ):
            matched.append(module_name)
    if not matched:
        raise ProbeValidationError(f"No UNet modules matched cell {cell.cell_id} for target_modules={tuple(suffixes)!r}")
    return sorted(set(matched))


def _run_real_cell_rank(
    *,
    components: Mapping[str, Any],
    cell: ProbeCell,
    candidate_rank: int,
    train_batches: Sequence[Sequence[EncodedExample]],
    val_batches: Sequence[Sequence[EncodedExample]],
    probe_inner_steps: int,
    learning_rate: float,
    target_modules: Sequence[str],
    use_rslora: bool,
    base_seed: int,
    task: str,
) -> dict[str, Any]:
    import torch  # type: ignore
    from peft import LoraConfig  # type: ignore

    unet = components["unet"]
    pre_loss: float
    post_loss: float
    optimizer_steps = 0

    if candidate_rank == 0:
        unet.eval()
        pre_loss = _measure_loss(
            components=components,
            batches=val_batches,
            cell=cell,
            base_seed=base_seed,
            candidate_rank=candidate_rank,
            split_offset=2,
        )
        post_loss = _measure_loss(
            components=components,
            batches=val_batches,
            cell=cell,
            base_seed=base_seed,
            candidate_rank=candidate_rank,
            split_offset=2,
        )
    else:
        adapter_name = f"probe_{cell.cell_index:02d}_r{candidate_rank}"
        lora_config = LoraConfig(
            r=int(candidate_rank),
            lora_alpha=int(candidate_rank),
            init_lora_weights="gaussian",
            use_rslora=bool(use_rslora),
            lora_dropout=0.0,
            target_modules=_cell_target_module_names(cell, unet, target_modules),
        )
        unet.add_adapter(lora_config, adapter_name=adapter_name)
        unet.set_adapters(adapter_name)
        trainable_params = [parameter for parameter in unet.parameters() if parameter.requires_grad]
        if not trainable_params:
            raise ProbeValidationError(f"No trainable adapter parameters were created for {cell.cell_id} rank {candidate_rank}")
        for p in trainable_params:
            p.data = p.data.float()
        optimizer = torch.optim.AdamW(trainable_params, lr=float(learning_rate))
        try:
            unet.eval()
            pre_loss = _measure_loss(
                components=components,
                batches=val_batches,
                cell=cell,
                base_seed=base_seed,
                candidate_rank=candidate_rank,
                split_offset=2,
            )

            unet.train()
            for step_index in range(int(probe_inner_steps)):
                optimizer.zero_grad(set_to_none=True)
                loss = _compute_loss(
                    components=components,
                    batch=train_batches[step_index % len(train_batches)],
                    band_timestep_values=cell.timestep_band.timestep_values,
                    seed=_build_seed(
                        base_seed=base_seed,
                        cell_index=cell.cell_index,
                        candidate_rank=candidate_rank,
                        split_offset=1,
                        step_index=step_index,
                    ),
                )
                loss.backward()
                optimizer.step()
                optimizer_steps += 1

            unet.eval()
            post_loss = _measure_loss(
                components=components,
                batches=val_batches,
                cell=cell,
                base_seed=base_seed,
                candidate_rank=candidate_rank,
                split_offset=2,
            )
        finally:
            optimizer.zero_grad(set_to_none=True)
            unet.delete_adapters(adapter_name)

    return build_probe_row(
        task=task,
        cell_id=cell.cell_id,
        layer_group=cell.layer_group.group_id,
        timestep_band=cell.timestep_band.band_id,
        candidate_rank=candidate_rank,
        pre_loss=pre_loss,
        post_loss=post_loss,
        optimizer_steps=optimizer_steps,
        train_batch_count=len(train_batches) if candidate_rank > 0 else 0,
        val_batch_count=len(val_batches),
    )


def run_cpu_mock_probe(config: Mapping[str, Any], schema: CellSchema, *, task: str) -> dict[str, Any]:
    candidate_ranks = tuple(int(rank) for rank in config["probe"]["candidate_ranks"])
    max_cells = min(int(config.get("max_cells", len(schema.cells))), len(schema.cells))
    rows: list[dict[str, Any]] = []
    for cell in schema.cells[:max_cells]:
        base_loss = 1.0 + (0.05 * cell.layer_group.group_index) + (0.01 * cell.timestep_band.band_index)
        for rank in candidate_ranks:
            improvement = 0.0 if rank == 0 else (0.0025 * rank) + (0.0005 * cell.cell_index)
            rows.append(
                build_probe_row(
                    task=task,
                    cell_id=cell.cell_id,
                    layer_group=cell.layer_group.group_id,
                    timestep_band=cell.timestep_band.band_id,
                    candidate_rank=rank,
                    pre_loss=base_loss,
                    post_loss=base_loss - improvement,
                    optimizer_steps=0,
                    train_batch_count=min(int(config["probe"]["max_train_batches"]), 1),
                    val_batch_count=min(int(config["probe"]["max_val_batches"]), 1),
                )
            )
    summary = build_probe_summary(
        probe_mode="cpu_mock",
        task=task,
        cell_count=max_cells,
        candidate_ranks=candidate_ranks,
        row_count=len(rows),
        loaded_components=[],
        used_gpu=False,
        peak_vram_mib=0.0,
    )
    provenance = collect_run_provenance(run_mode="cpu_mock", used_gpu=False, peak_vram_mib=0.0)
    validate_probe_payloads(rows=rows, summary=summary, provenance=provenance)
    return {"rows": rows, "summary": summary, "provenance": provenance}


def run_real_gpu_probe(config: Mapping[str, Any], *, task: str) -> dict[str, Any]:
    import torch  # type: ignore

    ensure_real_gpu_available(torch)
    torch.cuda.reset_peak_memory_stats()

    components, schema = load_real_sdxl_components(config)
    examples = load_task_examples(config, task, components)
    train_pool = examples[:-1]
    val_pool = examples[-1:]
    if not train_pool or not val_pool:
        raise ProbeValidationError("The real GPU probe requires disjoint train and validation microbatches")

    train_batches = [[example] for example in _repeat_examples(train_pool, count=int(config["probe"]["max_train_batches"]))]
    val_batches = [[example] for example in _repeat_examples(val_pool, count=int(config["probe"]["max_val_batches"]))]
    max_cells = min(int(config.get("max_cells", len(schema.cells))), len(schema.cells))
    rows: list[dict[str, Any]] = []
    base_seed = 20260410

    for cell in schema.cells[:max_cells]:
        for candidate_rank in config["probe"]["candidate_ranks"]:
            rows.append(
                _run_real_cell_rank(
                    components=components,
                    cell=cell,
                    candidate_rank=int(candidate_rank),
                    train_batches=train_batches,
                    val_batches=val_batches,
                    probe_inner_steps=int(config["probe"]["probe_inner_steps"]),
                    learning_rate=float(config["probe"]["learning_rate"]),
                    target_modules=config["probe"]["target_modules"],
                    use_rslora=bool(config["probe"]["use_rslora"]),
                    base_seed=base_seed,
                    task=task,
                )
            )

    peak_vram_mib = float(torch.cuda.max_memory_allocated()) / float(1024**2)
    if peak_vram_mib <= 0.0:
        raise ProbeValidationError("real_gpu probe must record peak_vram_mib > 0")
    summary = build_probe_summary(
        probe_mode="real_gpu_forward_backward",
        task=task,
        cell_count=max_cells,
        candidate_ranks=config["probe"]["candidate_ranks"],
        row_count=len(rows),
        loaded_components=components["loaded_components"],
        used_gpu=True,
        peak_vram_mib=peak_vram_mib,
    )
    provenance = collect_run_provenance(run_mode="real_gpu", used_gpu=True, peak_vram_mib=peak_vram_mib)
    validate_probe_payloads(rows=rows, summary=summary, provenance=provenance)
    return {"rows": rows, "summary": summary, "provenance": provenance}


def dispatch_probe(config: Mapping[str, Any]) -> dict[str, Any]:
    run_mode = str(config["run_mode"])
    schema = build_probe_schema(config)
    if run_mode == "cpu_mock":
        return run_cpu_mock_probe(config, schema, task=str(config["task"]))
    if run_mode == "real_gpu":
        ensure_real_gpu_available()
        return run_real_gpu_probe(config, task=str(config["task"]))
    raise ProbeValidationError(f"Unsupported run_mode {run_mode!r}")


def write_probe_outputs(output_dir: Path, payload: Mapping[str, Any], *, task: str, candidate_ranks: Sequence[int]) -> dict[str, str]:
    rows = list(payload["rows"])
    summary = dict(payload["summary"])
    provenance = dict(payload["provenance"])

    cell_utility_json_path = output_dir / "cell_utility.json"
    cell_utility_csv_path = output_dir / "cell_utility.csv"
    probe_summary_path = output_dir / "probe_summary.json"
    provenance_path = output_dir / "run_provenance.json"

    save_json(
        cell_utility_json_path,
        {
            "task": str(task),
            "candidate_ranks": [int(rank) for rank in candidate_ranks],
            "row_count": len(rows),
            "rows": rows,
        },
    )
    _write_csv(cell_utility_csv_path, rows)
    save_json(probe_summary_path, summary)
    save_json(provenance_path, provenance)

    return {
        "cell_utility_json": str(cell_utility_json_path),
        "cell_utility_csv": str(cell_utility_csv_path),
        "probe_summary": str(probe_summary_path),
        "run_provenance": str(provenance_path),
    }


def validate_probe_outputs(output_dir: Path) -> dict[str, Any]:
    cell_utility = output_dir / "cell_utility.json"
    summary = output_dir / "probe_summary.json"
    provenance = output_dir / "run_provenance.json"
    csv_path = output_dir / "cell_utility.csv"
    for path in (cell_utility, summary, provenance, csv_path):
        if not path.is_file():
            raise ProbeValidationError(f"Missing probe artifact: {path}")
    return {"ok": True, "output_dir": str(output_dir.resolve())}


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "ProbeValidationError",
    "build_probe_schema",
    "build_run_request",
    "default_probe_config",
    "dispatch_probe",
    "ensure_real_gpu_available",
    "load_real_sdxl_components",
    "resolve_probe_config",
    "run_cpu_mock_probe",
    "run_real_gpu_probe",
    "validate_probe_outputs",
    "write_probe_outputs",
]
