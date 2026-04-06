from __future__ import annotations

import csv
import inspect
import io
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from PIL import Image

from vericodec_diff.prompt_manifests import MANIFEST_SPECS, REQUIRED_COLUMNS, validate_manifest


DEFAULT_MODEL_ID = "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_GUIDANCE_SCALE = 4.5
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_TRACE_LATE_STEP_COUNT = 4
TRACE_PAYLOAD_VERSION = 1


class SanaGenerationError(RuntimeError):
    """Raised when Sana generation cannot complete with the requested config."""


class SanaBackendUnavailableError(SanaGenerationError):
    """Raised when the local runtime cannot load the Sana inference backend."""


def resolve_path(repo_root: Path, raw_path: str) -> Path:
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


def default_sana_generation_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "prompt_manifest": "data/manifests/prompt_manifest_kill.csv",
            "output_root": "outputs/sana",
            "suite": None,
        },
        "model": {
            "model_id": DEFAULT_MODEL_ID,
            "device": "cuda",
            "torch_dtype": "bfloat16",
            "vae_dtype": "bfloat16",
            "text_encoder_dtype": "bfloat16",
            "local_files_only": False,
        },
        "generation": {
            "height": DEFAULT_IMAGE_SIZE,
            "width": DEFAULT_IMAGE_SIZE,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
            "limit": None,
            "overwrite": False,
        },
        "trace": {
            "save_trace": False,
            "late_step_count": DEFAULT_TRACE_LATE_STEP_COUNT,
        },
    }


def derive_default_suite_name(prompt_manifest_path: Path) -> str:
    stem = prompt_manifest_path.stem
    prefix = "prompt_manifest_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _validate_suite_name(raw_suite: str) -> str:
    suite = raw_suite.strip()
    if not suite:
        raise ValueError("suite must be a non-empty path token")
    suite_path = Path(suite)
    if len(suite_path.parts) != 1 or suite_path.name != suite:
        raise ValueError("suite must be a single path component")
    if suite in {".", ".."}:
        raise ValueError("suite must not be '.' or '..'")
    return suite


def resolve_sana_generation_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_sana_generation_config(), raw_config)
    paths = dict(config.get("paths", {}))
    model = dict(config.get("model", {}))
    generation = dict(config.get("generation", {}))
    trace = dict(config.get("trace", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    prompt_manifest_path = resolve_path(
        resolved_repo_root,
        str(paths.get("prompt_manifest", "data/manifests/prompt_manifest_kill.csv")),
    ).resolve()
    output_root = resolve_path(
        resolved_repo_root,
        str(paths.get("output_root", "outputs/sana")),
    ).resolve()
    suite = _validate_suite_name(
        str(paths.get("suite") or derive_default_suite_name(prompt_manifest_path))
    )
    output_dir = (output_root / suite).resolve()

    if not prompt_manifest_path.is_file():
        raise FileNotFoundError(f"Prompt manifest not found: {prompt_manifest_path}")

    for name, path in {"output_root": output_root, "output_dir": output_dir}.items():
        try:
            path.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ValueError(f"{name} must resolve inside repo_root for stable paths") from exc

    height = int(generation.get("height", DEFAULT_IMAGE_SIZE))
    width = int(generation.get("width", DEFAULT_IMAGE_SIZE))
    if (height, width) != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
        raise ValueError(
            f"VeriCodec-Diff Sana inference is locked to {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}"
        )

    num_inference_steps = int(generation.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer")

    guidance_scale = float(generation.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    limit_value = generation.get("limit")
    limit = None if limit_value in (None, "") else int(limit_value)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided")

    late_step_count = int(trace.get("late_step_count", DEFAULT_TRACE_LATE_STEP_COUNT))
    if late_step_count <= 0:
        raise ValueError("trace.late_step_count must be a positive integer")

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "prompt_manifest": str(prompt_manifest_path),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "suite": suite,
        },
        "model": {
            "model_id": str(model.get("model_id", DEFAULT_MODEL_ID)),
            "device": str(model.get("device", "cuda")),
            "torch_dtype": str(model.get("torch_dtype", "bfloat16")),
            "vae_dtype": str(model.get("vae_dtype", model.get("torch_dtype", "bfloat16"))),
            "text_encoder_dtype": str(
                model.get("text_encoder_dtype", model.get("torch_dtype", "bfloat16"))
            ),
            "local_files_only": bool(model.get("local_files_only", False)),
        },
        "generation": {
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "limit": limit,
            "overwrite": bool(generation.get("overwrite", False)),
        },
        "trace": {
            "save_trace": bool(trace.get("save_trace", False)),
            "late_step_count": late_step_count,
        },
    }


def load_prompt_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if manifest_path.name in MANIFEST_SPECS:
        validate_manifest(manifest_path)

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != REQUIRED_COLUMNS:
            raise ValueError(f"{manifest_path.name}: expected columns {REQUIRED_COLUMNS}, found {fieldnames}")

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            prompt_id = (row.get("prompt_id") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            seed_text = (row.get("seed") or "").strip()
            if not prompt_id:
                raise ValueError(f"{manifest_path.name}: row {row_number} has empty prompt_id")
            if not prompt:
                raise ValueError(f"{manifest_path.name}: row {row_number} has empty prompt")
            try:
                seed = int(seed_text)
            except ValueError as exc:
                raise ValueError(
                    f"{manifest_path.name}: row {row_number} has non-integer seed {seed_text!r}"
                ) from exc
            if seed <= 0:
                raise ValueError(f"{manifest_path.name}: row {row_number} must use a positive seed")

            rows.append(
                {
                    field: (row.get(field) or "").strip()
                    for field in REQUIRED_COLUMNS
                }
            )
    return rows


def build_sample_id(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}__seed_{int(seed)}"


def build_sana_generation_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    repo_root = Path(config["paths"]["repo_root"])
    output_dir = Path(config["paths"]["output_dir"])
    rows = load_prompt_manifest_rows(config["paths"]["prompt_manifest"])
    limit = config["generation"]["limit"]
    if limit is not None:
        rows = rows[: int(limit)]

    plan: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        sample_id = build_sample_id(row["prompt_id"], seed)
        image_path = output_dir / f"{sample_id}.png"
        trace_path = output_dir / f"{sample_id}__trace.pt"
        plan.append(
            {
                "sample_id": sample_id,
                "prompt_id": row["prompt_id"],
                "category": row["category"],
                "stress_type": row["stress_type"],
                "prompt": row["prompt"],
                "negative_prompt": row["negative_prompt"],
                "split": row["split"],
                "seed": seed,
                "image_path": image_path,
                "trace_path": trace_path,
                "image_local_path": display_path(image_path, repo_root),
                "trace_local_path": display_path(trace_path, repo_root),
            }
        )
    return plan


class SanaDiffusersBackbone:
    """Thin diffusers-backed wrapper for frozen Sana inference."""

    backend_name = "diffusers:SanaPipeline"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        pipeline_cls: type[Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config
        self.model_config = dict(config["model"])
        self.generation_config = dict(config["generation"])
        self.trace_config = dict(config["trace"])
        self.torch = self._load_torch(torch_module)
        self.pipeline_cls = self._load_pipeline_cls(pipeline_cls)
        self.pipeline = self._load_pipeline()
        self._call_parameters = set(inspect.signature(self.pipeline.__call__).parameters)

    def _load_torch(self, torch_module: Any | None) -> Any:
        if torch_module is not None:
            return torch_module
        try:
            import torch as imported_torch
        except Exception as exc:
            raise SanaBackendUnavailableError(
                "torch is required for Sana inference but could not be imported"
            ) from exc
        return imported_torch

    def _load_pipeline_cls(self, pipeline_cls: type[Any] | None) -> type[Any]:
        if pipeline_cls is not None:
            return pipeline_cls
        try:
            from diffusers import SanaPipeline as imported_pipeline
        except Exception as exc:
            raise SanaBackendUnavailableError(
                "diffusers.SanaPipeline is required for Sana inference but could not be imported"
            ) from exc
        return imported_pipeline

    def _resolve_dtype(self, raw_dtype: str | None) -> Any | None:
        if raw_dtype in (None, "", "auto", "none", "None"):
            return None
        try:
            return getattr(self.torch, str(raw_dtype))
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype {raw_dtype!r}") from exc

    def _load_pipeline(self) -> Any:
        load_kwargs: dict[str, Any] = {
            "local_files_only": bool(self.model_config["local_files_only"]),
        }
        torch_dtype = self._resolve_dtype(self.model_config.get("torch_dtype"))
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype

        pipeline = self.pipeline_cls.from_pretrained(self.model_config["model_id"], **load_kwargs)
        pipeline.to(self.model_config["device"])

        for component_name, dtype_key in (
            ("vae", "vae_dtype"),
            ("text_encoder", "text_encoder_dtype"),
        ):
            component = getattr(pipeline, component_name, None)
            component_dtype = self._resolve_dtype(self.model_config.get(dtype_key))
            if component is None or component_dtype is None:
                continue
            try:
                component.to(dtype=component_dtype)
            except TypeError:
                component.to(component_dtype)
        return pipeline

    def _make_generator(self, seed: int) -> Any | None:
        generator_cls = getattr(self.torch, "Generator", None)
        if generator_cls is None:
            return None
        try:
            generator = generator_cls(device=self.model_config["device"])
        except Exception:
            generator = generator_cls()
        manual_seed = getattr(generator, "manual_seed", None)
        if callable(manual_seed):
            return manual_seed(int(seed))
        return generator

    def _clone_latent(self, latent: Any) -> Any:
        if hasattr(latent, "detach"):
            latent = latent.detach()
        if hasattr(latent, "cpu"):
            latent = latent.cpu()
        if hasattr(latent, "clone"):
            latent = latent.clone()
        return latent

    def _extract_first_image(self, output: Any) -> Image.Image:
        payload = getattr(output, "images", output)
        if isinstance(payload, tuple):
            payload = payload[0]
        if isinstance(payload, list):
            payload = payload[0]
        if not isinstance(payload, Image.Image):
            raise SanaGenerationError(
                f"Expected a PIL image from Sana output, received {type(payload).__name__}"
            )
        return payload

    def _extract_final_latent(self, output: Any) -> Any:
        for attribute_name in ("latents", "latent", "images", "image", "sample"):
            if hasattr(output, attribute_name):
                payload = getattr(output, attribute_name)
                break
        else:
            payload = output[0] if isinstance(output, tuple) else output
        if isinstance(payload, list):
            payload = payload[0]
        return payload

    def _module_device(self, module: Any) -> Any | None:
        if hasattr(module, "device"):
            return module.device
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return None
        try:
            return next(parameters()).device
        except (StopIteration, TypeError):
            return None

    def _decode_latents_to_pil(self, latents: list[Any]) -> list[Image.Image]:
        if hasattr(self.pipeline, "decode_trace_latents_to_pil"):
            return list(self.pipeline.decode_trace_latents_to_pil(latents))

        if not latents:
            return []

        vae = getattr(self.pipeline, "vae", None)
        image_processor = getattr(self.pipeline, "image_processor", None)
        if vae is None or not hasattr(vae, "decode") or image_processor is None:
            raise SanaGenerationError(
                "This Sana backend cannot decode latent traces into images."
            )

        batch = latents[0]
        if len(latents) > 1:
            if not hasattr(self.torch, "cat"):
                raise SanaGenerationError("torch.cat is required to decode multiple trace latents")
            batch = self.torch.cat(latents, dim=0)

        scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
        decode_input = batch / scaling_factor
        device = self._module_device(vae)
        dtype = getattr(vae, "dtype", None)
        if hasattr(decode_input, "to") and (device is not None or dtype is not None):
            to_kwargs: dict[str, Any] = {}
            if device is not None:
                to_kwargs["device"] = device
            if dtype is not None:
                to_kwargs["dtype"] = dtype
            decode_input = decode_input.to(**to_kwargs)

        no_grad = getattr(self.torch, "no_grad", None)
        with no_grad() if callable(no_grad) else nullcontext():
            try:
                decoded = vae.decode(decode_input, return_dict=False)[0]
            except TypeError:
                decoded = vae.decode(decode_input)
                decoded = getattr(decoded, "sample", decoded)

        images = image_processor.postprocess(decoded, output_type="pil")
        return list(images)

    def _pil_to_png_bytes(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def generate_sample(
        self,
        *,
        sample_id: str,
        prompt_id: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        trace_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        num_inference_steps = int(self.generation_config["num_inference_steps"])
        late_step_count = int(trace_config["late_step_count"])
        save_trace = bool(trace_config["save_trace"])

        call_kwargs: dict[str, Any] = {}
        for key, value in (
            ("prompt", prompt),
            ("negative_prompt", negative_prompt or None),
            ("height", int(self.generation_config["height"])),
            ("width", int(self.generation_config["width"])),
            ("guidance_scale", float(self.generation_config["guidance_scale"])),
            ("num_inference_steps", num_inference_steps),
            ("generator", self._make_generator(seed)),
        ):
            if key in self._call_parameters and value is not None:
                call_kwargs[key] = value
        if "return_dict" in self._call_parameters:
            call_kwargs["return_dict"] = True

        trace_state = {
            "step_indices": [],
            "timesteps": [],
            "latents": [],
        }
        limitations: list[str] = []
        trace_status = "disabled"

        if save_trace and "callback_on_step_end" in self._call_parameters:

            # Capture only the last few denoising states to keep trace files bounded.
            def callback_on_step_end(_: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]) -> dict[str, Any]:
                latents = callback_kwargs.get("latents")
                if latents is None:
                    return callback_kwargs
                if step_index >= max(0, num_inference_steps - late_step_count):
                    trace_state["step_indices"].append(int(step_index))
                    trace_state["timesteps"].append(int(timestep) if timestep is not None else None)
                    trace_state["latents"].append(self._clone_latent(latents))
                return callback_kwargs

            call_kwargs["callback_on_step_end"] = callback_on_step_end
            if "callback_on_step_end_tensor_inputs" in self._call_parameters:
                call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
        elif save_trace:
            limitations.append(
                "Backend does not expose callback_on_step_end; late-step traces fall back to final latent only."
            )

        final_latent = None
        if save_trace and "output_type" in self._call_parameters:
            call_kwargs["output_type"] = "latent"
        elif save_trace:
            limitations.append(
                "Backend does not expose output_type='latent'; exact decoded trace saving is unavailable."
            )

        output = self.pipeline(**call_kwargs)
        if save_trace and "output_type" in call_kwargs:
            final_latent = self._clone_latent(self._extract_final_latent(output))
            final_image = self._decode_latents_to_pil([final_latent])[0]
        else:
            final_image = self._extract_first_image(output)

        trace_payload = None
        if save_trace:
            trace_latents = list(trace_state["latents"])
            trace_step_indices = list(trace_state["step_indices"])
            trace_timesteps = list(trace_state["timesteps"])

            if final_latent is not None and (not trace_step_indices or trace_step_indices[-1] != num_inference_steps - 1):
                trace_latents.append(final_latent)
                trace_step_indices.append(num_inference_steps - 1)
                trace_timesteps.append(None)

            if trace_latents:
                trace_images = self._decode_latents_to_pil(trace_latents)
                trace_status = "late_steps" if trace_state["latents"] else "final_only"
                trace_payload = {
                    "trace_version": TRACE_PAYLOAD_VERSION,
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "seed": int(seed),
                    "trace_status": trace_status,
                    "backend_name": self.backend_name,
                    "format": "png_bytes",
                    "step_indices": trace_step_indices,
                    "timesteps": trace_timesteps,
                    "png_bytes": [self._pil_to_png_bytes(image) for image in trace_images],
                }
            else:
                trace_status = "unsupported"

        return {
            "image": final_image,
            "trace_payload": trace_payload,
            "trace_status": trace_status,
            "backend_limitations": limitations,
        }

    def close(self) -> None:
        return None


def _load_trace_saver(torch_module: Any | None) -> Any:
    if torch_module is not None:
        return torch_module
    try:
        import torch as imported_torch
    except Exception as exc:
        raise SanaBackendUnavailableError(
            "Trace saving requires torch.save, but torch could not be imported"
        ) from exc
    return imported_torch


def save_trace_payload(path: Path, payload: Mapping[str, Any], *, torch_module: Any | None = None) -> None:
    saver = _load_trace_saver(torch_module)
    if not hasattr(saver, "save"):
        raise ValueError("torch_module must provide a save(path, payload) compatible API")
    saver.save(dict(payload), path)


def write_generation_records_csv(path: str | Path, records: list[dict[str, Any]]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "sample_id",
        "prompt_id",
        "seed",
        "category",
        "stress_type",
        "split",
        "image_path",
        "trace_path",
        "trace_status",
        "generation_seconds",
        "status",
    )
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def generate_sana_outputs(
    config: Mapping[str, Any],
    *,
    backbone: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = build_sana_generation_plan(config)
    save_trace = bool(config["trace"]["save_trace"])
    overwrite = bool(config["generation"]["overwrite"])

    owned_backbone = backbone is None
    if backbone is None:
        backbone = SanaDiffusersBackbone(config)

    records: list[dict[str, Any]] = []
    category_counts = Counter(item["category"] for item in plan)
    split_counts = Counter(item["split"] for item in plan)
    trace_status_counts: Counter[str] = Counter()
    backend_limitations: set[str] = set()
    generated_count = 0
    saved_trace_count = 0

    try:
        for item in plan:
            image_path = Path(item["image_path"])
            trace_path = Path(item["trace_path"])
            should_skip = image_path.is_file() and (not save_trace or trace_path.is_file()) and not overwrite

            if should_skip:
                record = {
                    "sample_id": item["sample_id"],
                    "prompt_id": item["prompt_id"],
                    "seed": item["seed"],
                    "category": item["category"],
                    "stress_type": item["stress_type"],
                    "split": item["split"],
                    "image_path": item["image_local_path"],
                    "trace_path": item["trace_local_path"] if save_trace and trace_path.is_file() else "",
                    "trace_status": "existing" if save_trace and trace_path.is_file() else "disabled",
                    "generation_seconds": "0.000",
                    "status": "skipped_existing",
                }
                trace_status_counts[record["trace_status"]] += 1
                records.append(record)
                continue

            started_at = perf_counter()
            result = backbone.generate_sample(
                sample_id=item["sample_id"],
                prompt_id=item["prompt_id"],
                prompt=item["prompt"],
                negative_prompt=item["negative_prompt"],
                seed=int(item["seed"]),
                trace_config=config["trace"],
            )
            duration_seconds = perf_counter() - started_at

            image = result["image"]
            if not isinstance(image, Image.Image):
                raise SanaGenerationError(
                    f"Backbone returned {type(image).__name__} instead of PIL.Image.Image"
                )
            image.save(image_path)
            generated_count += 1

            trace_payload = result.get("trace_payload")
            trace_status = str(result.get("trace_status", "disabled"))
            trace_local_path = ""
            if trace_payload is not None:
                save_trace_payload(trace_path, trace_payload, torch_module=torch_module)
                saved_trace_count += 1
                trace_local_path = item["trace_local_path"]

            trace_status_counts[trace_status] += 1
            backend_limitations.update(str(message) for message in result.get("backend_limitations", []))
            records.append(
                {
                    "sample_id": item["sample_id"],
                    "prompt_id": item["prompt_id"],
                    "seed": item["seed"],
                    "category": item["category"],
                    "stress_type": item["stress_type"],
                    "split": item["split"],
                    "image_path": item["image_local_path"],
                    "trace_path": trace_local_path,
                    "trace_status": trace_status,
                    "generation_seconds": f"{duration_seconds:.3f}",
                    "status": "generated",
                }
            )
    finally:
        close = getattr(backbone, "close", None)
        if owned_backbone and callable(close):
            close()

    return {
        "summary": {
            "suite": config["paths"]["suite"],
            "manifest_path": display_path(Path(config["paths"]["prompt_manifest"]), repo_root),
            "output_dir": display_path(output_dir, repo_root),
            "model_id": config["model"]["model_id"],
            "planned_sample_count": len(plan),
            "generated_sample_count": generated_count,
            "category_counts": dict(category_counts),
            "split_counts": dict(split_counts),
            "trace": {
                "requested": save_trace,
                "late_step_count": int(config["trace"]["late_step_count"]),
                "saved_count": saved_trace_count,
                "status_counts": dict(trace_status_counts),
                "backend_limitations": sorted(backend_limitations),
            },
        },
        "records": records,
    }
