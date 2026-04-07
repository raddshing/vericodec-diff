from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps


IMAGE_SIZE = 1024
REAL_IMAGE_MANIFEST_COLUMNS = (
    "image_id",
    "category",
    "local_path",
    "source_name",
    "source_url",
    "license",
    "split",
)
SUPPORTED_IMAGE_SUFFIXES = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
_RESAMPLING = getattr(Image, "Resampling", Image)


class RealImagePreparationError(ValueError):
    """Raised when the real-image preparation inputs are invalid."""


class RealImageManifestValidationError(ValueError):
    """Raised when the real-image manifest violates the local schema."""


@dataclass(frozen=True)
class SourceImageItem:
    source_key: str
    source_url: str
    image_path: Path | None = None
    image_value: Any = None


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


def parse_csv_items(raw_value: str | Sequence[str] | None) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    return [str(item).strip() for item in raw_value if str(item).strip()]


def _slug_token(raw_value: str, *, fallback: str) -> str:
    characters = []
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


def default_real_image_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "prepared_root": "data/raw/real_prepared",
            "manifest_path": "data/manifests/real_images_manifest.csv",
            "output_dir": "outputs/real_image_prep",
        },
        "source": {
            "kind": None,
            "root": None,
            "openimages_classes": [],
            "hf_dataset": None,
            "hf_split": "train",
            "hf_config_name": None,
            "hf_data_dir": None,
            "hf_image_column": None,
            "hf_cache_dir": None,
        },
        "selection": {
            "category": None,
            "split": None,
            "limit": None,
            "source_name": None,
            "source_url": None,
            "license": "unknown",
        },
    }


def _default_source_name(kind: str, hf_dataset: str | None) -> str:
    if kind == "ffhq":
        return "ffhq"
    if kind == "openimages":
        return "openimages"
    if kind == "huggingface":
        return hf_dataset or "huggingface"
    raise ValueError(f"Unsupported source kind: {kind!r}")


def _default_source_url(kind: str, hf_dataset: str | None) -> str:
    if kind == "ffhq":
        return "https://github.com/NVlabs/ffhq-dataset"
    if kind == "openimages":
        return "https://storage.googleapis.com/openimages/web/index.html"
    if kind == "huggingface":
        if hf_dataset == "imagefolder":
            return "https://huggingface.co/docs/datasets/image_dataset"
        if hf_dataset:
            return f"https://huggingface.co/datasets/{hf_dataset}"
        return "https://huggingface.co/datasets"
    raise ValueError(f"Unsupported source kind: {kind!r}")


def resolve_real_image_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_real_image_config(), raw_config)
    paths = dict(config.get("paths", {}))
    source = dict(config.get("source", {}))
    selection = dict(config.get("selection", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    prepared_root = resolve_path(
        resolved_repo_root,
        str(paths.get("prepared_root", "data/raw/real_prepared")),
    ).resolve()
    manifest_path = resolve_path(
        resolved_repo_root,
        str(paths.get("manifest_path", "data/manifests/real_images_manifest.csv")),
    ).resolve()
    output_dir = resolve_path(
        resolved_repo_root,
        str(paths.get("output_dir", "outputs/real_image_prep")),
    ).resolve()

    for name, path in {
        "prepared_root": prepared_root,
        "manifest_path": manifest_path,
        "output_dir": output_dir,
    }.items():
        try:
            path.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ValueError(f"{name} must resolve inside repo_root for stable local paths") from exc

    kind = str(source.get("kind") or "").strip()
    if kind not in {"ffhq", "openimages", "huggingface"}:
        raise ValueError("source.kind must be one of {'ffhq', 'openimages', 'huggingface'}")

    source_root = None
    source_root_raw = source.get("root")
    if source_root_raw not in (None, ""):
        source_root = resolve_path(resolved_repo_root, str(source_root_raw)).resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"Source root not found: {source_root}")

    category = str(selection.get("category") or "").strip()
    if not category:
        raise ValueError("selection.category is required")

    hf_dataset = source.get("hf_dataset")
    hf_dataset_name = None if hf_dataset in (None, "") else str(hf_dataset)
    hf_split = str(source.get("hf_split", "train")).strip() or "train"
    hf_image_column = source.get("hf_image_column")
    hf_image_column_name = None if hf_image_column in (None, "") else str(hf_image_column)

    if kind in {"ffhq", "openimages"} and source_root is None:
        raise ValueError("source.root is required for ffhq and openimages ingestion")
    if kind == "huggingface" and not hf_dataset_name:
        raise ValueError("source.hf_dataset is required for Hugging Face ingestion")

    openimages_classes = parse_csv_items(source.get("openimages_classes"))
    if kind != "openimages" and openimages_classes:
        raise ValueError("source.openimages_classes is only valid for openimages ingestion")

    hf_data_dir = source.get("hf_data_dir")
    resolved_hf_data_dir = None
    if hf_data_dir not in (None, ""):
        resolved_hf_data_dir = resolve_path(resolved_repo_root, str(hf_data_dir)).resolve()
        if not resolved_hf_data_dir.is_dir():
            raise FileNotFoundError(f"Hugging Face data_dir not found: {resolved_hf_data_dir}")

    hf_cache_dir = source.get("hf_cache_dir")
    if hf_cache_dir in (None, ""):
        resolved_hf_cache_dir = (output_dir / "hf_cache").resolve()
    else:
        resolved_hf_cache_dir = resolve_path(resolved_repo_root, str(hf_cache_dir)).resolve()
    try:
        resolved_hf_cache_dir.relative_to(resolved_repo_root)
    except ValueError as exc:
        raise ValueError("source.hf_cache_dir must resolve inside repo_root for stable local paths") from exc

    limit_raw = selection.get("limit")
    limit = None if limit_raw in (None, "") else int(limit_raw)
    if limit is not None and limit <= 0:
        raise ValueError("selection.limit must be a positive integer when provided")

    selection_split = selection.get("split")
    resolved_split = str(selection_split).strip() if selection_split not in (None, "") else ""
    if not resolved_split:
        resolved_split = hf_split if kind == "huggingface" else "unspecified"

    source_name = str(selection.get("source_name") or "").strip() or _default_source_name(kind, hf_dataset_name)
    source_url = str(selection.get("source_url") or "").strip() or _default_source_url(kind, hf_dataset_name)
    license_name = str(selection.get("license", "unknown")).strip() or "unknown"

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "prepared_root": str(prepared_root),
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
        },
        "source": {
            "kind": kind,
            "root": str(source_root) if source_root is not None else None,
            "openimages_classes": openimages_classes,
            "hf_dataset": hf_dataset_name,
            "hf_split": hf_split,
            "hf_config_name": source.get("hf_config_name"),
            "hf_data_dir": str(resolved_hf_data_dir) if resolved_hf_data_dir is not None else None,
            "hf_image_column": hf_image_column_name,
            "hf_cache_dir": str(resolved_hf_cache_dir),
        },
        "selection": {
            "category": category,
            "split": resolved_split,
            "limit": limit,
            "source_name": source_name,
            "source_url": source_url,
            "license": license_name,
        },
    }


def _iter_image_paths(root: Path) -> list[Path]:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    return paths


def _iter_local_source_items(root: Path, *, source_url: str) -> list[SourceImageItem]:
    return [
        SourceImageItem(
            source_key=path.relative_to(root).as_posix(),
            source_url=source_url,
            image_path=path,
        )
        for path in _iter_image_paths(root)
    ]


def _iter_openimages_source_items(root: Path, classes: Sequence[str], *, source_url: str) -> list[SourceImageItem]:
    if not classes:
        return _iter_local_source_items(root, source_url=source_url)

    selected_paths: dict[str, Path] = {}
    for class_name in classes:
        class_root = root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"OpenImages class directory not found: {class_root}")
        for path in _iter_image_paths(class_root):
            relative_key = path.relative_to(root).as_posix()
            selected_paths[relative_key] = path

    return [
        SourceImageItem(source_key=key, source_url=source_url, image_path=selected_paths[key])
        for key in sorted(selected_paths)
    ]


def _infer_hf_image_column(column_names: Sequence[str], configured: str | None) -> str:
    if configured not in (None, ""):
        if configured not in column_names:
            raise RealImagePreparationError(
                f"Configured Hugging Face image column {configured!r} is missing from the dataset split"
            )
        return configured
    for candidate in ("image", "img"):
        if candidate in column_names:
            return candidate
    raise RealImagePreparationError(
        f"Could not infer Hugging Face image column; available columns are {tuple(column_names)}"
    )


def _decode_image_value(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.copy()
    if isinstance(image_value, Mapping):
        payload = dict(image_value)
        if payload.get("path"):
            with Image.open(payload["path"]) as image:
                return image.copy()
        if payload.get("bytes") is not None:
            with Image.open(io.BytesIO(payload["bytes"])) as image:
                return image.copy()
    raise RealImagePreparationError(f"Unsupported image payload type: {type(image_value)!r}")


def _build_hf_source_key(dataset_name: str, split_name: str, index: int, row: Mapping[str, Any]) -> str:
    pieces = [dataset_name, split_name, f"{index:08d}"]
    for column_name in ("id", "image_id", "file_name", "filename", "path", "url"):
        value = row.get(column_name)
        if value not in (None, ""):
            pieces.append(str(value))
            break
    return "::".join(pieces)


def _iter_hf_source_items(config: Mapping[str, Any], *, source_url: str) -> list[SourceImageItem]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for Hugging Face ingestion") from exc

    source_config = config["source"]
    dataset = load_dataset(
        path=source_config["hf_dataset"],
        name=source_config["hf_config_name"],
        data_dir=source_config["hf_data_dir"],
        split=source_config["hf_split"],
        cache_dir=source_config["hf_cache_dir"],
    )
    image_column = _infer_hf_image_column(dataset.column_names, source_config["hf_image_column"])
    items: list[SourceImageItem] = []
    for index, row in enumerate(dataset):
        row_mapping = dict(row)
        items.append(
            SourceImageItem(
                source_key=_build_hf_source_key(
                    str(source_config["hf_dataset"]),
                    str(source_config["hf_split"]),
                    index,
                    row_mapping,
                ),
                source_url=str(row_mapping.get("url") or row_mapping.get("source_url") or source_url),
                image_value=row_mapping[image_column],
            )
        )
    return items


def iter_source_items(config: Mapping[str, Any]) -> list[SourceImageItem]:
    kind = config["source"]["kind"]
    source_url = config["selection"]["source_url"]
    if kind == "ffhq":
        return _iter_local_source_items(Path(config["source"]["root"]), source_url=source_url)
    if kind == "openimages":
        return _iter_openimages_source_items(
            Path(config["source"]["root"]),
            config["source"]["openimages_classes"],
            source_url=source_url,
        )
    return _iter_hf_source_items(config, source_url=source_url)


def _normalize_image(image: Image.Image) -> Image.Image:
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.fit(
        prepared,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=_RESAMPLING.LANCZOS,
        centering=(0.5, 0.5),
    )


def _load_source_image(item: SourceImageItem) -> Image.Image:
    if item.image_path is not None:
        with Image.open(item.image_path) as image:
            return image.copy()
    return _decode_image_value(item.image_value)


def build_real_image_id(*, category: str, source_name: str, split: str, source_key: str) -> str:
    category_token = _slug_token(category, fallback="real")
    source_token = _slug_token(source_name, fallback="source")
    payload = f"{category}|{source_name}|{split}|{source_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).hexdigest()
    return f"real_{category_token}_{source_token}_{digest}"


def load_real_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != REAL_IMAGE_MANIFEST_COLUMNS:
            raise RealImageManifestValidationError(
                f"{path.name}: expected columns {REAL_IMAGE_MANIFEST_COLUMNS}, found {fieldnames}"
            )
        return [dict(row) for row in reader]


def write_real_manifest(rows: Sequence[Mapping[str, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REAL_IMAGE_MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in REAL_IMAGE_MANIFEST_COLUMNS})


def prepare_real_images(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    prepared_root = Path(config["paths"]["prepared_root"])
    manifest_path = Path(config["paths"]["manifest_path"])
    selection = config["selection"]

    items = iter_source_items(config)
    limit = selection["limit"]
    if limit is not None:
        items = items[: int(limit)]
    if not items:
        raise RealImagePreparationError("No source images were selected for preparation")

    existing_rows = {row["image_id"]: row for row in load_real_manifest(manifest_path)}
    prepared_rows: list[dict[str, str]] = []

    for item in items:
        image_id = build_real_image_id(
            category=str(selection["category"]),
            source_name=str(selection["source_name"]),
            split=str(selection["split"]),
            source_key=item.source_key,
        )
        output_path = prepared_root / str(selection["category"]) / f"{image_id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_image = _normalize_image(_load_source_image(item))
        normalized_image.save(output_path, format="PNG")

        row = {
            "image_id": image_id,
            "category": str(selection["category"]),
            "local_path": display_path(output_path, repo_root),
            "source_name": str(selection["source_name"]),
            "source_url": item.source_url or str(selection["source_url"]),
            "license": str(selection["license"]),
            "split": str(selection["split"]),
        }
        existing_rows[image_id] = row
        prepared_rows.append(row)

    all_rows = sorted(
        existing_rows.values(),
        key=lambda row: (row["category"], row["source_name"], row["image_id"]),
    )
    write_real_manifest(all_rows, manifest_path)

    return {
        "prepared_count": len(prepared_rows),
        "manifest_count": len(all_rows),
        "manifest_path": str(manifest_path),
        "category_counts": dict(Counter(row["category"] for row in all_rows)),
        "source_counts": dict(Counter(row["source_name"] for row in all_rows)),
        "sample_paths": [row["local_path"] for row in prepared_rows[:3]],
    }


def validate_real_manifest(
    path: str | Path,
    *,
    repo_root: Path,
    verify_files: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path)
    rows = load_real_manifest(manifest_path)

    image_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=2):
        for column in REAL_IMAGE_MANIFEST_COLUMNS:
            value = row.get(column, "")
            if value is None or not str(value).strip():
                raise RealImageManifestValidationError(
                    f"{manifest_path.name}: row {row_number} has empty value for {column}"
                )

        if row["image_id"] in image_ids:
            raise RealImageManifestValidationError(
                f"{manifest_path.name}: duplicate image_id {row['image_id']!r}"
            )
        image_ids.add(row["image_id"])

        if not row["image_id"].startswith("real_"):
            raise RealImageManifestValidationError(
                f"{manifest_path.name}: row {row_number} image_id must start with 'real_'"
            )

        local_path = Path(row["local_path"])
        if local_path.is_absolute():
            raise RealImageManifestValidationError(
                f"{manifest_path.name}: row {row_number} local_path must stay relative"
            )
        if local_path.suffix.lower() != ".png":
            raise RealImageManifestValidationError(
                f"{manifest_path.name}: row {row_number} local_path must point to a PNG file"
            )
        if local_path.stem != row["image_id"]:
            raise RealImageManifestValidationError(
                f"{manifest_path.name}: row {row_number} local_path stem must match image_id"
            )

        category_counts[row["category"]] += 1
        split_counts[row["split"]] += 1
        source_counts[row["source_name"]] += 1

        if verify_files:
            image_path = (repo_root / local_path).resolve()
            if not image_path.is_file():
                raise RealImageManifestValidationError(
                    f"{manifest_path.name}: row {row_number} points to missing image {row['local_path']!r}"
                )
            with Image.open(image_path) as image:
                normalized = image.convert("RGB")
            if normalized.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise RealImageManifestValidationError(
                    f"{manifest_path.name}: row {row_number} image must be {(IMAGE_SIZE, IMAGE_SIZE)}, found {normalized.size}"
                )

    return {
        "manifest": manifest_path.name,
        "total": len(rows),
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
    }
