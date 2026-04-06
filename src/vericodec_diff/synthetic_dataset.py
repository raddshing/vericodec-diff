from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


IMAGE_SIZE = 1024
MANIFEST_COLUMNS = ("image_id", "category", "local_path", "split", "seed")
SYNTHETIC_CATEGORIES = (
    "text-posters-signs",
    "thin-boundary",
    "repeated-pattern",
)
LOCKED_SPLIT_COUNTS = {
    "kill": 48,
    "main": 100,
    "hard": 30,
}
_TEXT_SNIPPETS = (
    "OPEN LATE",
    "MARKET",
    "FRESH",
    "EXIT 24",
    "NO ENTRY",
    "SLOW",
    "CITY HALL",
    "CAFE 81",
    "PLATFORM 6",
    "RED LINE",
    "NIGHT BUS",
    "YARD SALE",
    "EAST GATE",
    "CHECK IN",
    "MUSEUM",
    "CLEARANCE",
)
_TEXT_SUPPORT_LINES = (
    "WALK UPS ONLY",
    "LEFT LANE CLOSED",
    "ALL SALES FINAL",
    "PAY AT COUNTER",
    "KEEP MOVING",
    "AUTHORIZED STAFF",
    "PICK UP HERE",
    "OPEN WEEKENDS",
    "CAUTION WET FLOOR",
    "LAST TRAIN 23:40",
)
_RESAMPLING = getattr(Image, "Resampling", Image)


class SyntheticManifestValidationError(ValueError):
    """Raised when the synthetic manifest violates the local schema."""


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


def default_generation_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "raw_root": "data/raw/synthetic",
            "manifest_path": "data/manifests/synthetic_manifest.csv",
            "output_dir": "outputs/diagnostic_dataset",
        },
        "dataset": {
            "image_size": IMAGE_SIZE,
            "base_seed": 20260406,
            "splits": list(LOCKED_SPLIT_COUNTS),
            "categories": list(SYNTHETIC_CATEGORIES),
            "split_counts": deepcopy(LOCKED_SPLIT_COUNTS),
        },
    }


def resolve_generation_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(raw_config))
    paths = dict(config.get("paths", {}))
    dataset = dict(config.get("dataset", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    raw_root = resolve_path(resolved_repo_root, str(paths.get("raw_root", "data/raw/synthetic"))).resolve()
    manifest_path = resolve_path(
        resolved_repo_root,
        str(paths.get("manifest_path", "data/manifests/synthetic_manifest.csv")),
    ).resolve()
    output_dir = resolve_path(
        resolved_repo_root,
        str(paths.get("output_dir", "outputs/diagnostic_dataset")),
    ).resolve()

    for name, path in {
        "raw_root": raw_root,
        "manifest_path": manifest_path,
        "output_dir": output_dir,
    }.items():
        try:
            path.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ValueError(f"{name} must resolve inside repo_root for stable local paths") from exc

    image_size = int(dataset.get("image_size", IMAGE_SIZE))
    if image_size != IMAGE_SIZE:
        raise ValueError(f"VeriCodec-Diff synthetic diagnostics are locked to {IMAGE_SIZE}x{IMAGE_SIZE}")

    base_seed = int(dataset.get("base_seed", 20260406))
    if base_seed <= 0:
        raise ValueError("base_seed must be a positive integer")

    categories = [str(item) for item in dataset.get("categories", SYNTHETIC_CATEGORIES)]
    if not categories:
        raise ValueError("At least one category must be selected")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must be unique")
    unsupported_categories = [item for item in categories if item not in SYNTHETIC_CATEGORIES]
    if unsupported_categories:
        raise ValueError(f"Unsupported synthetic categories: {unsupported_categories}")

    splits = [str(item) for item in dataset.get("splits", LOCKED_SPLIT_COUNTS)]
    if not splits:
        raise ValueError("At least one split must be selected")
    if len(set(splits)) != len(splits):
        raise ValueError("splits must be unique")
    unsupported_splits = [item for item in splits if item not in LOCKED_SPLIT_COUNTS]
    if unsupported_splits:
        raise ValueError(f"Unsupported splits: {unsupported_splits}")

    raw_split_counts = dict(LOCKED_SPLIT_COUNTS)
    raw_split_counts.update(dict(dataset.get("split_counts", {})))
    split_counts = {}
    for split in LOCKED_SPLIT_COUNTS:
        count = int(raw_split_counts[split])
        if count < 0:
            raise ValueError(f"split_counts[{split!r}] must be non-negative")
        split_counts[split] = count

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "raw_root": str(raw_root),
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir),
        },
        "dataset": {
            "image_size": image_size,
            "base_seed": base_seed,
            "splits": splits,
            "categories": categories,
            "split_counts": split_counts,
        },
    }


def parse_csv_items(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_count_override(raw_value: str) -> tuple[str, int]:
    split, separator, count_text = raw_value.partition("=")
    if separator != "=":
        raise ValueError(f"Count override must look like split=count, received {raw_value!r}")
    split = split.strip()
    if split not in LOCKED_SPLIT_COUNTS:
        raise ValueError(f"Unsupported split in count override: {split!r}")
    count = int(count_text.strip())
    if count < 0:
        raise ValueError("Count override must be non-negative")
    return split, count


def build_image_id(split: str, category: str, index: int) -> str:
    category_token = category.replace("-", "_")
    return f"synthetic_{split}_{category_token}_{index:03d}"


def derive_image_seed(base_seed: int, split: str, category: str, index: int) -> int:
    payload = f"{base_seed}:{split}:{category}:{index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return (int.from_bytes(digest, byteorder="big") % 2_147_483_646) + 1


def build_generation_plan(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    repo_root = Path(config["paths"]["repo_root"])
    raw_root = Path(config["paths"]["raw_root"])
    dataset = config["dataset"]
    raw_root_relative = raw_root.relative_to(repo_root)

    plan: list[dict[str, Any]] = []
    for split in dataset["splits"]:
        count = int(dataset["split_counts"][split])
        for category in dataset["categories"]:
            for index in range(1, count + 1):
                image_id = build_image_id(split, category, index)
                seed = derive_image_seed(int(dataset["base_seed"]), split, category, index)
                image_path = raw_root / category / f"{image_id}.png"
                plan.append(
                    {
                        "image_id": image_id,
                        "category": category,
                        "split": split,
                        "seed": seed,
                        "index": index,
                        "image_path": image_path,
                        "local_path": (raw_root_relative / category / image_path.name).as_posix(),
                    }
                )
    return plan


def _interpolate_channel(start: int, end: int, ratio: float) -> int:
    return int(round(start + (end - start) * ratio))


def _gradient_image(size: tuple[int, int], start: tuple[int, int, int], end: tuple[int, int, int], *, horizontal: bool = False) -> Image.Image:
    image = Image.new("RGBA", size)
    draw = ImageDraw.Draw(image)
    length = max(size[0] if horizontal else size[1], 1)
    for offset in range(length):
        ratio = offset / max(length - 1, 1)
        color = tuple(_interpolate_channel(start[i], end[i], ratio) for i in range(3)) + (255,)
        if horizontal:
            draw.line([(offset, 0), (offset, size[1])], fill=color)
        else:
            draw.line([(0, offset), (size[0], offset)], fill=color)
    return image


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf")
        if bold
        else ("DejaVuSans.ttf", "DejaVuSansMono.ttf", "DejaVuSans-Bold.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _paste_centered(base: Image.Image, layer: Image.Image, center: tuple[int, int]) -> None:
    left = int(center[0] - (layer.width / 2))
    top = int(center[1] - (layer.height / 2))
    base.paste(layer, (left, top), layer)


def _wrap_text(text: str, line_length: int) -> str:
    words = text.split()
    if not words:
        return text
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= line_length:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return "\n".join(lines)


def _draw_text_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    spacing: int = 6,
) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = box[0] + ((box[2] - box[0] - text_width) / 2)
    y = box[1] + ((box[3] - box[1] - text_height) / 2)
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def _build_code_tag(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
    digits = "".join(rng.choice("0123456789") for _ in range(3))
    return f"{letters}-{digits}"


def _quadratic_curve(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    *,
    steps: int = 40,
) -> list[tuple[float, float]]:
    points = []
    for step in range(steps + 1):
        t = step / steps
        x = ((1 - t) ** 2 * start[0]) + (2 * (1 - t) * t * control[0]) + (t**2 * end[0])
        y = ((1 - t) ** 2 * start[1]) + (2 * (1 - t) * t * control[1]) + (t**2 * end[1])
        points.append((x, y))
    return points


def render_text_posters_signs(size: int, rng: random.Random) -> Image.Image:
    palette = rng.choice(
        (
            {"bg_start": (248, 239, 221), "bg_end": (225, 204, 171), "ink": (33, 36, 40), "accent": (195, 52, 40), "panel": (243, 231, 208)},
            {"bg_start": (229, 239, 245), "bg_end": (174, 202, 216), "ink": (25, 39, 52), "accent": (219, 109, 36), "panel": (244, 248, 250)},
            {"bg_start": (238, 230, 236), "bg_end": (206, 188, 206), "ink": (49, 31, 58), "accent": (32, 132, 113), "panel": (246, 242, 245)},
        )
    )
    image = _gradient_image((size, size), palette["bg_start"], palette["bg_end"])
    draw = ImageDraw.Draw(image, "RGBA")

    for offset in range(0, size, 28):
        alpha = 12 if (offset // 28) % 2 == 0 else 6
        draw.line([(0, offset), (size, offset + rng.randint(-14, 14))], fill=palette["ink"] + (alpha,), width=1)

    panel_count = rng.randint(4, 6)
    headline_font = _load_font(rng.randint(68, 110), bold=True)
    support_font = _load_font(rng.randint(18, 28), bold=False)
    badge_font = _load_font(26, bold=True)

    for _ in range(panel_count):
        panel_width = rng.randint(340, 720)
        panel_height = rng.randint(160, 280)
        layer = Image.new("RGBA", (panel_width, panel_height), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer, "RGBA")
        panel_fill = palette["panel"] + (235,)
        border = palette["ink"] + (255,)
        layer_draw.rounded_rectangle(
            [8, 8, panel_width - 8, panel_height - 8],
            radius=22,
            fill=panel_fill,
            outline=border,
            width=4,
        )
        stripe_y = rng.randint(18, 40)
        layer_draw.rectangle(
            [16, stripe_y, panel_width - 16, stripe_y + rng.randint(18, 36)],
            fill=palette["accent"] + (235,),
        )

        slogan = _wrap_text(rng.choice(_TEXT_SNIPPETS), 10)
        _draw_text_centered(
            layer_draw,
            (28, 38, panel_width - 28, int(panel_height * 0.65)),
            slogan,
            font=headline_font,
            fill=palette["ink"] + (255,),
            spacing=10,
        )

        detail_y = int(panel_height * 0.68)
        for detail_index in range(rng.randint(2, 4)):
            detail = f"{rng.choice(_TEXT_SUPPORT_LINES)}  {_build_code_tag(rng)}"
            layer_draw.text(
                (30, detail_y + (detail_index * 28)),
                detail,
                font=support_font,
                fill=palette["ink"] + (210,),
            )

        badge_box = [panel_width - 126, panel_height - 74, panel_width - 24, panel_height - 20]
        layer_draw.rounded_rectangle(badge_box, radius=14, fill=palette["accent"] + (235,))
        badge_text = rng.choice(("NEW", "SALE", "ZONE", "EXIT", "FAST", "OPEN"))
        _draw_text_centered(
            layer_draw,
            tuple(int(value) for value in badge_box),
            badge_text,
            font=badge_font,
            fill=(255, 247, 240, 255),
        )

        rotated = layer.rotate(rng.randint(-12, 12), resample=_RESAMPLING.BICUBIC, expand=True)
        center = (rng.randint(180, size - 180), rng.randint(160, size - 160))
        _paste_centered(image, rotated, center)

    footer = Image.new("RGBA", (size - 120, 96), (0, 0, 0, 0))
    footer_draw = ImageDraw.Draw(footer, "RGBA")
    footer_draw.rounded_rectangle([0, 0, footer.width, footer.height], radius=22, fill=(26, 30, 34, 208))
    footer_text = "  ".join(rng.choice(_TEXT_SUPPORT_LINES) for _ in range(6))
    footer_draw.text((24, 28), footer_text, font=_load_font(24, bold=False), fill=(247, 242, 232, 230))
    _paste_centered(image, footer, (size // 2, size - 82))
    return image.convert("RGB")


def render_thin_boundary_scene(size: int, rng: random.Random) -> Image.Image:
    palette = rng.choice(
        (
            {"bg_start": (18, 26, 39), "bg_end": (52, 74, 101), "line": (238, 247, 255), "accent": (113, 219, 216)},
            {"bg_start": (28, 18, 29), "bg_end": (88, 55, 82), "line": (251, 246, 238), "accent": (255, 188, 115)},
            {"bg_start": (13, 28, 20), "bg_end": (50, 97, 74), "line": (244, 250, 242), "accent": (166, 231, 184)},
        )
    )
    image = _gradient_image((size, size), palette["bg_start"], palette["bg_end"])
    line_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(line_layer, "RGBA")

    for _ in range(18):
        x = rng.randint(0, size)
        draw.line([(x, 0), (x + rng.randint(-40, 40), size)], fill=palette["line"] + (12,), width=1)
    for _ in range(24):
        y = rng.randint(0, size)
        draw.line([(0, y), (size, y + rng.randint(-36, 36))], fill=palette["accent"] + (10,), width=1)

    for _ in range(rng.randint(28, 38)):
        edge = rng.choice(("left", "right", "top", "bottom"))
        if edge == "left":
            start = (0.0, float(rng.randint(0, size)))
            end = (float(size), float(rng.randint(0, size)))
        elif edge == "right":
            start = (float(size), float(rng.randint(0, size)))
            end = (0.0, float(rng.randint(0, size)))
        elif edge == "top":
            start = (float(rng.randint(0, size)), 0.0)
            end = (float(rng.randint(0, size)), float(size))
        else:
            start = (float(rng.randint(0, size)), float(size))
            end = (float(rng.randint(0, size)), 0.0)
        control = (float(rng.randint(0, size)), float(rng.randint(0, size)))
        points = _quadratic_curve(start, control, end, steps=48)
        width = 1 if rng.random() < 0.72 else 2
        color = palette["line"] if rng.random() < 0.7 else palette["accent"]
        alpha = rng.randint(120, 220)
        draw.line(points, fill=color + (alpha,), width=width, joint="curve")

    for _ in range(rng.randint(16, 24)):
        center_x = rng.randint(60, size - 60)
        center_y = rng.randint(60, size - 60)
        radius_x = rng.randint(18, 120)
        radius_y = rng.randint(18, 140)
        outline = palette["line"] if rng.random() < 0.65 else palette["accent"]
        box = [center_x - radius_x, center_y - radius_y, center_x + radius_x, center_y + radius_y]
        draw.ellipse(box, outline=outline + (rng.randint(96, 196),), width=rng.randint(1, 2))
        if rng.random() < 0.4:
            inner_scale = rng.uniform(0.45, 0.75)
            inner_box = [
                center_x - int(radius_x * inner_scale),
                center_y - int(radius_y * inner_scale),
                center_x + int(radius_x * inner_scale),
                center_y + int(radius_y * inner_scale),
            ]
            draw.ellipse(inner_box, outline=outline + (rng.randint(64, 160),), width=1)

    for _ in range(rng.randint(12, 18)):
        polygon = []
        cx = rng.randint(80, size - 80)
        cy = rng.randint(80, size - 80)
        radius = rng.randint(30, 120)
        sides = rng.randint(3, 7)
        for side in range(sides):
            angle = (2 * math.pi * side / sides) + rng.uniform(-0.18, 0.18)
            polygon.append((cx + math.cos(angle) * radius, cy + math.sin(angle) * radius))
        draw.line(polygon + [polygon[0]], fill=palette["accent"] + (rng.randint(80, 170),), width=1)

    image.alpha_composite(line_layer)
    return image.convert("RGB")


def render_repeated_pattern_layout(size: int, rng: random.Random) -> Image.Image:
    palette = rng.choice(
        (
            {"bg_start": (236, 235, 227), "bg_end": (192, 195, 187), "grid": (74, 83, 91), "accent": (217, 136, 72), "shadow": (147, 153, 159)},
            {"bg_start": (222, 230, 236), "bg_end": (174, 192, 207), "grid": (38, 63, 79), "accent": (220, 92, 57), "shadow": (139, 153, 166)},
            {"bg_start": (236, 229, 238), "bg_end": (199, 186, 209), "grid": (63, 44, 84), "accent": (53, 146, 131), "shadow": (153, 142, 165)},
        )
    )
    image = _gradient_image((size, size), palette["bg_start"], palette["bg_end"])
    draw = ImageDraw.Draw(image, "RGBA")
    style = rng.choice(("window-grid", "tile-grid", "weave-grid"))
    cell_width = rng.randint(48, 86)
    cell_height = rng.randint(48, 94)
    row_offset = 0 if style == "window-grid" else rng.randint(0, cell_width // 2)

    for row_index, top in enumerate(range(0, size + cell_height, cell_height)):
        x_offset = row_offset if row_index % 2 else 0
        for col_index, left in enumerate(range(-cell_width, size + cell_width, cell_width)):
            x0 = left + x_offset
            y0 = top
            x1 = x0 + cell_width - 6
            y1 = y0 + cell_height - 6
            if x1 < 0 or y1 < 0 or x0 > size or y0 > size:
                continue

            if style == "window-grid":
                fill = palette["shadow"] + (145,)
                accent_fill = palette["accent"] + (70 if (row_index + col_index) % 3 else 150,)
                draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=fill, outline=palette["grid"] + (180,), width=2)
                inset = 8
                draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], outline=palette["grid"] + (120,), width=1)
                draw.line([(x0 + (cell_width // 2), y0 + inset), (x0 + (cell_width // 2), y1 - inset)], fill=palette["grid"] + (120,), width=1)
                draw.rectangle([x0 + inset + 2, y0 + inset + 2, x1 - inset - 2, y1 - inset - 2], fill=accent_fill)
            elif style == "tile-grid":
                fill = palette["shadow"] + (115,)
                draw.rectangle([x0, y0, x1, y1], fill=fill, outline=palette["grid"] + (160,), width=2)
                stripe_step = max(6, cell_width // 6)
                for stripe in range(x0 + 4, x1 - 4, stripe_step):
                    draw.line([(stripe, y0 + 4), (stripe, y1 - 4)], fill=palette["grid"] + (80,), width=1)
                if (row_index + col_index) % 2 == 0:
                    draw.rectangle([x0 + 10, y0 + 10, x1 - 10, y1 - 10], outline=palette["accent"] + (170,), width=2)
            else:
                fill = palette["shadow"] + (98,)
                draw.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=fill, outline=palette["grid"] + (120,), width=1)
                for band in range(0, cell_height, 9):
                    draw.line([(x0 + 4, y0 + band), (x1 - 4, y0 + band + rng.randint(-3, 3))], fill=palette["grid"] + (85,), width=1)
                for band in range(0, cell_width, 9):
                    draw.line([(x0 + band, y0 + 4), (x0 + band + rng.randint(-3, 3), y1 - 4)], fill=palette["accent"] + (72,), width=1)

    dot_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    dot_draw = ImageDraw.Draw(dot_layer, "RGBA")
    for y in range(18, size, 24):
        for x in range(18, size, 24):
            radius = 1 if (x + y) % 48 else 2
            color = palette["grid"] if (x // 24 + y // 24) % 4 else palette["accent"]
            dot_draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color + (68,))
    image.alpha_composite(dot_layer)
    return image.convert("RGB")


_CATEGORY_RENDERERS = {
    "text-posters-signs": render_text_posters_signs,
    "thin-boundary": render_thin_boundary_scene,
    "repeated-pattern": render_repeated_pattern_layout,
}


def write_manifest(rows: Sequence[Mapping[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in MANIFEST_COLUMNS})


def generate_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    manifest_path = Path(config["paths"]["manifest_path"])
    plan = build_generation_plan(config)
    rows: list[dict[str, Any]] = []

    for item in plan:
        image_path = Path(item["image_path"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = _CATEGORY_RENDERERS[item["category"]]
        image = renderer(int(config["dataset"]["image_size"]), random.Random(int(item["seed"])))
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"{item['image_id']} rendered at {image.size}, expected {(IMAGE_SIZE, IMAGE_SIZE)}")
        image.save(image_path, format="PNG")
        rows.append({column: item[column] for column in MANIFEST_COLUMNS})

    write_manifest(rows, manifest_path)
    return {
        "image_count": len(rows),
        "manifest_path": str(manifest_path),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "sample_paths": [display_path(Path(plan_item["image_path"]), repo_root) for plan_item in plan[:3]],
    }


def validate_synthetic_manifest(
    path: str | Path,
    *,
    repo_root: Path,
    config: Mapping[str, Any] | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != MANIFEST_COLUMNS:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: expected columns {MANIFEST_COLUMNS}, found {fieldnames}"
            )
        rows = list(reader)

    image_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=2):
        for column in MANIFEST_COLUMNS:
            value = row.get(column, "")
            if value is None or not str(value).strip():
                raise SyntheticManifestValidationError(
                    f"{manifest_path.name}: row {row_number} has empty value for {column}"
                )

        if row["category"] not in SYNTHETIC_CATEGORIES:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: row {row_number} has unsupported category {row['category']!r}"
            )
        if row["split"] not in LOCKED_SPLIT_COUNTS:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: row {row_number} has unsupported split {row['split']!r}"
            )
        if Path(row["local_path"]).is_absolute():
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: row {row_number} local_path must stay relative"
            )
        expected_prefix = f"synthetic_{row['split']}_{row['category'].replace('-', '_')}_"
        if not row["image_id"].startswith(expected_prefix):
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: row {row_number} image_id does not match split/category"
            )
        try:
            seed = int(row["seed"])
        except ValueError as exc:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: row {row_number} has non-integer seed {row['seed']!r}"
            ) from exc
        if seed <= 0:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: row {row_number} seed must be positive"
            )
        if row["image_id"] in image_ids:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: duplicate image_id {row['image_id']!r}"
            )
        image_ids.add(row["image_id"])
        category_counts[row["category"]] += 1
        split_counts[row["split"]] += 1

        if verify_files:
            image_path = (repo_root / row["local_path"]).resolve()
            if not image_path.is_file():
                raise SyntheticManifestValidationError(
                    f"{manifest_path.name}: row {row_number} points to missing image {row['local_path']!r}"
                )

    if config is not None:
        selected_categories = list(config["dataset"]["categories"])
        selected_splits = list(config["dataset"]["splits"])
        expected_total = sum(int(config["dataset"]["split_counts"][split]) for split in selected_splits) * len(
            selected_categories
        )
        if len(rows) != expected_total:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: expected {expected_total} rows, found {len(rows)}"
            )
        expected_category_counts = {
            category: sum(int(config["dataset"]["split_counts"][split]) for split in selected_splits)
            for category in selected_categories
        }
        expected_split_counts = {
            split: int(config["dataset"]["split_counts"][split]) * len(selected_categories)
            for split in selected_splits
        }
        if dict(category_counts) != expected_category_counts:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: expected category counts {expected_category_counts}, found {dict(category_counts)}"
            )
        if dict(split_counts) != expected_split_counts:
            raise SyntheticManifestValidationError(
                f"{manifest_path.name}: expected split counts {expected_split_counts}, found {dict(split_counts)}"
            )

    return {
        "manifest": manifest_path.name,
        "total": len(rows),
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
    }
