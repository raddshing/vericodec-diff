from __future__ import annotations

import importlib.metadata
import platform
import re
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml


_DEPENDENCY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")
_GIB = 1024**3


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


def parse_dependency_name(spec: str) -> str:
    cleaned = spec.strip()
    if "::" in cleaned:
        cleaned = cleaned.split("::", maxsplit=1)[1]
    match = _DEPENDENCY_NAME_RE.match(cleaned)
    if match is None:
        raise ValueError(f"Could not parse dependency name from {spec!r}")
    return match.group(0)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def resolve_runtime_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(dict(raw_config))
    paths = dict(config.get("paths", {}))
    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    output_dir = resolve_path(resolved_repo_root, str(paths.get("output_dir", "logs/tests"))).resolve()

    disk_checks = []
    for entry in paths.get("disk_checks", []):
        disk_checks.append(
            {
                "name": str(entry["name"]),
                "path": str(resolve_path(resolved_repo_root, str(entry["path"])).resolve()),
                "must_exist": bool(entry.get("must_exist", True)),
            }
        )

    packages = dict(config.get("packages", {}))
    environment_files = [
        str(resolve_path(resolved_repo_root, str(path)).resolve())
        for path in packages.get("environment_files", [])
    ]
    distribution_candidates = {
        str(name): [str(candidate) for candidate in candidates]
        for name, candidates in dict(packages.get("distribution_candidates", {})).items()
    }

    checkpoints = []
    for entry in config.get("checkpoints", []):
        checkpoints.append(
            {
                "name": str(entry["name"]),
                "path": str(resolve_path(resolved_repo_root, str(entry["path"])).resolve()),
                "type": str(entry.get("type", "any")),
                "required": bool(entry.get("required", True)),
            }
        )

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "output_dir": str(output_dir),
            "disk_checks": disk_checks,
        },
        "packages": {
            "environment_files": environment_files,
            "exclude_names": [str(name) for name in packages.get("exclude_names", [])],
            "distribution_candidates": distribution_candidates,
        },
        "checkpoints": checkpoints,
    }


def collect_expected_packages(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    package_config = dict(config.get("packages", {}))
    repo_root = Path(config["paths"]["repo_root"])
    exclude_names = {
        parse_dependency_name(str(name)).lower() for name in package_config.get("exclude_names", [])
    }
    distribution_candidates = {
        str(name).lower(): [str(candidate) for candidate in candidates]
        for name, candidates in dict(package_config.get("distribution_candidates", {})).items()
    }

    packages_by_name: dict[str, dict[str, Any]] = {}
    for raw_path in package_config.get("environment_files", []):
        env_path = Path(raw_path)
        environment_data = load_yaml(env_path)
        dependencies = environment_data.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise ValueError(f"{env_path} dependencies must be a list")

        source = display_path(env_path, repo_root)
        for dependency in dependencies:
            specs: list[str] = []
            if isinstance(dependency, str):
                specs = [dependency]
            elif isinstance(dependency, dict):
                pip_dependencies = dependency.get("pip", [])
                if not isinstance(pip_dependencies, list):
                    raise ValueError(f"{env_path} pip dependencies must be a list")
                specs = [str(spec) for spec in pip_dependencies]

            for spec in specs:
                name = parse_dependency_name(spec)
                key = name.lower()
                if key in exclude_names:
                    continue
                entry = packages_by_name.setdefault(
                    key,
                    {
                        "name": name,
                        "candidates": distribution_candidates.get(key, [name]),
                        "sources": [],
                    },
                )
                if source not in entry["sources"]:
                    entry["sources"].append(source)

    return sorted(packages_by_name.values(), key=lambda item: item["name"].lower())


def inspect_python() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(Path(platform.python_executable() if hasattr(platform, "python_executable") else "")),
    }


def inspect_torch(torch_module: Any | None = None) -> dict[str, Any]:
    if torch_module is None:
        try:
            import torch as imported_torch
        except Exception as exc:
            return {
                "installed": False,
                "version": None,
                "cuda_visible": False,
                "gpu_count": 0,
                "gpu_names": [],
                "cuda_version": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        torch_module = imported_torch

    details = {
        "installed": True,
        "version": getattr(torch_module, "__version__", None),
        "cuda_visible": False,
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
    }
    cuda_api = getattr(torch_module, "cuda", None)
    if cuda_api is None:
        return details

    try:
        details["cuda_visible"] = bool(cuda_api.is_available())
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        return details

    try:
        details["gpu_count"] = int(cuda_api.device_count())
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        return details

    gpu_names: list[str] = []
    for index in range(details["gpu_count"]):
        try:
            gpu_names.append(str(cuda_api.get_device_name(index)))
        except Exception as exc:
            gpu_names.append(f"<error: {type(exc).__name__}: {exc}>")
    details["gpu_names"] = gpu_names
    return details


def inspect_disk(
    path_entries: Sequence[Mapping[str, Any]],
    *,
    disk_usage_func: Callable[[str | Path], shutil._ntuple_diskusage] = shutil.disk_usage,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    missing_required = 0

    for entry in path_entries:
        path = Path(str(entry["path"]))
        must_exist = bool(entry.get("must_exist", True))
        exists = path.exists()
        item = {
            "name": str(entry["name"]),
            "path": str(path),
            "must_exist": must_exist,
            "exists": exists,
        }

        if not exists:
            item["status"] = "missing" if must_exist else "skipped_missing"
            item["free_bytes"] = None
            item["free_gib"] = None
            if must_exist:
                missing_required += 1
            items.append(item)
            continue

        usage = disk_usage_func(path)
        item["status"] = "ok"
        item["free_bytes"] = int(usage.free)
        item["free_gib"] = round(usage.free / _GIB, 2)
        items.append(item)

    return {
        "items": items,
        "summary": {
            "checked": len([item for item in items if item["exists"]]),
            "missing_required": missing_required,
        },
    }


def inspect_packages(
    expected_packages: Sequence[Mapping[str, Any]],
    *,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    missing = 0
    for package in expected_packages:
        matched_candidate = None
        version = None
        for candidate in package["candidates"]:
            try:
                version = version_getter(candidate)
                matched_candidate = candidate
                break
            except importlib.metadata.PackageNotFoundError:
                continue

        present = matched_candidate is not None
        if not present:
            missing += 1

        items.append(
            {
                "name": package["name"],
                "candidates": list(package["candidates"]),
                "sources": list(package["sources"]),
                "present": present,
                "matched_candidate": matched_candidate,
                "version": version,
            }
        )

    return {
        "items": items,
        "summary": {
            "expected": len(items),
            "present": len(items) - missing,
            "missing": missing,
            "missing_names": [item["name"] for item in items if not item["present"]],
        },
    }


def inspect_checkpoints(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    missing_required = 0

    for checkpoint in checkpoints:
        path = Path(str(checkpoint["path"]))
        expected_type = str(checkpoint.get("type", "any"))
        required = bool(checkpoint.get("required", True))
        exists = path.exists()

        present = exists
        if present and expected_type == "file":
            present = path.is_file()
        elif present and expected_type == "directory":
            present = path.is_dir()

        if required and not present:
            missing_required += 1

        items.append(
            {
                "name": str(checkpoint["name"]),
                "path": str(path),
                "type": expected_type,
                "required": required,
                "present": present,
                "status": "present" if present else "missing",
            }
        )

    return {
        "items": sorted(items, key=lambda item: item["name"].lower()),
        "summary": {
            "configured": len(items),
            "present": len([item for item in items if item["present"]]),
            "missing_required": missing_required,
        },
    }


def build_environment_report(
    config: Mapping[str, Any],
    *,
    torch_module: Any | None = None,
    version_getter: Callable[[str], str] = importlib.metadata.version,
    disk_usage_func: Callable[[str | Path], shutil._ntuple_diskusage] = shutil.disk_usage,
) -> dict[str, Any]:
    expected_packages = collect_expected_packages(config)
    torch_report = inspect_torch(torch_module)
    disk_report = inspect_disk(config["paths"]["disk_checks"], disk_usage_func=disk_usage_func)
    packages_report = inspect_packages(expected_packages, version_getter=version_getter)
    checkpoints_report = inspect_checkpoints(config.get("checkpoints", []))

    issues: list[str] = []
    if not torch_report["installed"]:
        issues.append("torch is not importable.")
    elif not torch_report["cuda_visible"]:
        issues.append("torch cannot see CUDA devices.")

    if disk_report["summary"]["missing_required"] > 0:
        issues.append("One or more required disk check paths are missing.")

    if packages_report["summary"]["missing"] > 0:
        issues.append(
            f"{packages_report['summary']['missing']} required Python package(s) are missing."
        )

    if checkpoints_report["summary"]["missing_required"] > 0:
        issues.append(
            f"{checkpoints_report['summary']['missing_required']} required checkpoint(s) are missing."
        )

    return {
        "schema_version": 1,
        "summary": {
            "status": "ok" if not issues else "issues_found",
            "issues": issues,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "torch": torch_report,
        "disk": disk_report,
        "packages": packages_report,
        "checkpoints": checkpoints_report,
    }


def render_environment_report_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Environment Report",
        "",
        "## Summary",
        f"- Status: `{summary['status']}`",
        f"- Python: `{report['python']['version']}` ({report['python']['implementation']})",
        f"- Torch installed: `{'yes' if report['torch']['installed'] else 'no'}`",
        f"- Torch CUDA visible: `{'yes' if report['torch']['cuda_visible'] else 'no'}`",
        f"- GPU count: `{report['torch']['gpu_count']}`",
        f"- Torch CUDA version: `{report['torch']['cuda_version'] or 'n/a'}`",
        "",
    ]

    if summary["issues"]:
        lines.extend(["## Issues"])
        for issue in summary["issues"]:
            lines.append(f"- {issue}")
        lines.append("")

    lines.extend(
        [
            "## Disk",
            "| Name | Path | Status | Free GiB | Free bytes |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for item in report["disk"]["items"]:
        lines.append(
            f"| {item['name']} | `{item['path']}` | `{item['status']}` | "
            f"{item['free_gib'] if item['free_gib'] is not None else 'n/a'} | "
            f"{item['free_bytes'] if item['free_bytes'] is not None else 'n/a'} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Torch",
            "| Field | Value |",
            "| --- | --- |",
            f"| Version | `{report['torch']['version'] or 'n/a'}` |",
            f"| CUDA visible | `{'yes' if report['torch']['cuda_visible'] else 'no'}` |",
            f"| GPU count | `{report['torch']['gpu_count']}` |",
            f"| GPU names | `{', '.join(report['torch']['gpu_names']) if report['torch']['gpu_names'] else 'n/a'}` |",
            f"| CUDA version | `{report['torch']['cuda_version'] or 'n/a'}` |",
        ]
    )
    if report["torch"].get("error"):
        lines.append(f"| Error | `{report['torch']['error']}` |")
    lines.append("")

    lines.extend(
        [
            "## Python Packages",
            "| Package | Present | Version | Source |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in report["packages"]["items"]:
        lines.append(
            f"| {item['name']} | `{'yes' if item['present'] else 'no'}` | "
            f"`{item['version'] or 'n/a'}` | `{', '.join(item['sources'])}` |"
        )
    lines.append("")

    lines.extend(
        [
            "## Checkpoints",
            "| Name | Path | Type | Required | Present |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    if report["checkpoints"]["items"]:
        for item in report["checkpoints"]["items"]:
            lines.append(
                f"| {item['name']} | `{item['path']}` | `{item['type']}` | "
                f"`{'yes' if item['required'] else 'no'}` | `{'yes' if item['present'] else 'no'}` |"
            )
    else:
        lines.append("| none configured | `n/a` | `n/a` | `n/a` | `n/a` |")

    lines.append("")
    return "\n".join(lines)

