from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

try:
    from omegaconf import OmegaConf as _OmegaConf
except ImportError:
    _OmegaConf = None


class _FallbackOmegaConf:
    @staticmethod
    def create(data: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(data)

    @staticmethod
    def to_container(config: dict[str, Any], resolve: bool = True) -> dict[str, Any]:
        return deepcopy(config)

    @staticmethod
    def save(config: dict[str, Any], path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)


OmegaConf = _OmegaConf or _FallbackOmegaConf()

