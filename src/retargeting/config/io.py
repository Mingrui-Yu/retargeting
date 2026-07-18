from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def to_plain_config_data(data: Any) -> Any:
    """Convert nested config containers to plain Python values.

    Args:
        data: A nested mapping, sequence, or scalar produced by a YAML loader or Hydra.

    Returns:
        A recursively converted structure using built-in dict, list, tuple, and scalar types.
    """
    if isinstance(data, dict) or hasattr(data, "items"):
        return {str(key): to_plain_config_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [to_plain_config_data(item) for item in data]
    if isinstance(data, tuple):
        return tuple(to_plain_config_data(item) for item in data)
    return data


def project_root(start: str | Path | None = None) -> Path:
    current = Path.cwd() if start is None else Path(start).resolve()
    if current.is_file():
        current = current.parent
    for path in [current, *current.parents]:
        if (path / "pyproject.toml").is_file() and (path / "src" / "retargeting").is_dir():
            return path
    return Path.cwd()


def resolve_project_path(path: str | Path, root: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root(root) / path


def resolve_asset_path(path: str | Path, root: str | Path | None = None, *, follow_symlink: bool = True) -> Path:
    resolved = resolve_project_path(path, root=root)
    if not follow_symlink or not resolved.is_symlink():
        return resolved

    target = Path(os.readlink(resolved))
    if target.is_absolute():
        return target

    parent_relative = resolved.parent / target
    if parent_relative.exists():
        return parent_relative

    root_relative = project_root(root) / target
    if root_relative.exists():
        return root_relative

    return parent_relative


def load_config_data(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {config_path}.")
    return data


def load_config_source(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load typed-config source data from a path or composed mapping.

    Args:
        source: YAML path, plain mapping, or Hydra/OmegaConf-style mapping.

    Returns:
        Plain Python dictionary for schema construction.
    """
    if isinstance(source, Mapping) or hasattr(source, "items"):
        data = to_plain_config_data(source)
        if not isinstance(data, dict):
            raise ValueError("Expected mapping config data.")
        return data
    return load_config_data(source)
