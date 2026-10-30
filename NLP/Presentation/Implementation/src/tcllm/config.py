"""Configuration loading, path resolution and reproducibility helpers.

Paths are derived from the file's own location, so scripts work no matter which
directory they are launched from. The artifacts root can be redirected with the
``TCLLM_ARTIFACTS`` environment variable, which is how a Colab run points at
Drive — or at local disk, which is much faster — without editing any code.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

# src/tcllm/config.py -> src/tcllm -> src -> Implementation/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_DATASETS_CONFIG = CONFIG_DIR / "datasets.yaml"

ARTIFACTS_ENV_VAR = "TCLLM_ARTIFACTS"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"

# huggingface_hub warns on every single call that a Windows machine without
# Developer Mode cannot symlink its cache. The warning is accurate but not
# actionable here, and it buries the real output. Downloads still work, they just
# use more disk. Opt out of the warning only, never of the cache itself.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def artifacts_dir(override: str | Path | None = None) -> Path:
    """Resolve the artifacts root: explicit argument > env var > ``Implementation/artifacts``."""
    if override is not None:
        return Path(override).expanduser().resolve()
    from_env = os.environ.get(ARTIFACTS_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser().resolve()
    return DEFAULT_ARTIFACTS


def data_dir(artifacts: str | Path | None = None, *parts: str) -> Path:
    """``<artifacts>/data/<parts...>``, created on demand by the writers."""
    return Path(artifacts_dir(artifacts)).joinpath("data", *parts)


def processed_dir(task: str, language_iso: str, artifacts: str | Path | None = None) -> Path:
    return data_dir(artifacts, "processed", task, language_iso)


def raw_cache_dir(artifacts: str | Path | None = None) -> Path:
    return data_dir(artifacts, "raw", "hf_cache")


# --------------------------------------------------------------------------- #
# yaml
# --------------------------------------------------------------------------- #
def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level, got {type(data).__name__}")
    return data


def load_datasets_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and sanity-check ``configs/datasets.yaml``."""
    cfg = load_yaml(path or DEFAULT_DATASETS_CONFIG)
    validate_datasets_config(cfg)
    return cfg


def validate_datasets_config(cfg: dict[str, Any]) -> None:
    """Fail early on inconsistent registry entries rather than mid-experiment."""
    languages = cfg.get("languages") or {}
    datasets = cfg.get("datasets") or {}
    if not languages:
        raise ValueError("datasets config has no 'languages' section")
    if not datasets:
        raise ValueError("datasets config has no 'datasets' section")

    for key, lang in languages.items():
        for field in ("iso", "name"):
            if field not in lang:
                raise ValueError(f"language {key!r} is missing the {field!r} field")

    for task, spec in datasets.items():
        labels = spec.get("labels")
        n_labels = spec.get("n_labels")
        if labels is None or n_labels is None:
            raise ValueError(f"dataset {task!r} must declare both 'labels' and 'n_labels'")
        if len(labels) != n_labels:
            raise ValueError(
                f"dataset {task!r}: n_labels={n_labels} but {len(labels)} labels are listed"
            )

    declared = set(cfg.get("paper_languages") or [])
    unknown = declared - set(languages)
    if unknown:
        raise ValueError(f"paper_languages references unknown languages: {sorted(unknown)}")


# --------------------------------------------------------------------------- #
# registry lookups
# --------------------------------------------------------------------------- #
def get_dataset_spec(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    try:
        return cfg["datasets"][task]
    except KeyError:
        known = ", ".join(sorted(cfg["datasets"]))
        raise KeyError(f"unknown task {task!r}; configs/datasets.yaml declares: {known}") from None


def resolve_language(cfg: dict[str, Any], language: str) -> tuple[str, dict[str, Any]]:
    """Accept either the registry key (``english``) or the ISO code (``en``).

    Returns ``(key, spec)``. Accepting both keeps artifact paths short (``en``)
    while the paper's tables, which use language names, stay readable.
    """
    languages = cfg["languages"]
    if language in languages:
        return language, languages[language]
    for key, spec in languages.items():
        if spec.get("iso") == language:
            return key, spec
    known = ", ".join(sorted(languages))
    raise KeyError(f"unknown language {language!r}; configs/datasets.yaml declares: {known}")


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed every RNG that might reach a result. Optional deps are seeded if importable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
