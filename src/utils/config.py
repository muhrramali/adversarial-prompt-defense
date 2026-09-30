"""
src/utils/config.py
====================
Centralized configuration loader for the project.

WHY THIS EXISTS
---------------
Every later module needs to read `configs/default.yaml` and override values
with environment variables from `.env`. Hard-coding `yaml.safe_load(open(...))`
in every file would duplicate logic and make testing painful. This module
provides a single `load_config()` function that:

1. Reads the YAML config file.
2. Loads `.env` (without overwriting existing OS env vars).
3. Overrides specific config fields with env vars (so secrets never enter YAML).
4. Sets the global random seed for reproducibility.

USAGE
-----
    from src.utils.config import load_config
    cfg = load_config()
    print(cfg["detection"]["default_detector"])  # "rule_based"

HOW IT WORKS
------------
- `yaml.safe_load` parses the YAML file into a nested dict.
- `dotenv.load_dotenv()` populates `os.environ` from `.env` (if present).
- We then walk the dict and selectively override based on env vars.
- `random.seed` and `numpy.random.seed` are set so that dataset splitting,
  model initialization, and shuffle operations are reproducible.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Project root = parent of parent of this file:
#   src/utils/config.py  ->  src/utils/  ->  src/  ->  PROJECT_ROOT
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_ENV_PATH: Path = PROJECT_ROOT / ".env"


def load_config(config_path: str | os.PathLike | None = None,
                env_path: str | os.PathLike | None = None) -> dict[str, Any]:
    """
    Load YAML configuration with environment variable overrides applied.

    Parameters
    ----------
    config_path : path to the YAML config file. Defaults to configs/default.yaml.
    env_path    : path to the .env file. Defaults to .env in project root.

    Returns
    -------
    A nested dict representing the merged configuration.

    Side effects
    ------------
    - Loads .env into os.environ (does NOT overwrite already-set env vars).
    - Sets random.seed and numpy.random.seed if numpy is installed.
    """
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    env_p = Path(env_path) if env_path else DEFAULT_ENV_PATH

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    # Load .env (override=False means existing OS env vars win)
    if env_p.exists():
        load_dotenv(env_p, override=False)

    # Parse YAML
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Apply environment variable overrides (secrets must NEVER be in YAML)
    _apply_env_overrides(cfg)

    # Set global random seed for reproducibility
    seed = cfg.get("project", {}).get("random_seed", 42)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        # numpy not installed yet (Stage 2 installs it, but be defensive)
        pass

    # Stash paths that other modules need
    cfg["_paths"] = {
        "project_root": str(PROJECT_ROOT),
        "config_file": str(cfg_path),
        "env_file": str(env_p) if env_p.exists() else None,
    }

    return cfg


def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """
    Apply environment variable overrides for select fields.

    Mapping (env var -> config path):
        USE_MOCK_LLM          -> llm.use_mock
        OPENAI_API_KEY        -> llm.real.api_key (stored as _api_key, never logged)
        DETECTOR_MODEL_PATH   -> detection.bert.model_path
        FORCE_CPU             -> detection.bert.train_on_cpu
        API_HOST              -> api.host
        API_PORT              -> api.port
        LOG_LEVEL             -> logging.level
        RANDOM_SEED           -> project.random_seed
    """
    def _to_bool(v: str | None) -> bool | None:
        if v is None:
            return None
        return v.strip().lower() in ("1", "true", "yes", "on")

    def _to_int(v: str | None) -> int | None:
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return None

    if "USE_MOCK_LLM" in os.environ:
        cfg.setdefault("llm", {})["use_mock"] = _to_bool(os.environ["USE_MOCK_LLM"])

    if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
        cfg.setdefault("llm", {}).setdefault("real", {})["_api_key"] = os.environ["OPENAI_API_KEY"]

    if "DETECTOR_MODEL_PATH" in os.environ and os.environ["DETECTOR_MODEL_PATH"]:
        cfg.setdefault("detection", {}).setdefault("bert", {})["model_path"] = os.environ["DETECTOR_MODEL_PATH"]

    if "FORCE_CPU" in os.environ:
        cfg.setdefault("detection", {}).setdefault("bert", {})["train_on_cpu"] = _to_bool(os.environ["FORCE_CPU"])

    if "API_HOST" in os.environ:
        cfg.setdefault("api", {})["host"] = os.environ["API_HOST"]

    if "API_PORT" in os.environ:
        cfg.setdefault("api", {})["port"] = _to_int(os.environ["API_PORT"]) or cfg.get("api", {}).get("port", 8000)

    if "LOG_LEVEL" in os.environ:
        cfg.setdefault("logging", {})["level"] = os.environ["LOG_LEVEL"]

    if "RANDOM_SEED" in os.environ:
        cfg.setdefault("project", {})["random_seed"] = _to_int(os.environ["RANDOM_SEED"]) or 42


if __name__ == "__main__":
    # Quick smoke test: run `python -m src.utils.config` from project root.
    cfg = load_config()
    print(f"Project: {cfg['project']['name']} v{cfg['project']['version']}")
    print(f"Default detector: {cfg['detection']['default_detector']}")
    print(f"Random seed: {cfg['project']['random_seed']}")
    print(f"Config file: {cfg['_paths']['config_file']}")
    print(f"Env file: {cfg['_paths']['env_file']}")
