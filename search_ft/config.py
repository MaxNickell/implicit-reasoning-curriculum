import argparse
import os
import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


RESULTS_ENV_VAR = "GSFT_RESULTS_DIR"
RESULTS_DEFAULT = "./results"


def get_results_dir() -> str:
    return os.environ.get(RESULTS_ENV_VAR, RESULTS_DEFAULT)


@dataclass
class PathsConfig:
    base_dir: str = ""

    def __post_init__(self):
        if not self.base_dir:
            self.base_dir = get_results_dir()

    @property
    def data_dir(self) -> str:
        return os.path.join(self.base_dir, "data")

    @property
    def downstream_data_dir(self) -> str:
        return os.path.join(self.base_dir, "data", "downstream")

    @property
    def runs_dir(self) -> str:
        return os.path.join(self.base_dir, "runs")

    def run_dir(self, name: str) -> str:
        return os.path.join(self.base_dir, "runs", name)

    def heldout_path(self, max_lookahead: int) -> str:
        return os.path.join(self.base_dir, "data", f"heldout_L{max_lookahead}.json")


@dataclass
class ExperimentConfig:
    name: str = ""
    seed: int = 1234
    deterministic: bool = False


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    reinit_weights: bool = False
    use_liger: bool = True


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5
    effective_batch_size: int = 768
    per_device_batch_size: int = 192
    warmup_steps: int = 100
    max_steps: int = 100_000_000
    save_steps: int = 500
    accuracy_window: int = 1000


@dataclass
class CurriculumConfig:
    base_lookahead: int = 1
    max_lookahead: int = 96
    lookahead_step: int = 8
    accuracy_threshold: float = 0.98
    min_steps_per_stage: int = 200
    check_every: int = 50


@dataclass
class DataConfig:
    fixed_vocab: bool = False
    instruction_mix_ratio: float = 0.0


@dataclass
class EvalConfig:
    samples: int = 1000
    stage_eval: bool = True
    stage_eval_every: int = 8
    eval_every_steps: int = 0
    print_examples: int = 0
    persist_every: int = 2000
    save_stage_checkpoints: bool = True


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# --- YAML Loading with Inheritance ---


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    base_ref = data.pop("_base_", None)
    if base_ref is not None:
        base_path = path.parent / base_ref
        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path} (referenced from {path})")
        base_data = _load_yaml(base_path)
        data = _deep_merge(base_data, data)

    return data


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# --- CLI Override Parsing ---


def _parse_overrides(argv: list[str]) -> tuple[Path | None, list[tuple[str, str]]]:
    config_path = None
    overrides = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config requires a path argument")
            config_path = Path(argv[i + 1])
            i += 2
        elif arg.startswith("--config="):
            config_path = Path(arg.split("=", 1)[1])
            i += 1
        elif arg.startswith("--") and "." in arg.split("=")[0]:
            if "=" in arg:
                key, value = arg[2:].split("=", 1)
            else:
                if i + 1 >= len(argv):
                    raise ValueError(f"{arg} requires a value")
                key = arg[2:]
                value = argv[i + 1]
                i += 1
            overrides.append((key, value))
            i += 1
        else:
            i += 1

    return config_path, overrides


def _apply_overrides(data: dict, overrides: list[tuple[str, str]]) -> dict:
    for key, value in overrides:
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = _coerce_value(value)
    return data


def _coerce_value(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none" or value.lower() == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# --- Config Construction ---


def _build_config(data: dict) -> Config:
    sections = {}
    section_classes = {
        "paths": PathsConfig,
        "experiment": ExperimentConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "curriculum": CurriculumConfig,
        "data": DataConfig,
        "eval": EvalConfig,
    }

    for section_name, cls in section_classes.items():
        section_data = data.get(section_name, {})
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in section_data.items() if k in valid_fields}
        sections[section_name] = cls(**filtered)

    return Config(**sections)


# --- Public API ---


def load_config(argv: Optional[list[str]] = None) -> Config:
    if argv is None:
        argv = sys.argv[1:]

    config_path, overrides = _parse_overrides(argv)

    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        data = _load_yaml(config_path)
    else:
        data = {}

    if overrides:
        data = _apply_overrides(data, overrides)

    return _build_config(data)


def config_to_dict(config: Config) -> dict:
    return asdict(config)
