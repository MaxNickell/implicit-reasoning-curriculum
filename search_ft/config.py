import argparse
import os
import sys
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Optional

import yaml

from .curriculum.lookahead import parse_lookahead_mixture


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


ATTN_KERNELS = ("auto", "fa2", "fa3")
HEAD_MODES = ("sparse", "full")


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    # Flash-attention family used for training (transformers' packed varlen path). auto = FA3 on Hopper (sm90),
    # FA2 elsewhere; fa2/fa3 force one. In-run evaluations always run under SDPA (see model/attention.py).
    attn_kernel: str = "auto"
    gradient_checkpointing: bool = True
    # Selective gradient checkpointing: recompute only every N-th decoder layer (1 = every layer) ...
    ckpt_every_n_layers: int = 2
    # ... unless a micro-batch exceeds this many tokens, in which case every layer is recomputed (0 = never relax).
    # 88,000 was measured for Qwen3-0.6B at per_device_batch_size 48 on 80 GB GPUs; other models or batch sizes
    # need their own measurement.
    ckpt_relax_max_tokens: int = 88000
    reinit_weights: bool = False
    use_liger: bool = True

    def __post_init__(self):
        if self.attn_kernel not in ATTN_KERNELS:
            raise ValueError(f"model.attn_kernel must be one of {ATTN_KERNELS}, got {self.attn_kernel!r}")
        if self.ckpt_every_n_layers < 1:
            raise ValueError(f"model.ckpt_every_n_layers must be >= 1, got {self.ckpt_every_n_layers}")
        if self.ckpt_relax_max_tokens < 0:
            raise ValueError(f"model.ckpt_relax_max_tokens must be >= 0, got {self.ckpt_relax_max_tokens}")


@dataclass
class TrainingConfig:
    # Constant learning rate for the whole run (no warmup, no schedule).
    learning_rate: float = 5e-5
    effective_batch_size: int = 768
    per_device_batch_size: int = 192
    max_steps: int = 100_000_000
    save_steps: int = 500
    # Rolling window (sequences, total across ranks) for the curriculum's full-word accuracy gate.
    accuracy_window: int = 800
    # Output head: sparse = lm_head, argmax and cross-entropy on the labelled rows only; full = every packed row
    # (identical loss up to GEMM rounding, see trainer/forward.py).
    head: str = "sparse"
    # False (the paper's setting): each rank trains its own replica, gradients are never averaged across ranks.
    # True: all-reduce (AVG) every trainable gradient after backward, i.e. standard DistributedDataParallel.
    ddp_grad_average: bool = False
    # Compute budget in PFLOPs (6N x cumulative tokens, the accounting of curriculum/callback.py). Checked when
    # the Trainer saves a checkpoint (every save_steps); training stops at the first save where the cumulative
    # compute has reached it. 0 = no budget. The cumulative token count carries over a resume.
    max_total_pflops: float = 0.0

    def __post_init__(self):
        if self.head not in HEAD_MODES:
            raise ValueError(f"training.head must be one of {HEAD_MODES}, got {self.head!r}")
        if self.max_total_pflops < 0:
            raise ValueError(f"training.max_total_pflops must be >= 0, got {self.max_total_pflops}")


@dataclass
class CurriculumConfig:
    base_lookahead: int = 1
    max_lookahead: int = 96
    lookahead_step: int = 8
    accuracy_threshold: float = 0.98
    min_steps_per_stage: int = 200
    check_every: int = 50
    # Edge-budget multiplier per unit lookahead. 2 (the paper's setting) = one goal path + one equal-length
    # distractor at max depth; 3/4/5 widen the deepest instances to that many branches and scale the vertex-ID
    # pool to breadth*L+2. Caps the reachable L at ((n-5)//3-1)//breadth for context size n.
    breadth: int = 2
    # Exposure-matched shuffled control: 'L:steps,L:steps,...' (or a list of [L, steps] pairs) taken from a
    # curriculum run's per-stage step counts. Every search example then draws its lookahead L at random,
    # weighted by the steps the curriculum spent at L, so the run sees the curriculum's multiset in random
    # order. Combine with an accuracy_threshold above 1 so the stage never advances. null = off.
    shuffled_mixture: Any = None

    def __post_init__(self):
        if self.breadth < 1:
            raise ValueError(f"curriculum.breadth must be >= 1, got {self.breadth}")
        parse_lookahead_mixture(self.shuffled_mixture)  # raises on a malformed spec


VOCAB_POOLS = ("none", "grow", "fixed")


@dataclass
class DataConfig:
    # Entity-vocabulary curriculum: map each symbolic vertex ID to one fixed attribute name. grow = the ID
    # range follows the stage (breadth*L+2 names at stage L, i.e. 2(L+1) at breadth 2; ID 0 is reserved);
    # fixed = the ID range is pinned at the context maximum ((n-5)//3+1 names) throughout; none = fresh
    # random names per instance (the paper's main runs).
    vocab_pool: str = "none"
    # Context budget of the symbolic generator in generator tokens (n). 0 = 6 x curriculum.max_lookahead,
    # the paper's setting at breadth 2; a breadth-b run that is to reach L needs (n-5)//3 >= b*L+1.
    max_input_size: int = 0
    instruction_mix_ratio: float = 0.0

    def __post_init__(self):
        if self.vocab_pool is None:  # a CLI override `--data.vocab_pool none` arrives as None (_coerce_value)
            self.vocab_pool = "none"
        if self.vocab_pool not in VOCAB_POOLS:
            raise ValueError(f"data.vocab_pool must be one of {VOCAB_POOLS}, got {self.vocab_pool!r}")
        if self.max_input_size < 0:
            raise ValueError(f"data.max_input_size must be >= 0, got {self.max_input_size}")


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
