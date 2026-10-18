"""Config checks: configs/base.yaml loads with the training-semantics fields, defaults match the paper's setting, and
invalid values are rejected.
Run from the repository root:  python tests/test_config.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from search_ft.config import load_config, Config, ModelConfig, TrainingConfig, config_to_dict


def test_base_yaml():
    cfg = load_config(["--config", os.path.join(ROOT, "configs", "base.yaml")])
    assert cfg.model.attn_kernel == "auto"
    assert cfg.model.ckpt_every_n_layers == 2
    assert cfg.model.ckpt_relax_max_tokens == 88000
    assert cfg.model.gradient_checkpointing is True
    assert cfg.training.head == "sparse"
    assert cfg.training.ddp_grad_average is False
    assert cfg.training.accuracy_window == 800
    assert cfg.training.learning_rate == 5e-5
    assert not hasattr(cfg.training, "warmup_steps")
    d = config_to_dict(cfg)
    assert "warmup_steps" not in d["training"]
    assert {"attn_kernel", "ckpt_every_n_layers", "ckpt_relax_max_tokens"} <= set(d["model"])
    assert {"head", "ddp_grad_average", "accuracy_window"} <= set(d["training"])
    print("test_base_yaml: OK")


def test_example_yaml_inherits():
    cfg = load_config(["--config", os.path.join(ROOT, "configs", "example.yaml")])
    assert cfg.experiment.name == "qwen06b_curriculum"
    assert cfg.training.per_device_batch_size == 96
    assert cfg.training.accuracy_window == 800 and cfg.training.head == "sparse"
    assert cfg.model.attn_kernel == "auto" and cfg.model.ckpt_every_n_layers == 2
    print("test_example_yaml_inherits: OK")


def test_defaults_and_overrides():
    cfg = Config()
    assert cfg.training.accuracy_window == 800
    assert cfg.training.head == "sparse" and cfg.training.ddp_grad_average is False
    assert cfg.model.attn_kernel == "auto" and cfg.model.ckpt_every_n_layers == 2 \
        and cfg.model.ckpt_relax_max_tokens == 88000
    cfg = load_config(["--model.attn_kernel=fa3", "--training.head=full", "--training.ddp_grad_average=true",
                       "--model.ckpt_every_n_layers=1", "--model.ckpt_relax_max_tokens=0"])
    assert cfg.model.attn_kernel == "fa3" and cfg.training.head == "full" and cfg.training.ddp_grad_average is True
    assert cfg.model.ckpt_every_n_layers == 1 and cfg.model.ckpt_relax_max_tokens == 0
    for bad in (lambda: ModelConfig(attn_kernel="sdpa"), lambda: ModelConfig(ckpt_every_n_layers=0),
                lambda: ModelConfig(ckpt_relax_max_tokens=-1), lambda: TrainingConfig(head="dense")):
        try:
            bad()
            raise AssertionError("invalid config value must raise")
        except ValueError:
            pass
    print("test_defaults_and_overrides: OK")


def main():
    test_base_yaml()
    test_example_yaml_inherits()
    test_defaults_and_overrides()
    print("test_config: ALL OK")


if __name__ == "__main__":
    main()
