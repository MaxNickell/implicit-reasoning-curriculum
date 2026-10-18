"""Which kernel, head mode, flags and library versions a run used; written to run_meta.json so later
re-evaluations are attributable."""
import os
from typing import Dict, Any, Optional

import torch


def _dist_version(*names) -> Optional[str]:
    """Installed version of the first distribution name that resolves (liger ships as liger-kernel-nightly)."""
    import importlib.metadata as md
    for n in names:
        try:
            return md.version(n)
        except Exception:
            continue
    return None


def run_provenance(model, config) -> Dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "head": config.training.head,
        "attn_kernel": config.model.attn_kernel,
        "attn_implementation": getattr(cfg, "_attn_implementation", None) if cfg is not None else None,
        "ckpt_every_n_layers": config.model.ckpt_every_n_layers,
        "ckpt_relax_max_tokens": config.model.ckpt_relax_max_tokens,
        "ddp_grad_average": config.training.ddp_grad_average,
        "use_liger": config.model.use_liger,
        "FLASH_ATTENTION_DETERMINISTIC": os.environ.get("FLASH_ATTENTION_DETERMINISTIC"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "versions": {
            "torch": torch.__version__,
            "transformers": _dist_version("transformers"),
            "flash_attn": _dist_version("flash_attn"),
            "flash_attn_3": _dist_version("flash_attn_3", "flash-attn-3"),
            "liger_kernel": _dist_version("liger-kernel-nightly", "liger-kernel"),
        },
    }
