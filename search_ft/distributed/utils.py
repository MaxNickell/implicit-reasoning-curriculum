import os
import random

import numpy as np
import torch


def dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if dist_is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if dist_is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    return get_rank() == 0


def rank_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def barrier():
    if dist_is_initialized():
        torch.distributed.barrier()


def broadcast_object(obj, src: int = 0):
    if not dist_is_initialized():
        return obj
    obj_list = [obj]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def set_all_seeds(seed: int, deterministic: bool = False):
    rank = get_rank()
    true_seed = seed + rank * 9973
    rank_print(f"[SEED] Setting all random seeds to {true_seed}")

    random.seed(true_seed)
    np.random.seed(true_seed)
    torch.manual_seed(true_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(true_seed)
        torch.cuda.manual_seed_all(true_seed)

    if deterministic:
        # Bitwise-reproducible training on the same hardware and GPU count. Three things must hold, and all of
        # them must be in place before the first CUDA kernel runs (this is called before the model is loaded):
        #   FLASH_ATTENTION_DETERMINISTIC=1: transformers reads it on every flash-attention call and selects the
        #     deterministic backward (FlashAttention-2 and -3); without it the attention backward is the one
        #     nondeterministic op on this training path and runs with the same seed diverge after a few steps;
        #   CUBLAS_WORKSPACE_CONFIG=:4096:8: deterministic cuBLAS reductions (required by PyTorch for
        #     use_deterministic_algorithms on CUDA >= 10.2);
        #   torch.use_deterministic_algorithms with warn_only=True: deterministic kernels wherever PyTorch has one,
        #     a warning (not an error) for the few ops that do not.
        # Cost: about 6-7% step time at L=128 on H100 in our measurements. The paper's runs did not use it.
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        rank_print("[SEED] Deterministic mode: FLASH_ATTENTION_DETERMINISTIC=1, CUBLAS_WORKSPACE_CONFIG=:4096:8, "
                   "cudnn deterministic, torch deterministic algorithms (warn_only)")

    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(true_seed)
    except Exception:
        pass
