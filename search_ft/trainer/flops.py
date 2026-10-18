"""Compute accounting.

The cumulative compute of a run is 6N x tokens (Kaplan et al.), with N every parameter of the model
(tied embeddings counted once) and tokens every packed token on every rank. With the sparse output head
(training.head: sparse) the lm_head only runs on the labelled rows, so the FLOPs actually executed in a
step are 6(N - N_head) x tokens + 6 N_head x head_rows; that count feeds the achieved_tflops throughput
diagnostic only (executed_flops), the cumulative accounting stays 6N x tokens.
"""
from ..model.arch import resolve_model_parts


def estimate_flops_per_token(model) -> int:
    """6N: six FLOPs per parameter per token, every parameter."""
    total_params = sum(p.numel() for p in model.parameters())
    return 6 * total_params


def estimate_head_params(model) -> int:
    """N_head: parameter count of the output head the sparse path runs (lm_head for Qwen, where it
    shares the embedding weight and so enters N once; embed_out for Pythia, a separate weight).
    0 when the head cannot be resolved (achieved_tflops then falls back to 6N x tokens)."""
    try:
        return int(sum(p.numel() for p in resolve_model_parts(model).lm_head.parameters()))
    except Exception:
        return 0


def executed_flops(tokens: int, head_rows: int, flops_per_token: int, flops_per_head_row, head: str) -> float:
    """FLOPs actually executed for `tokens` packed tokens of which `head_rows` went through the lm_head.
    head='full' (or N_head unknown): 6N x tokens. head='sparse': 6N x tokens - 6 N_head x (tokens - head_rows),
    clamped so it never exceeds 6N x tokens."""
    flops = tokens * flops_per_token
    if head == "sparse" and flops_per_head_row:
        flops -= flops_per_head_row * max(0, tokens - head_rows)
    return flops
