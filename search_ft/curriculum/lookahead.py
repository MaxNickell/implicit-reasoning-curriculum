from typing import Optional


def effective_search_L(alpha: float,
                       n: int,
                       max_lookahead_cap: Optional[int] = None,
                       tokens_per_edge: int = 3,
                       fixed_tokens: int = 4,
                       reserve_edges: int = 1) -> int:
    edges_unscaled = max(0, (n - fixed_tokens) // tokens_per_edge)
    max_edges = int(alpha * edges_unscaled)
    safe_edges = max(0, max_edges - reserve_edges)

    L_edges = safe_edges // 2
    L_tokens = max(0, (n - fixed_tokens) // (2 * tokens_per_edge))

    L_eff = min(L_edges, L_tokens)
    if max_lookahead_cap:
        L_eff = min(L_eff, int(max_lookahead_cap))
    return L_eff


def alpha_for_lookahead(L_target: int,
                        n: int,
                        tokens_per_edge: int = 3,
                        fixed_tokens: int = 4,
                        reserve_edges: int = 1) -> float:
    if L_target <= 0:
        return 0.0

    edges_unscaled = max(1, (n - fixed_tokens) // tokens_per_edge)

    # From effective_search_L:
    # L_edges = (alpha * edges_unscaled - reserve_edges) // 2
    # To get L_target, we need: (alpha * edges_unscaled - reserve_edges) >= 2 * L_target
    # alpha >= (2 * L_target + reserve_edges) / edges_unscaled

    needed_alpha = (2 * L_target + reserve_edges) / edges_unscaled

    return min(max(needed_alpha, 0.0), 1.0)
