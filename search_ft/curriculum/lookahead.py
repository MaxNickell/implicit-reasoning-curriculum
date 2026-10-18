from typing import List, Optional, Tuple, Union


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
                        reserve_edges: int = 1,
                        breadth: int = 2) -> float:
    if L_target <= 0:
        return 0.0

    edges_unscaled = max(1, (n - fixed_tokens) // tokens_per_edge)

    # From effective_search_L:
    # L_edges = (alpha * edges_unscaled - reserve_edges) // 2
    # To get L_target, we need: (alpha * edges_unscaled - reserve_edges) >= 2 * L_target
    # alpha >= (2 * L_target + reserve_edges) / edges_unscaled

    # breadth=2 (default) is the minimum budget for a depth-L search: one goal
    # path plus one equal-length distractor. Larger values widen the deepest
    # instances to `breadth` branches (and scale the vertex-ID pool with them),
    # at the cost of a lower reachable L for a given context size.
    needed_alpha = (breadth * L_target + reserve_edges) / edges_unscaled

    return min(max(needed_alpha, 0.0), 1.0)


MixtureSpec = Union[str, List[Union[List[int], Tuple[int, int]]], None]


def parse_lookahead_mixture(spec: MixtureSpec) -> Optional[List[Tuple[int, int]]]:
    """Parse curriculum.shuffled_mixture into [(lookahead, weight_in_steps), ...].

    Accepts the string form 'L:steps,L:steps,...' (the form a curriculum run's per-stage step counts are
    quoted in) or a YAML list of [L, steps] pairs. Returns None when the spec is empty (mixture off)."""
    if not spec:
        return None
    out = []
    if isinstance(spec, str):
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            L, n = part.split(":")
            out.append((int(L), int(n)))
    else:
        for pair in spec:
            if len(pair) != 2:
                raise ValueError(f"shuffled_mixture entries must be [L, steps] pairs, got {pair!r}")
            out.append((int(pair[0]), int(pair[1])))
    for L, n in out:
        if L <= 0 or n < 0:
            raise ValueError(f"shuffled_mixture needs L > 0 and a non-negative weight, got {L}:{n}")
    if not out:
        return None
    return out
