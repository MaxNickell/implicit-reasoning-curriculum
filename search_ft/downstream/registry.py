BENCHMARKS = {}


def register(name):
    """Decorator to register a benchmark function.

    The function must have signature:
        def eval_xxx(model, tokenizer, use_chat, n=None) -> dict

    Returns a dict with at minimum {"accuracy": float} and optionally
    any benchmark-specific metrics.
    """
    def _wrap(fn):
        BENCHMARKS[name] = fn
        return fn
    return _wrap


def list_benchmarks():
    return sorted(BENCHMARKS.keys())
