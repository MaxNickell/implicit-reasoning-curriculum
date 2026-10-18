BENCHMARKS = {}
PREPARERS = {}


def register(name):
    """Decorator to register a benchmark function.

    The function must have signature:
        def eval_xxx(model, tokenizer, use_chat, n=None) -> dict

    Returns a dict with at minimum {"accuracy": float} (for stratified benchmarks under
    the key "overall") and any benchmark-specific metrics.
    """
    def _wrap(fn):
        BENCHMARKS[name] = fn
        return fn
    return _wrap


def register_prepare(name):
    """Decorator for a benchmark's model-free data preparation function,
        def prepare(n=None) -> dict
    which downloads or builds the data and returns the scored strata (used by
    ``python -m search_ft.downstream.prepare_data`` to fetch everything up front)."""
    def _wrap(fn):
        PREPARERS[name] = fn
        return fn
    return _wrap


def list_benchmarks():
    return sorted(BENCHMARKS.keys())
