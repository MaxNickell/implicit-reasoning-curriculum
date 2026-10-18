"""Deterministic generators for the two depth-extended benchmarks (PrOntoQA-Deep and
Web-of-Lies-Deep). Each is run on first use by its benchmark module and writes a fixed
4,000-item file (8 depths x 500 items) into ``$DOWNSTREAM_DATA_DIR``; both can also be run
as ``python -m search_ft.downstream.generators.<name>``."""
