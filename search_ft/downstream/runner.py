"""Evaluate one checkpoint on the downstream suite.

    python -m search_ft.downstream.runner --checkpoint <path or HF id> \
        --benchmarks proofwriter proofwriter_cwa pararules zebra_mc prontoqa_deep folio \
                     justlogic proverqa clutrr logicbench_bqa ruletaker web_of_lies_deep \
                     logiqa nlgraph \
        [--items-output items.json]

Without --n every benchmark runs at the paper's sample size. The scoring flags default to
the paper's final protocol (--answer-cue assistant --label-variant auto --length-norm off
--cue-suffix ""); see docs/DOWNSTREAM.md.
"""
import argparse
import json
import os
import sys
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..config import PathsConfig
from . import scoring
from .registry import BENCHMARKS, list_benchmarks
from . import benchmarks as _  # noqa: F401  (registers the benchmarks)


def load_model(checkpoint_path: str, device: str = "auto", dtype: str = "auto"):
    print(f"Loading model: {checkpoint_path}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(trust_remote_code=True, torch_dtype=torch_dtype)
    if device == "cuda" and torch_dtype in (torch.bfloat16, torch.float16):
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **kwargs)
    model.eval()
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params loaded on {device} ({dtype}"
          f"{', flash_attention_2' if 'attn_implementation' in kwargs else ''})")
    return model, tokenizer


def determine_use_chat(tokenizer, chat_mode: str = "auto") -> bool:
    if chat_mode == "always":
        return True
    if chat_mode == "never":
        return False
    has_chat = getattr(tokenizer, "chat_template", None) is not None
    print(f"  Chat template: {'yes' if has_chat else 'no'} (auto-detected)")
    return has_chat


def _infer_run_name(checkpoint_path: str):
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _atomic_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the downstream benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint or HF model name")
    parser.add_argument("--benchmarks", nargs="+", required=True,
                        help=f"Benchmarks to run. Available: {list_benchmarks()}")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (output path and the alias in --items-output). "
                             "Inferred from a .../runs/<name>/... checkpoint path if omitted.")
    parser.add_argument("--n", type=int, default=None,
                        help="Examples per benchmark, spread over its strata (default: the paper's n, "
                             "see docs/DOWNSTREAM.md)")
    parser.add_argument("--chat-template", choices=["auto", "always", "never"], default="auto",
                        help="Chat template usage (auto detects from tokenizer)")
    parser.add_argument("--output", type=str, default=None,
                        help="Results JSON (default $GSFT_RESULTS_DIR/runs/<run_name>/eval/downstream.json)")
    parser.add_argument("--items-output", type=str, default=None,
                        help="Optional JSON of per-item outcomes, {run_name: {benchmark: [records]}}, "
                             "the input of search_ft.downstream.mcnemar")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Benchmark data directory (default $DOWNSTREAM_DATA_DIR, else "
                             "$GSFT_RESULTS_DIR/data/downstream)")
    # scoring protocol (defaults = the paper's final protocol)
    parser.add_argument("--label-variant", choices=list(scoring.LABEL_VARIANTS), default="auto",
                        help="Surface form of each label that is scored: 'auto' picks the bare label "
                             "at a chat turn boundary and the space-prefixed label after a cue or in "
                             "completion mode; 'space' / 'bare' force one; 'max' keeps the higher.")
    parser.add_argument("--answer-cue", choices=list(scoring.ANSWER_CUES), default="assistant",
                        help="Chat mode only. 'assistant': a trailing 'Answer:' / 'A:' / 'the answer is' "
                             "is prefilled at the start of the assistant turn. 'user': the whole prompt "
                             "stays in the user turn.")
    parser.add_argument("--length-norm", choices=list(scoring.LENGTH_NORMS), default="off",
                        help="Divide each label's summed log-prob by its UTF-8 byte length (lm-eval "
                             "acc_norm); 'auto' only for label sets of unequal token counts. 'off' is "
                             "the paper's setting.")
    parser.add_argument("--cue-suffix", default="",
                        help="Text prefilled after the answer cue in chat mode, e.g. '**' for an "
                             "untrained Qwen3 chat model that answers in markdown bold.")
    parser.add_argument("--gen-max-new-tokens", type=int, default=200,
                        help="Generation budget of the generation-scored benchmark (nlgraph)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto",
                        help="auto = bfloat16 on cuda, float32 on cpu")
    args = parser.parse_args()

    paths = PathsConfig()
    data_dir = args.data_dir or os.environ.get("DOWNSTREAM_DATA_DIR") or paths.downstream_data_dir
    os.environ["DOWNSTREAM_DATA_DIR"] = data_dir

    run_name = args.run_name or _infer_run_name(args.checkpoint)
    if args.output:
        output_path = args.output
        run_name = run_name or os.path.basename(os.path.normpath(args.checkpoint))
    elif run_name:
        output_path = os.path.join(paths.run_dir(run_name), "eval", "downstream.json")
    else:
        print("ERROR: Cannot determine run name. Provide --run_name or --output, or use a "
              "checkpoint path under .../runs/{name}/...")
        sys.exit(1)

    for bm in args.benchmarks:
        if bm not in BENCHMARKS:
            print(f"ERROR: Unknown benchmark '{bm}'. Available: {list_benchmarks()}")
            sys.exit(1)

    scoring.configure(label_variant=args.label_variant, answer_cue=args.answer_cue,
                      length_norm=args.length_norm, cue_suffix=args.cue_suffix,
                      gen_max_new_tokens=args.gen_max_new_tokens)
    print(f"[scorer] {scoring.protocol()}")
    if args.items_output:
        scoring.enable_item_log()

    model, tokenizer = load_model(args.checkpoint, args.device, args.dtype)
    use_chat = determine_use_chat(tokenizer, args.chat_template)

    results = {
        "checkpoint": args.checkpoint,
        "run_name": run_name,
        "data_dir": data_dir,
        "chat_template": use_chat,
        "protocol": scoring.protocol(),
        "n": args.n,
        "benchmarks": {},
    }
    items = {run_name: {}}

    for bm_name in args.benchmarks:
        print(f"\n{'='*60}\nRunning benchmark: {bm_name}\n{'='*60}")
        try:
            bm_result = BENCHMARKS[bm_name](model, tokenizer, use_chat, n=args.n)
            results["benchmarks"][bm_name] = bm_result
            summary = bm_result.get("overall", bm_result)
            print(f"  Result: accuracy={summary.get('accuracy', float('nan')):.4f} "
                  f"total={summary.get('total')}")
        except Exception as e:  # noqa: BLE001 - one failing benchmark must not lose the others
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results["benchmarks"][bm_name] = {"error": str(e)}
        _atomic_dump(results, output_path)
        if args.items_output:
            recs = scoring.pop_item_log(bm_name)
            items[run_name][bm_name] = recs if "error" not in results["benchmarks"][bm_name] else []
            _atomic_dump(items, args.items_output)

    print(f"\nResults saved to {output_path}")
    if args.items_output:
        print(f"Per-item outcomes saved to {args.items_output}")


if __name__ == "__main__":
    main()
