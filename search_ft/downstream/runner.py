import argparse
import json
import os
import sys
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..config import PathsConfig
from .registry import BENCHMARKS, list_benchmarks
from . import benchmarks as _  # trigger registration


def load_model(checkpoint_path: str):
    print(f"Loading model: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    model.cuda()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e6:.1f}M params loaded")
    return model, tokenizer


def determine_use_chat(tokenizer, chat_mode: str = "auto") -> bool:
    if chat_mode == "always":
        return True
    if chat_mode == "never":
        return False
    has_chat = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    print(f"  Chat template: {'yes' if has_chat else 'no'} (auto-detected)")
    return has_chat


def _infer_run_name(checkpoint_path: str) -> str:
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on downstream benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint or HF model name")
    parser.add_argument("--benchmarks", nargs="+", required=True,
                        help=f"Benchmarks to run. Available: {list_benchmarks()}")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (for output path). Inferred from checkpoint path if omitted.")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of examples per benchmark (None = full test set)")
    parser.add_argument("--chat-template", choices=["auto", "always", "never"], default="auto",
                        help="Chat template usage (auto detects from tokenizer)")
    args = parser.parse_args()

    paths = PathsConfig()

    # Set downstream data directory for all benchmarks
    os.environ["DOWNSTREAM_DATA_DIR"] = paths.downstream_data_dir

    # Resolve output path
    run_name = args.run_name or _infer_run_name(args.checkpoint)
    if run_name:
        output_path = os.path.join(paths.run_dir(run_name), "eval", "downstream.json")
    else:
        print("ERROR: Cannot determine run name. Provide --run_name or use a checkpoint "
              "path under .../runs/{name}/...")
        sys.exit(1)

    # Validate benchmark names
    for bm in args.benchmarks:
        if bm not in BENCHMARKS:
            print(f"ERROR: Unknown benchmark '{bm}'. Available: {list_benchmarks()}")
            sys.exit(1)

    # Load model
    model, tokenizer = load_model(args.checkpoint)
    use_chat = determine_use_chat(tokenizer, args.chat_template)

    # Run benchmarks
    results = {
        "checkpoint": args.checkpoint,
        "data_dir": paths.downstream_data_dir,
        "benchmarks": {},
    }

    for bm_name in args.benchmarks:
        print(f"\n{'='*60}")
        print(f"Running benchmark: {bm_name}")
        print(f"{'='*60}")

        try:
            bm_fn = BENCHMARKS[bm_name]
            bm_result = bm_fn(model, tokenizer, use_chat, n=args.n)
            results["benchmarks"][bm_name] = bm_result
            print(f"  Result: {bm_result}")
        except NotImplementedError as e:
            print(f"  SKIPPED: {e}")
            results["benchmarks"][bm_name] = {"error": str(e)}
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results["benchmarks"][bm_name] = {"error": str(e)}

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
