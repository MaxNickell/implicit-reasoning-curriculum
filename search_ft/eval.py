import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import PathsConfig
from .data.heldout import load_heldout, eval_data_fingerprint
from .distributed.utils import rank_print, is_main_process, barrier, get_world_size
from .evaluation.greedy import run_eval_greedy
from .evaluation.tf_loss import run_eval_tf_loss


def load_checkpoint_model(checkpoint_path: str):
    rank_print(f"[EVAL] Loading from: {checkpoint_path}")

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
    rank_print(f"[EVAL] Loaded: {n_params/1e6:.1f}M params")

    return model, tokenizer


def _infer_run_name(checkpoint_path: str) -> str:
    parts = os.path.normpath(checkpoint_path).split(os.sep)
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on synthetic search task")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory or HF model name")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name (for output path). Inferred from checkpoint path if omitted.")
    parser.add_argument("--max_lookahead", type=int, default=96,
                        help="Max lookahead (to find heldout file, default: 96)")
    parser.add_argument("--print_mistakes", type=int, default=0,
                        help="Number of mistakes to print")
    args = parser.parse_args()

    paths = PathsConfig()
    heldout_path = paths.heldout_path(args.max_lookahead)

    if not os.path.isfile(heldout_path):
        rank_print(f"[EVAL] ERROR: Heldout file not found: {heldout_path}")
        rank_print(f"       Run: python -m search_ft.generate_heldout --max_lookahead {args.max_lookahead}")
        sys.exit(1)

    # Resolve run name and output path
    run_name = args.run_name or _infer_run_name(args.checkpoint)
    output_path = None
    if run_name:
        output_path = os.path.join(paths.run_dir(run_name), "eval", "synthetic.json")

    # Load model
    model, tokenizer = load_checkpoint_model(args.checkpoint)

    # Load eval data
    eval_inputs, eval_labels = None, None
    if is_main_process():
        heldout_examples, _ = load_heldout(heldout_path)
        eval_inputs = [ex["input_text"] for ex in heldout_examples]
        eval_labels = [ex["output_texts"] for ex in heldout_examples]
        fp = eval_data_fingerprint(eval_inputs, eval_labels)
        rank_print(f"[EVAL] Loaded {len(eval_inputs)} examples, fingerprint={fp}")

    barrier()

    if get_world_size() > 1:
        from .distributed.utils import broadcast_object
        eval_inputs = broadcast_object(eval_inputs, src=0)
        eval_labels = broadcast_object(eval_labels, src=0)

    if not eval_inputs:
        rank_print("[EVAL] ERROR: No eval data!")
        sys.exit(1)

    # Determine if chat template was used during training
    use_chat_template = False
    run_name = args.run_name or _infer_run_name(args.checkpoint)
    if run_name:
        import yaml
        run_config_path = os.path.join(paths.run_dir(run_name), "config.yaml")
        if os.path.isfile(run_config_path):
            with open(run_config_path) as f:
                run_cfg = yaml.safe_load(f) or {}
            mix_ratio = run_cfg.get("data", {}).get("instruction_mix_ratio", 0.0)
            use_chat_template = mix_ratio > 0

    # Run evaluation
    rank_print(f"\n[EVAL] Evaluating on {len(eval_inputs)} heldout samples...")
    if use_chat_template:
        rank_print("[EVAL] Using chat template (instruction mixing was enabled during training)")

    with torch.no_grad():
        tf_loss = run_eval_tf_loss(model, tokenizer, eval_inputs, eval_labels,
                                   seed=0, use_chat_template=use_chat_template)
        greedy_result = run_eval_greedy(model, tokenizer, eval_inputs, eval_labels,
                                        print_examples=args.print_mistakes,
                                        use_chat_template=use_chat_template)

    results = {
        "checkpoint": args.checkpoint,
        "heldout_path": heldout_path,
        "max_lookahead": args.max_lookahead,
        "eval_samples": len(eval_inputs),
        "first_token_acc": greedy_result["first_token_acc"],
        "full_word_acc": greedy_result["full_word_acc"],
        "tf_loss": tf_loss,
        "total": greedy_result["total"],
    }

    if is_main_process():
        rank_print(f"\n{'='*60}")
        rank_print(f"[RESULTS] Checkpoint: {args.checkpoint}")
        rank_print(f"[RESULTS] Samples: {results['total']}")
        rank_print(f"[RESULTS] First Token Acc: {results['first_token_acc']:.2%}")
        rank_print(f"[RESULTS] Full Word Acc:   {results['full_word_acc']:.2%}")
        rank_print(f"[RESULTS] TF Loss:         {results['tf_loss']:.4f}")
        rank_print(f"{'='*60}\n")

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            rank_print(f"[EVAL] Results saved to {output_path}")
        else:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
