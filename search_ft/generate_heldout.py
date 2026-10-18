import argparse

from .config import PathsConfig, VOCAB_POOLS
from .data.heldout import build_heldout_set, save_heldout, eval_data_fingerprint


def main():
    parser = argparse.ArgumentParser(
        description="Generate a held-out evaluation set for the synthetic search task"
    )
    parser.add_argument("--max_lookahead", type=int, required=True,
                        help="Maximum lookahead (determines problem complexity)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of held-out examples to generate")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_input_size", type=int, default=0,
                        help="Generator context size n (0 = 6 x max_lookahead); must equal the training run's "
                             "data.max_input_size (0 there as well = 6 x max_lookahead)")
    parser.add_argument("--vocab_pool", type=str, default="none", choices=list(VOCAB_POOLS),
                        help="Must equal the training run's data.vocab_pool (grow and fixed render the "
                             "held-out set through the same fixed ID -> name dictionary as training)")
    args = parser.parse_args()

    paths = PathsConfig()
    max_input_size = args.max_input_size or 6 * args.max_lookahead
    output_path = paths.heldout_path(args.max_lookahead)

    print(f"Generating held-out evaluation set:")
    print(f"  GSFT_RESULTS_DIR = {paths.base_dir}")
    print(f"  max_lookahead    = {args.max_lookahead}")
    print(f"  max_input_size   = {max_input_size}")
    print(f"  vocab_pool       = {args.vocab_pool}")
    print(f"  samples          = {args.samples}")
    print(f"  seed             = {args.seed}")
    print(f"  output           = {output_path}")
    print()

    examples, _ = build_heldout_set(
        size=args.samples,
        max_input_size=max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed,
        vocab_pool=args.vocab_pool,
    )

    inputs = [ex["input_text"] for ex in examples]
    labels = [ex["output_texts"] for ex in examples]
    fingerprint = eval_data_fingerprint(inputs, labels)

    save_heldout(
        output_path,
        examples,
        max_lookahead=args.max_lookahead,
        max_input_size=max_input_size,
        seed=args.seed,
        vocab_pool=args.vocab_pool,
    )

    print(f"Saved {len(examples)} examples to {output_path}")
    print(f"Fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
