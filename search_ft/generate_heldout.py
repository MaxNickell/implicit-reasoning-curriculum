import argparse

from .config import PathsConfig
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
    args = parser.parse_args()

    paths = PathsConfig()
    max_input_size = 6 * args.max_lookahead
    output_path = paths.heldout_path(args.max_lookahead)

    print(f"Generating held-out evaluation set:")
    print(f"  GSFT_RESULTS_DIR = {paths.base_dir}")
    print(f"  max_lookahead    = {args.max_lookahead}")
    print(f"  max_input_size   = {max_input_size}")
    print(f"  samples          = {args.samples}")
    print(f"  seed             = {args.seed}")
    print(f"  output           = {output_path}")
    print()

    examples, _ = build_heldout_set(
        size=args.samples,
        max_input_size=max_input_size,
        max_lookahead=args.max_lookahead,
        seed=args.seed,
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
    )

    print(f"Saved {len(examples)} examples to {output_path}")
    print(f"Fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()
