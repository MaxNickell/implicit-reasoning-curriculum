# Unlocking Long-Horizon Implicit Reasoning in LLMs via Curriculum Learning

Curriculum finetuning on synthetic search problems with evaluation on downstream reasoning benchmarks.

```bibtex
@article{huang2026unlocking,
  title={Unlocking Long-Horizon Implicit Reasoning in LLMs via Curriculum Learning},
  author={Huang, Jimson and Nickell, Maxwell and Zhang, Ruowang and Saparov, Abulhair},
  year={2026},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

## Setup

```bash
conda env create -f environment.yml
conda activate search-ft
python setup.py build_ext --inplace
```

```bash
export GSFT_RESULTS_DIR=/path/to/results
export HF_HOME=/path/to/hf_cache
```

## Usage

**1. Generate held-out evaluation set** (run once per `max_lookahead`):
```bash
python -m search_ft.generate_heldout --max_lookahead 96
```

The held-out set is excluded from training data and used for all evaluations.

**2. Train:**
```bash
torchrun --nproc_per_node=4 -m search_ft.train --config configs/example.yaml
```

Training automatically resumes from the latest checkpoint if the run directory already exists. The effective batch size is held constant regardless of GPU count.

**3. Evaluate on synthetic task:**
```bash
python -m search_ft.eval --checkpoint $GSFT_RESULTS_DIR/runs/<name>/checkpoint-N/
```

**4. Evaluate on downstream benchmarks:**
```bash
python -m search_ft.downstream.runner \
    --checkpoint $GSFT_RESULTS_DIR/runs/<name>/checkpoint-N/ \
    --benchmarks proofwriter folio logiqa ruletaker clutrr stepgame \
                 prontoqa_ood multilogieval zebra_mc logicbench \
                 nlgraph_gen proofwriter_cwa \
    --n 1000
```

Benchmark datasets are downloaded automatically on first use.

## Configuration

Configs use YAML with `_base_` inheritance. See `configs/example.yaml` for a complete example. Override any parameter via CLI:
```bash
torchrun ... --config configs/example.yaml --curriculum.lookahead_step=1 --data.instruction_mix_ratio=0.06
```

All parameters are defined in `search_ft/config.py`.

## Output Structure

```
$GSFT_RESULTS_DIR/
├── data/
│   ├── heldout_L96.json
│   └── downstream/
└── runs/<experiment_name>/
    ├── config.yaml
    ├── loss_history.jsonl
    ├── checkpoint-N/
    └── eval/
        ├── synthetic.json
        └── downstream.json
```