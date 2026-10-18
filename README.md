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
    --benchmarks proofwriter proofwriter_cwa pararules zebra_mc prontoqa_deep \
                 folio justlogic proverqa clutrr logicbench_bqa ruletaker \
                 web_of_lies_deep logiqa nlgraph
```

Benchmark datasets are downloaded automatically on first use.

The fourteen benchmarks above are those of the paper's Table 4, each at the sample size and few-shot setting of that table when `--n` is omitted. The scoring flags default to the paper's final protocol (`--answer-cue assistant --label-variant auto --length-norm off`), `--items-output` writes the per-item outcomes that `python -m search_ft.downstream.mcnemar` pairs for the significance tests, and `python -m search_ft.downstream.prepare_data` fetches all data up front. See [docs/DOWNSTREAM.md](docs/DOWNSTREAM.md) for the benchmark table, the protocol, how to reproduce Table 4 and the known caveats.

## Configuration

Configs use YAML with `_base_` inheritance. See `configs/example.yaml` for a complete example. Override any parameter via CLI:
```bash
torchrun ... --config configs/example.yaml --curriculum.lookahead_step=1 --data.instruction_mix_ratio=0.06
```

All parameters are defined in `search_ft/config.py`.

## Training semantics

**Data parallelism.** Each GPU rank trains its own replica of the model. One optimizer step on a rank consumes `training.per_device_batch_size` x `gradient_accumulation_steps` packed sequences, and gradients are not averaged across ranks: the training forward calls the transformer body directly, which bypasses `DistributedDataParallel`, so the replicas drift apart between checkpoint reloads. The checkpoint that is saved, evaluated and resumed from is rank 0's replica. `training.effective_batch_size` counts the sequences consumed per optimizer step across all ranks (`per_device_batch_size` x `gradient_accumulation_steps` x `world_size`); gradient accumulation is derived from it so that this count does not depend on the GPU count. The curriculum gate uses the rolling full-word accuracy over the last `training.accuracy_window` sequences, all-gathered across ranks and averaged, so every rank advances at the same step. Compute is counted for all ranks. Setting `training.ddp_grad_average: true` all-reduces (averages) every trainable gradient across ranks at the end of each accumulation window, which is standard data-parallel training; this is not what the paper's runs used.

**Learning rate.** Constant for the whole run (`lr_scheduler_type: constant`), with no warmup and no per-stage schedule.

**Compute.** Compute is reported as 6 x N x tokens, with N every parameter of the model (tied embeddings counted once) and tokens every packed token seen by every rank. `loss_history.jsonl` carries `tokens`, `cumulative_tokens` and `cumulative_pflops` per optimizer step. With the sparse output head the FLOPs actually executed are lower; that enters only the `achieved_tflops` throughput diagnostic, not the cumulative accounting.

**Attention kernel.** Training runs the Hugging Face transformer body through transformers' packed (varlen) flash-attention path: each micro-batch is one flat sequence with per-sequence `position_ids` and int32 `cu_seq_lens`, so there is no padding and no attention across sequences. `model.attn_kernel` selects the kernel family at model load: `auto` uses FlashAttention-3 on Hopper GPUs (sm90) and FlashAttention-2 elsewhere; `fa2` and `fa3` force one. The paper's Qwen3 runs used FA2 and the Pythia runs FA3. Every in-run evaluation (teacher-forced loss and greedy decoding) switches the model to SDPA, the transformers default, and restores the training kernel afterwards; `search_ft.eval` loads checkpoints with SDPA as well.

**Output head.** `training.head: sparse` (the default) runs the `lm_head`, the argmax and the cross-entropy only on the labelled rows of the packed sequence, which is all the loss and the gate metrics read; the prompt rows, about 99% of the tokens, would carry zero gradient. `full` runs the head over every row in chunks. The two give the same loss up to GEMM rounding.

**Gradient checkpointing.** `model.ckpt_every_n_layers` recomputes only every N-th decoder layer in the backward pass; `model.ckpt_relax_max_tokens` falls back to every layer for micro-batches above that many tokens, since activation memory grows with the curriculum stage. Gradients are identical either way. The defaults (2 and 88,000) were measured for Qwen3-0.6B at `per_device_batch_size: 48` with 4 ranks on 80 GB GPUs and are not portable to other models or batch sizes without a measurement.

The kernel, head mode, checkpointing settings and library versions of a run are written to `run_meta.json` in the run directory.

## Output Structure

```
$GSFT_RESULTS_DIR/
├── data/
│   ├── heldout_L96.json
│   └── downstream/
└── runs/<experiment_name>/
    ├── config.yaml
    ├── run_meta.json
    ├── loss_history.jsonl
    ├── checkpoint-N/
    └── eval/
        ├── synthetic.json
        └── downstream.json
```