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

### Ablations

Four fields reproduce the paper's ablations; each defaults to the main runs' setting.

`data.vocab_pool` (`none` | `grow` | `fixed`) sets how the symbolic vertex IDs are named. `none`, the main runs, draws fresh random attribute names for every instance. `grow` and `fixed` render every vertex ID through one fixed ID-to-name dictionary (the same list in every worker and rank), so the active vocabulary is exactly the ID range the generator draws from. Under `grow` that range follows the stage (`breadth*L+2` names at stage L, so 2(L+1) at breadth 2; ID 0 is reserved); under `fixed` it is pinned at the context maximum (`(n-5)//3+1` names, 255 at n=768) from the first stage on. The held-out set has to be generated with the same `--vocab_pool`, so that it is named the way the training data is; `search_ft.train` refuses a held-out file whose `vocab_pool` differs from the run's.

`curriculum.breadth` (default 2) is the edge budget per unit lookahead from which the stage alpha is derived, alpha(L) = (breadth*L+1)/((n-4)//3). Breadth 2 is one goal path plus one equal-length distractor at the maximum depth; 3 or 4 widen the deepest instances to that many branches and scale the vertex-ID pool with them. The reachable L for a context n is ((n-5)//3-1)//breadth, so a breadth above 2 needs `data.max_input_size` above the default 6 x `max_lookahead` (the paper's breadth runs used n=3080 for L=256) and a held-out set generated with the same `--max_input_size`.

`curriculum.shuffled_mixture` is the exposure-matched shuffled control. It takes the per-stage step counts of a curriculum run as `'L:steps,L:steps,...'` (a YAML list of `[L, steps]` pairs is also accepted). Every search example then draws its lookahead from that mixture, weighted by the step counts, from the per-batch-reseeded worker RNG, so the run sees the curriculum's multiset of examples in random order. Combine it with a `curriculum.accuracy_threshold` above 1, so that the stage never advances, and a `training.max_steps` equal to the curriculum's step count.

`training.max_total_pflops` (default 0, off) stops training once the cumulative compute, 6N x tokens as reported in `loss_history.jsonl`, has reached the budget. It is checked when the Trainer saves a checkpoint (every `training.save_steps`), so the stop coincides with a full checkpoint and a resume with a larger budget continues from it; the cumulative token count is restored from `loss_history.jsonl` on resume.

`configs/vocabpool_grow.yaml`, `configs/vocabpool_fixed.yaml`, `configs/breadth4.yaml` and `configs/shuffled_matched.yaml` hold the paper's settings for these arms.

## Training semantics

**Data parallelism.** Each GPU rank trains its own replica of the model. One optimizer step on a rank consumes `training.per_device_batch_size` x `gradient_accumulation_steps` packed sequences, and gradients are not averaged across ranks: the training forward calls the transformer body directly, which bypasses `DistributedDataParallel`, so the replicas drift apart between checkpoint reloads. The checkpoint that is saved, evaluated and resumed from is rank 0's replica. `training.effective_batch_size` counts the sequences consumed per optimizer step across all ranks (`per_device_batch_size` x `gradient_accumulation_steps` x `world_size`); gradient accumulation is derived from it so that this count does not depend on the GPU count. The curriculum gate uses the rolling full-word accuracy over the last `training.accuracy_window` sequences, all-gathered across ranks and averaged, so every rank advances at the same step. Compute is counted for all ranks. Setting `training.ddp_grad_average: true` all-reduces (averages) every trainable gradient across ranks at the end of each accumulation window, which is standard data-parallel training; this is not what the paper's runs used.

**Learning rate.** Constant for the whole run (`lr_scheduler_type: constant`), with no warmup and no per-stage schedule.

**Compute.** Compute is reported as 6 x N x tokens, with N every parameter of the model (tied embeddings counted once) and tokens every packed token seen by every rank. `loss_history.jsonl` carries `tokens`, `cumulative_tokens` and `cumulative_pflops` per optimizer step. With the sparse output head the FLOPs actually executed are lower; that enters only the `achieved_tflops` throughput diagnostic, not the cumulative accounting.

**Attention kernel.** Training runs the Hugging Face transformer body through transformers' packed (varlen) flash-attention path: each micro-batch is one flat sequence with per-sequence `position_ids` and int32 `cu_seq_lens`, so there is no padding and no attention across sequences. `model.attn_kernel` selects the kernel family at model load: `auto` uses FlashAttention-3 on Hopper GPUs (sm90) and FlashAttention-2 elsewhere; `fa2` and `fa3` force one. The paper's Qwen3 runs used FA2 and the Pythia runs FA3. Every in-run evaluation (teacher-forced loss and greedy decoding) switches the model to SDPA, the transformers default, and restores the training kernel afterwards; `search_ft.eval` loads checkpoints with SDPA as well.

**Output head.** `training.head: sparse` (the default) runs the `lm_head`, the argmax and the cross-entropy only on the labelled rows of the packed sequence, which is all the loss and the gate metrics read; the prompt rows, about 99% of the tokens, would carry zero gradient. `full` runs the head over every row in chunks. The two give the same loss up to GEMM rounding.

**Gradient checkpointing.** `model.ckpt_every_n_layers` recomputes only every N-th decoder layer in the backward pass; `model.ckpt_relax_max_tokens` falls back to every layer for micro-batches above that many tokens, since activation memory grows with the curriculum stage. Gradients are identical either way. The defaults (2 and 88,000) were measured for Qwen3-0.6B at `per_device_batch_size: 48` with 4 ranks on 80 GB GPUs and are not portable to other models or batch sizes without a measurement.

**Reproducibility.** The data stream is a deterministic function of `experiment.seed`, the rank and the batch index, so two runs with the same configuration see the same examples in the same order. The kernels are not bitwise deterministic by default: the flash-attention backward is nondeterministic, so two such runs drift apart after the first steps (as do the replicas on different ranks). `experiment.deterministic: true` makes training bitwise reproducible on the same hardware and GPU count by selecting the deterministic flash-attention backward (`FLASH_ATTENTION_DETERMINISTIC=1`), deterministic cuBLAS reductions (`CUBLAS_WORKSPACE_CONFIG=:4096:8`), deterministic cuDNN and `torch.use_deterministic_algorithms`; it costs about 6-7% step time at L=128 on H100. The paper's runs did not use it.

The kernel, head mode, checkpointing settings, determinism flags and library versions of a run are written to `run_meta.json` in the run directory.

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