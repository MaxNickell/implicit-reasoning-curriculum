# Configs

One YAML per training arm of the paper. Every file inherits from `base.yaml` (directly or through a shared
`*_base.yaml`), states only what differs from it, and names in its header the paper element it reproduces, the launch
script or run directory its values were taken from, and the `search_ft.generate_heldout` command for its held-out set.

```bash
python -m search_ft.generate_heldout --max_lookahead 256 --samples 500     # each header gives the exact command
torchrun --nproc_per_node=4 -m search_ft.train --config configs/qwen06b_s8.yaml
```

Conventions:

- Batch geometry is `training.effective_batch_size` (sequences per optimizer step across all ranks) and
  `training.per_device_batch_size`; gradient accumulation is derived as effective / (per_device x world_size), so a
  file reproduces its archived run on any GPU count that divides it. The values below were checked at 4 ranks against
  each launch script's accumulation steps.
- Every file states `training.accuracy_window: 800`, the gate window the paper reports. Where the archived run gated on
  another window, the header says so in one line.
- The archived launcher took a stage count (`n_stages`) and a warmup flag. This code derives the stage count from
  `base_lookahead`, `max_lookahead` and `lookahead_step`, and the warmup flag was a no-op under the constant
  learning-rate schedule, so neither appears here.
- A fixed-L run without curriculum (the archived `--lookahead_step 0 --n_stages 1`) is written as
  `base_lookahead = max_lookahead = L` with `accuracy_threshold: 1.1`: stage 1 is then the final stage, the gate never
  fires, and `training.max_total_pflops` ends the run. `lookahead_step: 0` itself is not valid here.
- `model.name` is a Hugging Face model id; `model.reinit_weights: true` re-draws every weight before training.

| File | Paper element | Model | Key settings |
|---|---|---|---|
| `qwen06b_s8.yaml` | Section 4.2 reference curriculum; Table 1 "with curriculum"; Table 11 row s=8 | Qwen3-0.6B | s=8, L 8 to 256, gate 0.98, batch 768 (48 x 4 ranks x 4), lr 5e-5, 300K PFLOPs |
| `stepsize_s{1,2,4,8,16,32,64}.yaml` (+ `stepsize_base.yaml`) | Section 4.4 step-size sweep; Table 11 | Qwen3-0.6B | base = step = s, L to 512 at n 3079, gate 0.98, batch 768, 1.5M PFLOPs, no in-run eval |
| `pythia160m_s1.yaml` | Section 4.5 Pythia family; Tables 12 to 15 | Pythia-160M | s=1, L to 96, gate 0.98, batch 2048 (256 x 4 x 2), lr 1e-4, 1M PFLOPs, no in-run eval |
| `pythia410m_s1.yaml` | Section 4.5; Tables 12 to 15 | Pythia-410M | s=1, L to 96, batch 192 (48 x 4 x 1), lr 5e-5, no in-run eval |
| `pythia14b_s1.yaml` | Section 4.5; Tables 12 to 15; pretrained arm of Section 4.6 | Pythia-1.4B | s=1, L to 96, batch 192 (24 x 4 x 2), lr 2.7e-5, 5M PFLOPs |
| `pythia28b_s1.yaml` | Section 4.5; Tables 12 to 15 | Pythia-2.8B | s=1, L to 256, batch 1536 (48 x 4 x 8), lr 2.7e-5, min_steps_per_stage 0 |
| `qwen06b_s1_random_init.yaml` | Section 4.6 random initialization; Table 16 | Qwen3-0.6B, random init | s=1, L to 256, batch 1536 (96 x 4 x 4), lr 3e-4, no in-run eval |
| `pythia14b_s1_random_init.yaml` | Section 4.6; Table 16 | Pythia-1.4B, random init | s=1, L to 96, batch 768 (192 x 4 x 1), lr 1e-4 |
| `qwen06b_curr48_thr99.yaml` | Section 4.7 Dolci-mixed curriculum; Table 3 | Qwen3-0.6B | s=1, L to 96, gate 0.99, 6% Dolci-Instruct-SFT under the chat template, batch 768, 300K PFLOPs |
| `qwen06b_curr48_thr98.yaml` | Section 4.7; Table 9 row "Qwen3-0.6B curriculum, 6% Dolci" | Qwen3-0.6B | as above at gate 0.98 |
| `qwen17b_curr48.yaml` | Section 4.7 curriculum arm; Tables 3 and 4 | Qwen3-1.7B | s=1, L to 128 at n 780, gate 0.99, 6% Dolci, batch 768, 2M PFLOPs; the tables use the L=48 stage checkpoint |
| `qwen17b_nocurr_L{8,16,32,48}.yaml` (+ `qwen17b_nocurr_base.yaml`) | Section 4.7 no-curriculum baselines; Tables 3 and 4 | Qwen3-1.7B | fixed L, gate 1.1, 6% Dolci, batch 768; budgets 5,425 / 25,576 / 86,298 / 127,463 PFLOPs at n 192 / 192 / 192 / 288 |
| `vocabpool_grow.yaml`, `vocabpool_fixed.yaml` | Section 4.1 entity-vocabulary ablation; Table 10 | Qwen3-0.6B | s=8, L to 128, 87,483 PFLOPs |
| `breadth3.yaml`, `breadth4.yaml` | Appendix D.6 breadth ablation; Table 17 | Qwen3-0.6B | s=8, L to 256 at n 3080, breadth 3 / 4, 1.5M PFLOPs |
| `shuffled_matched.yaml` | exposure-matched shuffled control of the s=8 curriculum | Qwen3-0.6B | the s=8 run's stage mix in random order, 22,125 steps |
| `example.yaml` | none, a short example | Qwen3-0.6B | s=8, L to 96 |

Section and table numbers follow the September 2026 draft of the paper.
