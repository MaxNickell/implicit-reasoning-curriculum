"""CPU-only end-to-end check of PackedSequenceTrainer.compute_loss on a packed micro-batch.

Builds a 2-layer random-init Qwen3 model from a config (no download, fp32, attn_implementation='eager'), packs
four hand-made sequences with search_ft.data.collate.pack_batch and runs the real PackedSequenceTrainer.compute_loss
(hf_forward_body -> head_ce_and_preds -> loss and gate metrics) on CPU. Under eager/SDPA transformers derives a
block-diagonal causal mask from the restarting position_ids, so the packed forward must agree with a per-sequence
forward:
  * loss == sum over sequences of (HF per-sequence CE sum) / total labelled rows, to 1e-5;
  * sparse and full head give the same loss; head_rows is n_valid (sparse) and Tm1 (full);
  * the gate deques receive one first-token and one full-word entry per search sequence (none for the
    instruction-style sequence with prompt_len 0);
  * with gradient checkpointing on and ckpt_every_n_layers=2 the loss is unchanged, backward runs, and the layer
    flags are [True, False]; above ckpt_relax_max_tokens every layer is recomputed ([True, True]);
  * the step counters (_step_tokens, _step_samples, _step_head_rows) accumulate across micro-batches.
Run from the repository root:  python tests/test_packed_loss_cpu.py   (no GPU, no flash-attn, no generator build)
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments

from search_ft.data.collate import pack_batch
from search_ft.trainer.packed_trainer import PackedSequenceTrainer

TOL = 1e-5


def tiny_qwen():
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
        attn_implementation="eager", tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg), cfg


def make_samples(V, seed=0):
    g = torch.Generator().manual_seed(seed)
    specs = [(7, 2), (5, 3), (9, 1)]   # (prompt_len, answer_len) search sequences
    samples = []
    for pl, al in specs:
        ids = torch.randint(3, V, (pl + al,), generator=g).tolist()
        samples.append({
            "input_ids": ids, "labels": [-100] * pl + ids[pl:], "prompt_len": pl,
            "valid_first_targets": sorted({ids[pl], (ids[pl] + 1) % V}), "seq_len": len(ids),
        })
    # instruction-style sequence: prompt_len 0, labels on a middle span, first label -100 (as in the real data)
    ids = torch.randint(3, V, (10,), generator=g).tolist()
    labels = [-100] * 10
    for t in range(4, 9):
        labels[t] = ids[t]
    samples.append({"input_ids": ids, "labels": labels, "prompt_len": 0, "valid_first_targets": [], "seq_len": 10})
    return samples


def reference_loss(model, samples):
    """Per-sequence HF forward (labels shifted inside the model): sum of CE over all labelled rows / their count."""
    total_ce, n_valid = 0.0, 0
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor([s["input_ids"]])
            lab = torch.tensor([s["labels"]])
            out = model(input_ids=ids, labels=lab)
            k = int((lab[0, 1:] != -100).sum())
            total_ce += out.loss.item() * k
            n_valid += k
    return total_ce / n_valid, n_valid


def make_trainer(model, tmpdir, gradient_checkpointing=False, **kw):
    args = TrainingArguments(
        output_dir=tmpdir, use_cpu=True, report_to="none", per_device_train_batch_size=1,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        logging_strategy="no", save_strategy="no",
    )
    return PackedSequenceTrainer(model=model, args=args, **kw)


def test_packed_loss_matches_per_sequence():
    torch.manual_seed(0)
    model, cfg = tiny_qwen()
    model.eval()
    samples = make_samples(cfg.vocab_size)
    batch = pack_batch(samples)
    ref, n_valid = reference_loss(model, samples)
    Tm1 = batch["input_ids"].shape[1] - 1
    n_search = sum(1 for s in samples if s["prompt_len"] > 0)

    with tempfile.TemporaryDirectory() as tmp:
        for head in ("sparse", "full"):
            trainer = make_trainer(model, tmp, head=head, accuracy_window=800)
            loss = trainer.compute_loss(model, dict(batch))
            assert abs(loss.item() - ref) < TOL, (head, loss.item(), ref)
            assert trainer._step_head_rows == (n_valid if head == "sparse" else Tm1), (head, trainer._step_head_rows)
            assert trainer._step_tokens == batch["input_ids"].shape[1] and trainer._step_samples == len(samples)
            assert trainer._last_batch_tokens == int(batch["cu_seq_lens"][-1])
            assert len(trainer.first_token_correct) == n_search, len(trainer.first_token_correct)
            assert len(trainer.full_word_correct) == n_search, len(trainer.full_word_correct)
            assert trainer.recent_losses[-1] == loss.item()
            # a second micro-batch accumulates the step counters (the callback resets them per optimizer step)
            trainer.compute_loss(model, dict(batch))
            assert trainer._step_tokens == 2 * batch["input_ids"].shape[1]
            assert trainer._step_samples == 2 * len(samples)
            assert len(trainer.first_token_correct) == 2 * n_search
            print(f"test_packed_loss [{head}]: loss={loss.item():.6f} == per-sequence {ref:.6f}; "
                  f"head_rows={trainer._step_head_rows // 2}, n_valid={n_valid}, Tm1={Tm1}  OK")


def test_packed_loss_with_checkpointing():
    torch.manual_seed(0)
    model, cfg = tiny_qwen()
    samples = make_samples(cfg.vocab_size)
    batch = pack_batch(samples)
    model.eval()
    ref, _ = reference_loss(model, samples)

    with tempfile.TemporaryDirectory() as tmp:
        # What Trainer does at train start, then train mode so the per-layer flags take effect.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        layers = model.model.layers

        trainer = make_trainer(model, tmp, gradient_checkpointing=True, head="sparse",
                               ckpt_every_n_layers=2, ckpt_relax_max_tokens=0)
        loss = trainer.compute_loss(model, dict(batch))
        assert [l.gradient_checkpointing for l in layers] == [True, False]
        assert abs(loss.item() - ref) < TOL, (loss.item(), ref)
        loss.backward()
        assert layers[0].mlp.down_proj.weight.grad is not None
        assert layers[1].mlp.down_proj.weight.grad is not None
        model.zero_grad()

        # Relax gate: the micro-batch (T tokens) exceeds ckpt_relax_max_tokens -> every layer recomputed.
        T = batch["input_ids"].shape[1]
        trainer2 = make_trainer(model, tmp, gradient_checkpointing=True, head="sparse",
                                ckpt_every_n_layers=2, ckpt_relax_max_tokens=T - 1)
        loss2 = trainer2.compute_loss(model, dict(batch))
        assert [l.gradient_checkpointing for l in layers] == [True, True]
        assert abs(loss2.item() - ref) < TOL
        loss2.backward()
        model.zero_grad()

        # Gate inactive when the micro-batch is at or below the threshold.
        trainer3 = make_trainer(model, tmp, gradient_checkpointing=True, head="full",
                                ckpt_every_n_layers=2, ckpt_relax_max_tokens=T)
        loss3 = trainer3.compute_loss(model, dict(batch))
        assert [l.gradient_checkpointing for l in layers] == [True, False]
        assert abs(loss3.item() - ref) < TOL
        print(f"test_packed_loss_with_checkpointing: ckpt_every=2 -> [True, False], relax gate -> [True, True], "
              f"loss unchanged ({loss.item():.6f})  OK")


def main():
    test_packed_loss_matches_per_sequence()
    test_packed_loss_with_checkpointing()
    print("test_packed_loss_cpu: ALL OK")


if __name__ == "__main__":
    main()
