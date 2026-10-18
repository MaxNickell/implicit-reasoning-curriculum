import datetime
import json
import os

from transformers import TrainingArguments

from .config import load_config, config_to_dict
from .distributed.utils import (
    set_all_seeds, rank_print, is_main_process, get_world_size, barrier
)
from .model.loader import load_model_and_tokenizer
from .data.dataset import PackedSequenceDataset
from .data.heldout import load_heldout, eval_data_fingerprint
from .trainer.packed_trainer import PackedSequenceTrainer
from .trainer.flops import estimate_flops_per_token, estimate_head_params
from .trainer.provenance import run_provenance
from .curriculum.callback import CurriculumCallback
from .checkpoint.resume import prepare_resume, apply_resume_to_callback, apply_resume_to_trainer
from .evaluation.inline import run_eval


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config = load_config()

    # Derived values
    max_input_size = 6 * config.curriculum.max_lookahead

    # Output directory: {base_dir}/runs/{name}/
    if not config.experiment.name:
        raise ValueError("experiment.name must be set (used as the run directory name)")
    output_dir = config.paths.run_dir(config.experiment.name)
    os.makedirs(output_dir, exist_ok=True)

    # Seeds
    set_all_seeds(config.experiment.seed, deterministic=config.experiment.deterministic)

    # Print config
    if is_main_process():
        print("\n" + "=" * 60)
        print("SYNTHETIC SEARCH CURRICULUM TRAINING")
        print("=" * 60)
        for section_name, section in config_to_dict(config).items():
            for k, v in section.items():
                print(f"  {section_name}.{k}: {v}")
        print(f"  [derived] max_input_size: {max_input_size}")
        print(f"  [derived] world_size: {get_world_size()}")
        print("=" * 60 + "\n")

    # --- Resume / grad_accum ---
    resume_info = prepare_resume(
        output_dir=output_dir,
        effective_batch_size=config.training.effective_batch_size,
        per_device_batch_size=config.training.per_device_batch_size,
    )

    grad_accum = resume_info.grad_accum
    resume_step = resume_info.resume_step

    rank_print(f"[TRAIN] grad_accum={grad_accum} | "
               f"effective_batch={config.training.effective_batch_size} | "
               f"per_device={config.training.per_device_batch_size} | "
               f"world_size={get_world_size()}")

    # --- Model + Tokenizer ---
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.name,
        dtype=config.model.dtype,
        gradient_checkpointing=config.model.gradient_checkpointing,
        reinit_weights=config.model.reinit_weights,
        use_liger=config.model.use_liger,
        attn_kernel=config.model.attn_kernel,
    )

    # Compute accounting: cumulative compute is 6N x tokens over every parameter and every rank; N_head
    # only corrects the achieved_tflops throughput diagnostic for the sparse head (trainer/flops.py).
    flops_per_token = estimate_flops_per_token(model)
    n_head = estimate_head_params(model)
    flops_per_head_row = 6 * n_head if n_head > 0 else None
    rank_print(f"[FLOPS] Model has {flops_per_token / 6 / 1e9:.3f}B params, 6N = {flops_per_token / 1e9:.1f}B FLOPs/token; "
               f"output head N_head = {n_head / 1e6:.1f}M params; training.head={config.training.head}")

    # --- Heldout eval set ---
    heldout_path = config.paths.heldout_path(config.curriculum.max_lookahead)
    if not os.path.isfile(heldout_path):
        raise FileNotFoundError(
            f"Heldout file not found: {heldout_path}\n"
            "Generate it first:\n"
            f"  python -m search_ft.generate_heldout "
            f"--max_lookahead {config.curriculum.max_lookahead}"
        )

    rank_print(f"[DATA] Loading heldout eval set from {heldout_path}...")
    heldout_examples, reserved_inputs = load_heldout(heldout_path)
    eval_inputs = [ex["input_text"] for ex in heldout_examples]
    eval_labels = [ex["output_texts"] for ex in heldout_examples]
    eval_fp = eval_data_fingerprint(eval_inputs, eval_labels)
    rank_print(f"[DATA] Eval set: {len(eval_inputs)} examples, fingerprint={eval_fp}")

    # --- Dataset ---
    initial_stage = 1
    if resume_info.curriculum_state:
        initial_stage = resume_info.curriculum_state.get("stage", 1)

    dataset = PackedSequenceDataset(
        tokenizer=tokenizer,
        batch_size=config.training.per_device_batch_size,
        stage=initial_stage,
        max_input_size=max_input_size,
        max_lookahead=config.curriculum.max_lookahead,
        base_lookahead=config.curriculum.base_lookahead,
        lookahead_step=config.curriculum.lookahead_step,
        reserved_inputs=reserved_inputs,
        seed=config.experiment.seed,
        resume_step=resume_step,
        fixed_vocab=config.data.fixed_vocab,
        instruction_mix_ratio=config.data.instruction_mix_ratio,
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # We handle batching ourselves (packed)
        gradient_accumulation_steps=grad_accum,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type="constant",  # constant LR for the whole run: no warmup, no schedule
        max_steps=config.training.max_steps,
        save_steps=config.training.save_steps,
        save_total_limit=1,  # One rolling checkpoint for resume safety
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=config.model.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if config.model.gradient_checkpointing else None
        ),
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # --- Curriculum Callback ---
    curriculum = CurriculumCallback(
        dataset=dataset,
        max_lookahead=config.curriculum.max_lookahead,
        base_lookahead=config.curriculum.base_lookahead,
        lookahead_step=config.curriculum.lookahead_step,
        accuracy_threshold=config.curriculum.accuracy_threshold,
        min_steps_per_stage=config.curriculum.min_steps_per_stage,
        check_every=config.curriculum.check_every,
        do_stage_eval=config.eval.stage_eval,
        stage_eval_every=config.eval.stage_eval_every,
        eval_every_steps=config.eval.eval_every_steps,
        eval_inputs=eval_inputs,
        eval_labels=eval_labels,
        eval_fn=run_eval,
        tokenizer=tokenizer,
        max_input_size=max_input_size,
        seed=config.experiment.seed,
        print_examples=config.eval.print_examples,
        persist_every=config.eval.persist_every,
        save_stage_checkpoints=config.eval.save_stage_checkpoints,
        flops_per_token=flops_per_token,
        flops_per_head_row=flops_per_head_row,
        head=config.training.head,
    )

    # --- Trainer ---
    trainer = PackedSequenceTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        accuracy_window=config.training.accuracy_window,
        head=config.training.head,
        ckpt_every_n_layers=config.model.ckpt_every_n_layers,
        ckpt_relax_max_tokens=config.model.ckpt_relax_max_tokens,
        ddp_grad_average=config.training.ddp_grad_average,
        callbacks=[curriculum],
    )

    # Wire up references
    curriculum.trainer = trainer

    # --- Apply resume state ---
    if not resume_info.is_fresh:
        apply_resume_to_callback(resume_info, curriculum)
        apply_resume_to_trainer(resume_info, trainer)

    # Save config for reproducibility
    if is_main_process():
        config_path = os.path.join(output_dir, "config.yaml")
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config_to_dict(config), f, default_flow_style=False)
        rank_print(f"[TRAIN] Config saved to {config_path}")

        # Run provenance (kernel, head mode, flags, library versions) so later re-evaluations are attributable
        try:
            meta = {
                "created_at": datetime.datetime.now().isoformat(),
                "resume_from": resume_info.checkpoint_path,
                "world_size": get_world_size(),
                "grad_accum": grad_accum,
                "eval_fingerprint": eval_fp,
                "flops_per_token": flops_per_token,
                "provenance": run_provenance(model, config),
            }
            with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            rank_print(f"[META][WARN] Could not write run_meta.json: {e}")

    # --- Train ---
    rank_print(f"[TRAIN] Starting training...")
    trainer.train(resume_from_checkpoint=resume_info.checkpoint_path)

    # --- Done ---
    barrier()
    rank_print("[TRAIN] Training complete!")


if __name__ == "__main__":
    main()
