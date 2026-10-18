import datetime
import os
import time
from collections import deque
from typing import List, Optional, Set

import numpy as np
import torch
from transformers import TrainerCallback

from ..distributed.utils import (
    get_rank, get_world_size, is_main_process, rank_print, barrier, broadcast_object
)
from .lookahead import effective_search_L
from .state import (
    save_curriculum_state, append_to_jsonl, atomic_write_json
)
from ..model.arch import resolve_model_parts


# Compute accounting. The cumulative compute of a run is 6N x tokens (Kaplan et al.), with N every
# parameter of the model (tied embeddings counted once) and tokens every packed token on every rank.
# With the sparse output head (training.head: sparse) the lm_head only runs on the labelled rows, so the
# FLOPs actually executed in a step are 6(N - N_head) x tokens + 6 N_head x head_rows; that count feeds
# the achieved_tflops throughput diagnostic only (executed_flops), the cumulative accounting stays 6N x tokens.

def estimate_flops_per_token(model) -> int:
    """6N: six FLOPs per parameter per token, every parameter."""
    total_params = sum(p.numel() for p in model.parameters())
    return 6 * total_params


def estimate_head_params(model) -> int:
    """N_head: parameter count of the output head the sparse path runs (lm_head for Qwen, where it
    shares the embedding weight and so enters N once; embed_out for Pythia, a separate weight).
    0 when the head cannot be resolved (achieved_tflops then falls back to 6N x tokens)."""
    try:
        return int(sum(p.numel() for p in resolve_model_parts(model).lm_head.parameters()))
    except Exception:
        return 0


def executed_flops(tokens: int, head_rows: int, flops_per_token: int, flops_per_head_row, head: str) -> float:
    """FLOPs actually executed for `tokens` packed tokens of which `head_rows` went through the lm_head.
    head='full' (or N_head unknown): 6N x tokens. head='sparse': 6N x tokens - 6 N_head x (tokens - head_rows),
    clamped so it never exceeds 6N x tokens."""
    flops = tokens * flops_per_token
    if head == "sparse" and flops_per_head_row:
        flops -= flops_per_head_row * max(0, tokens - head_rows)
    return flops


class CurriculumCallback(TrainerCallback):

    def __init__(
        self,
        dataset,
        max_lookahead: int,
        base_lookahead: int = 1,
        lookahead_step: int = 8,
        accuracy_threshold: float = 0.98,
        min_steps_per_stage: int = 500,
        check_every: int = 50,
        # Eval config
        do_stage_eval: bool = False,
        stage_eval_every: int = 1,
        eval_every_steps: int = 0,
        eval_inputs: Optional[List[str]] = None,
        eval_labels: Optional[List[List[str]]] = None,
        eval_fn=None,
        tokenizer=None,
        max_input_size: int = 576,
        seed: Optional[int] = None,
        print_examples: int = 0,
        # Checkpoint config
        persist_every: int = 2000,
        save_stage_checkpoints: bool = True,
        # Compute accounting (trainer/flops.py): 6N per token; 6 N_head per head row (sparse head only)
        flops_per_token: int = 0,
        flops_per_head_row: Optional[int] = None,
        head: str = "sparse",
    ):
        self.dataset = dataset
        self.flops_per_token = flops_per_token
        self.flops_per_head_row = flops_per_head_row
        self.head = head
        self.max_lookahead = max_lookahead
        self.base_lookahead = base_lookahead
        self.lookahead_step = lookahead_step
        self.acc_thr = accuracy_threshold
        self.min_steps = min_steps_per_stage
        self.check_every = check_every

        self.trainer = None
        self.stage_start_step = 0
        self._last_log = -1
        self.finished = False

        # Eval config
        self.do_stage_eval = do_stage_eval
        self.stage_eval_every = stage_eval_every
        self.eval_every_steps = eval_every_steps
        self.eval_inputs = eval_inputs
        self.eval_labels = eval_labels
        self.eval_fn = eval_fn
        self.tokenizer = tokenizer
        self.max_input_size = max_input_size
        self.seed = seed
        self.print_examples = print_examples

        # Checkpoint config
        self.persist_every = persist_every
        self.save_stage_checkpoints = save_stage_checkpoints
        self._last_persist_step = 0

        # Stage timing
        self.stage_start_time = None
        self.samples_this_stage = 0
        self.tokens_this_stage = 0
        self._cumulative_tokens = 0

        # Wall time tracking
        self.wall_time_offset = 0.0
        self.training_start_time = None

        # Loss history + JSONL persistence
        self.loss_history = []
        self.stage_eval_history = []
        self._jsonl_buffer = []
        self._jsonl_flush_every = 10

        # Resume tracking
        self.resume_points = []
        self._mark_next_as_resume = False
        self._last_persist_step = 0

        # Capture grad_norm and lr from Trainer logs
        self.last_grad_norm = 0.0
        self.last_lr = 0.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.last_grad_norm = logs.get('grad_norm', self.last_grad_norm)
            self.last_lr = logs.get('learning_rate', self.last_lr)
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        if self.stage_start_time is None:
            self.stage_start_time = datetime.datetime.now()
        if self.training_start_time is None:
            self.training_start_time = time.time()
        return control

    def _current_wall_time(self) -> float:
        wt = self.wall_time_offset
        if self.training_start_time is not None:
            wt += (time.time() - self.training_start_time)
        return wt

    def _current_metric(self) -> float:
        local_acc = self.trainer.get_full_word_acc()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            device = self.trainer.args.device
            metric_tensor = torch.tensor([local_acc], dtype=torch.float32, device=device)
            torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
            return metric_tensor.item() / torch.distributed.get_world_size()

        return local_acc

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or self.finished or state.global_step == 0:
            return control

        # --- Track samples/tokens for this optimizer step (all micro-batches, all ranks) ---
        ws = get_world_size()
        step_samples = getattr(self.trainer, '_step_samples', 0) * ws
        step_tokens = getattr(self.trainer, '_step_tokens', 0) * ws
        step_head_rows = getattr(self.trainer, '_step_head_rows', 0) * ws
        self.samples_this_stage += step_samples
        self.tokens_this_stage += step_tokens
        # Reset accumulators for next optimizer step
        self.trainer._step_samples = 0
        self.trainer._step_tokens = 0
        self.trainer._step_head_rows = 0

        # --- Record loss history entry ---
        if self.trainer.recent_losses:
            current_loss = self.trainer.recent_losses[-1]
            effective_L = self.dataset._stage_target_lookahead()
            tokens_this_step = step_tokens
            self._cumulative_tokens += tokens_this_step

            wall_time = self._current_wall_time()
            prev_wall = self.loss_history[-1]["wall_time"] if self.loss_history else 0.0
            dt = wall_time - prev_wall
            tokens_per_sec = tokens_this_step / dt if dt > 0 else 0.0
            # Executed FLOPs of this optimizer step (sparse head: the lm_head ran on head_rows rows only)
            # over its wall time; the cumulative PFLOPs stay 6N x tokens.
            achieved_tflops = 0.0
            if self.flops_per_token and dt > 0 and tokens_this_step > 0:
                achieved_tflops = executed_flops(tokens_this_step, step_head_rows, self.flops_per_token,
                                                 self.flops_per_head_row, self.head) / dt / 1e12

            lr = self.trainer.optimizer.param_groups[0]["lr"] if self.trainer.optimizer else 0.0

            entry = {
                "step": state.global_step,
                "loss": current_loss,
                "stage": self.dataset.stage,
                "alpha": self.dataset._stage_alpha(),
                "effective_L": effective_L,
                "tokens": tokens_this_step,
                "head_rows": step_head_rows,
                "head": self.head,
                "cumulative_tokens": self._cumulative_tokens,
                "cumulative_pflops": self._cumulative_tokens * self.flops_per_token / 1e15,
                "wall_time": wall_time,
                "tokens_per_sec": round(tokens_per_sec),
                "achieved_tflops": round(achieved_tflops, 2),
                "n_gpus": ws,
                "lr": lr,
                "first_token_acc": self.trainer.get_first_token_acc(),
                "full_word_acc": self.trainer.get_full_word_acc(),
            }

            if self._mark_next_as_resume:
                entry["resume"] = True
                self._mark_next_as_resume = False

            self.loss_history.append(entry)

            # JSONL incremental persistence
            if is_main_process():
                self._jsonl_buffer.append(entry)
                if len(self._jsonl_buffer) >= self._jsonl_flush_every:
                    try:
                        append_to_jsonl(args.output_dir, self._jsonl_buffer)
                        self._jsonl_buffer = []
                    except Exception as e:
                        rank_print(f"[LOSS][WARN] JSONL flush failed: {e}")

        # --- Logging (every 10 steps) ---
        current_time = datetime.datetime.now()
        if state.global_step % 10 == 0 and state.global_step != self._last_log and is_main_process():
            loss = np.mean(self.trainer.recent_losses) if self.trainer.recent_losses else 0.0
            f1 = self.trainer.get_first_token_acc()
            fw = self.trainer.get_full_word_acc()

            stage_time_str = "0m"
            tokens_per_sec = 0.0
            if self.stage_start_time:
                stage_time = (current_time - self.stage_start_time).total_seconds()
                stage_time_str = f"{stage_time / 60:.1f}m"
                if stage_time > 0 and self.tokens_this_stage > 0:
                    tokens_per_sec = self.tokens_this_stage / stage_time

            tokens_per_sec_str = f"{tokens_per_sec / 1000:.1f}K" if tokens_per_sec >= 1000 else f"{tokens_per_sec:.0f}"

            target_L = self.dataset._stage_target_lookahead()
            expected_stages = (self.max_lookahead - self.base_lookahead) // self.lookahead_step + 1
            stage_str = f"[Stage {self.dataset.stage}/{expected_stages} L={target_L}]"

            print(f"{stage_str} step {state.global_step} | "
                  f"loss={loss:.4f} | First={f1:.2%} | Full={fw:.2%} | "
                  f"lr={self.last_lr:.2e} | grad_norm={self.last_grad_norm:.2f} | "
                  f"{tokens_per_sec_str} toks/s | stage_time={stage_time_str}")
            self._last_log = state.global_step

        # --- Periodic eval ---
        if self.eval_every_steps > 0 and state.global_step % self.eval_every_steps == 0:
            self._run_eval(self.dataset.stage, state.global_step, include_stage_alpha=False)

        # --- Stage advancement check ---
        if (state.global_step % self.check_every == 0
                and (state.global_step - self.stage_start_step) >= self.min_steps):

            m = self._current_metric()

            if is_main_process():
                print(f"[CHECK] Stage {self.dataset.stage} full={m:.2%} target>={self.acc_thr:.2%}")

            barrier()

            if m >= self.acc_thr:
                self._advance_stage(state, control, current_time)

        return control

    def _advance_stage(self, state, control, current_time):
        old_stage = self.dataset.stage

        if is_main_process():
            if self.stage_start_time:
                total_stage_time = (current_time - self.stage_start_time).total_seconds()
                steps_in_stage = state.global_step - self.stage_start_step
                print(f"[COMPLETE] Stage {old_stage} complete in {steps_in_stage} steps, "
                      f"{total_stage_time / 60:.1f} minutes | "
                      f"{self.samples_this_stage:,} samples")

        # Run stage eval (skip if not a multiple of stage_eval_every)
        target_L = self.dataset._stage_target_lookahead()
        if self.stage_eval_every <= 1 or (target_L and target_L % self.stage_eval_every == 0):
            self._run_eval(old_stage, state.global_step, include_stage_alpha=True)

        # Save stage checkpoint
        if self.save_stage_checkpoints:
            try:
                ckpt_L = target_L if target_L is not None else "?"
                stage_ckpt_dir = os.path.join(
                    self.trainer.args.output_dir,
                    "stage_checkpoints",
                    f"stage_{old_stage}_step_{state.global_step}_L{ckpt_L}"
                )
                rank_print(f"[STAGE-CKPT] Saving to {stage_ckpt_dir}")
                self.trainer.save_model(stage_ckpt_dir)
                barrier()
                if is_main_process():
                    save_curriculum_state(
                        stage_ckpt_dir, old_stage, self.stage_start_step,
                        self._current_wall_time(),
                        first_token_correct=self.trainer.first_token_correct,
                        full_word_correct=self.trainer.full_word_correct,
                        recent_losses=self.trainer.recent_losses,
                        samples_this_stage=self.samples_this_stage,
                        tokens_this_stage=self.tokens_this_stage,
                    )
            except Exception as e:
                rank_print(f"[STAGE-CKPT] Warning: Could not save: {e}")

        # Check if final stage
        if self.dataset._is_final_stage():
            if is_main_process():
                print("[FINISHED] Curriculum complete (reached max_lookahead)")
            control.should_save = True
            control.should_training_stop = True
            self.finished = True
        else:
            # Advance to next stage
            self.dataset.stage += 1
            self.stage_start_step = state.global_step
            self.stage_start_time = datetime.datetime.now()
            self.samples_this_stage = 0
            self.tokens_this_stage = 0

            # Clear accuracy tracking for fresh measurement
            self.trainer.first_token_correct.clear()
            self.trainer.full_word_correct.clear()

            # Reset timing
            if hasattr(self.trainer, 'reset_timing'):
                self.trainer.reset_timing()

            new_alpha = self.dataset._stage_alpha()
            new_L = self.dataset._stage_target_lookahead()
            if is_main_process():
                print(f" -> Advanced to stage {self.dataset.stage} | "
                      f"alpha={new_alpha:.3f} | L={new_L}/{self.max_lookahead}")

                # Log stage advancement event
                advance_entry = {
                    "step": state.global_step,
                    "event": "stage_advance",
                    "new_stage": self.dataset.stage,
                    "new_L": new_L,
                    "new_alpha": new_alpha,
                    "wall_time": self._current_wall_time(),
                    "cumulative_tokens": self._cumulative_tokens,
                }
                self._jsonl_buffer.append(advance_entry)
                self.loss_history.append(advance_entry)

                save_curriculum_state(
                    self.trainer.args.output_dir, self.dataset.stage,
                    self.stage_start_step, self._current_wall_time(),
                    first_token_correct=self.trainer.first_token_correct,
                    full_word_correct=self.trainer.full_word_correct,
                    recent_losses=self.trainer.recent_losses,
                    samples_this_stage=self.samples_this_stage,
                    tokens_this_stage=self.tokens_this_stage,
                )

    def _run_eval(self, stage: int, global_step: int, include_stage_alpha: bool = False):
        if not self.do_stage_eval:
            return
        if self.eval_inputs is None or self.eval_fn is None:
            return

        target_L = self.dataset._stage_target_lookahead()
        rank_print(f"\n[EVAL] Stage {stage} | Step {global_step} | L={target_L} | Evaluating...")

        model = self.trainer.model
        was_training = model.training
        model.eval()

        with torch.no_grad():
            # Full difficulty eval (alpha=1.0, heldout set)
            result = self.eval_fn(
                model=model,
                tokenizer=self.tokenizer,
                inputs=self.eval_inputs,
                labels=self.eval_labels,
                print_examples=self.print_examples,
                use_chat_template=self.dataset.use_chat_template,
            )

            # Stage-alpha eval (if not already at full difficulty)
            stage_result = None
            if include_stage_alpha and not self.dataset._is_final_stage():
                stage_alpha = self.dataset._stage_alpha()
                rank_print(f"[EVAL] Also evaluating at stage alpha={stage_alpha:.4f}...")

                # Generate stage-difficulty eval data on rank 0, broadcast to all
                stage_eval_inputs, stage_eval_labels = None, None
                if is_main_process():
                    from ..data.heldout import generate_eval_at_alpha
                    stage_eval_inputs, stage_eval_labels = generate_eval_at_alpha(
                        n_samples=min(len(self.eval_inputs), 500),
                        max_input_size=self.max_input_size,
                        max_lookahead=self.max_lookahead,
                        alpha=stage_alpha,
                        tokenizer=self.tokenizer,
                        seed=(self.seed or 0) + global_step,
                    )
                barrier()
                stage_eval_inputs = broadcast_object(stage_eval_inputs, src=0)
                stage_eval_labels = broadcast_object(stage_eval_labels, src=0)

                if stage_eval_inputs:
                    stage_result = self.eval_fn(
                        model=model,
                        tokenizer=self.tokenizer,
                        inputs=stage_eval_inputs,
                        labels=stage_eval_labels,
                        print_examples=0,
                        use_chat_template=self.dataset.use_chat_template,
                    )

        if was_training:
            model.train()

        # Log results
        if result:
            log_msg = (f"[EVAL] Stage {stage} | Step {global_step} | L={target_L} | "
                       f"TF Loss={result.get('tf_loss', 0):.4f} | "
                       f"First={result.get('first_token_acc', 0):.2%} | "
                       f"Full={result.get('full_word_acc', 0):.2%}")
            if stage_result:
                log_msg += (f"\n[EVAL] (stage-alpha) "
                            f"First={stage_result.get('first_token_acc', 0):.2%} | "
                            f"Full={stage_result.get('full_word_acc', 0):.2%}")
            rank_print(log_msg)

            # Store history
            entry = {
                "stage": stage,
                "step": global_step,
                "effective_L": target_L,
                "alpha_training": self.dataset._stage_alpha(),
                "tf_loss": result.get("tf_loss", 0),
                "greedy_first": result.get("first_token_acc", 0),
                "greedy_full": result.get("full_word_acc", 0),
            }
            if stage_result:
                entry["stage_greedy_first"] = stage_result.get("first_token_acc", 0)
                entry["stage_greedy_full"] = stage_result.get("full_word_acc", 0)
            self.stage_eval_history.append(entry)

            # Save to file
            if is_main_process():
                try:
                    out_path = os.path.join(self.trainer.args.output_dir, "stage_eval_history.json")
                    atomic_write_json(out_path, self.stage_eval_history)
                except Exception as e:
                    rank_print(f"[EVAL] Warning: Could not save history: {e}")

    def on_save(self, args, state, control, **kwargs):
        barrier()

        if is_main_process():
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            save_curriculum_state(
                checkpoint_dir, self.dataset.stage, self.stage_start_step,
                self._current_wall_time(),
                first_token_correct=self.trainer.first_token_correct,
                full_word_correct=self.trainer.full_word_correct,
                recent_losses=self.trainer.recent_losses,
                samples_this_stage=self.samples_this_stage,
                tokens_this_stage=self.tokens_this_stage,
            )

            # Flush JSONL buffer
            if self._jsonl_buffer:
                try:
                    append_to_jsonl(args.output_dir, self._jsonl_buffer)
                    self._jsonl_buffer = []
                except Exception as e:
                    rank_print(f"[LOSS][WARN] JSONL flush on save failed: {e}")


        # Persistent checkpoint
        if (self.persist_every > 0
                and state.global_step >= self._last_persist_step + self.persist_every):
            try:
                persist_dir = os.path.join(args.output_dir, "persistent_checkpoints",
                                           f"step_{state.global_step}")
                rank_print(f"[PERSIST] Saving persistent checkpoint to {persist_dir}")
                self.trainer.save_model(persist_dir)
                barrier()
                if is_main_process():
                    save_curriculum_state(
                        persist_dir, self.dataset.stage, self.stage_start_step,
                        self._current_wall_time(),
                        first_token_correct=self.trainer.first_token_correct,
                        full_word_correct=self.trainer.full_word_correct,
                        recent_losses=self.trainer.recent_losses,
                        samples_this_stage=self.samples_this_stage,
                        tokens_this_stage=self.tokens_this_stage,
                    )
                self._last_persist_step = state.global_step
            except Exception as e:
                rank_print(f"[PERSIST] Warning: Could not save: {e}")

        barrier()
        return control
