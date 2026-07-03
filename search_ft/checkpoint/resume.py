import os
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from transformers.trainer_utils import get_last_checkpoint

from ..distributed.utils import rank_print, is_main_process, get_world_size
from ..curriculum.state import (
    restore_curriculum_state,
    get_checkpoint_step,
    save_run_config,
    load_run_config,
    load_from_jsonl,
    rewrite_jsonl,
)


@dataclass
class ResumeInfo:
    checkpoint_path: Optional[str] = None
    resume_step: int = 0
    grad_accum: int = 1
    # Curriculum state from checkpoint
    curriculum_state: Optional[Dict[str, Any]] = None
    # Restored loss/eval histories
    loss_history: List[Dict] = field(default_factory=list)
    stage_eval_history: List[Dict] = field(default_factory=list)
    # Whether this is a fresh start
    is_fresh: bool = True


def compute_grad_accum(effective_batch_size: int, per_device_batch_size: int, world_size: int) -> int:
    if per_device_batch_size * world_size > effective_batch_size:
        rank_print(f"[WARN] per_device_batch_size ({per_device_batch_size}) * world_size ({world_size}) "
                   f"= {per_device_batch_size * world_size} > effective_batch_size ({effective_batch_size}). "
                   f"Setting grad_accum=1.")
        return 1

    grad_accum = effective_batch_size // (per_device_batch_size * world_size)
    if grad_accum < 1:
        grad_accum = 1

    actual_eff = per_device_batch_size * grad_accum * world_size
    if actual_eff != effective_batch_size:
        rank_print(f"[WARN] effective_batch_size={effective_batch_size} not evenly divisible. "
                   f"Using grad_accum={grad_accum} → actual effective={actual_eff}")

    return grad_accum


def prepare_resume(
    output_dir: str,
    effective_batch_size: int,
    per_device_batch_size: int,
) -> ResumeInfo:
    current_world_size = get_world_size()
    grad_accum = compute_grad_accum(effective_batch_size, per_device_batch_size, current_world_size)

    info = ResumeInfo(grad_accum=grad_accum)

    # Check for existing checkpoint
    checkpoint_path = None
    if os.path.isdir(output_dir):
        checkpoint_path = get_last_checkpoint(output_dir)

    if checkpoint_path is None:
        # Fresh start
        rank_print(f"[RESUME] Fresh start (no checkpoint found in {output_dir})")
        info.is_fresh = True

        # Save run config for future resumes to detect world_size changes
        if is_main_process():
            os.makedirs(output_dir, exist_ok=True)
            save_run_config(
                output_dir,
                effective_batch_size=effective_batch_size,
                per_device_batch_size=per_device_batch_size,
                grad_accum=grad_accum,
                world_size=current_world_size,
            )

        return info

    # --- Resuming from checkpoint ---
    info.is_fresh = False
    info.checkpoint_path = checkpoint_path
    info.resume_step = get_checkpoint_step(checkpoint_path)
    rank_print(f"[RESUME] Found checkpoint: {checkpoint_path} (step {info.resume_step})")

    # Detect world_size change
    saved_config = load_run_config(output_dir)
    if saved_config:
        saved_ws = saved_config.get("world_size", current_world_size)
        if saved_ws != current_world_size:
            rank_print(f"[RESUME] World size changed: {saved_ws} → {current_world_size}")
            rank_print(f"[RESUME] Adjusting grad_accum to maintain effective_batch_size={effective_batch_size}")
            # grad_accum already computed correctly above from current world_size
        else:
            rank_print(f"[RESUME] World size unchanged ({current_world_size})")

    # Update run config with current settings
    if is_main_process():
        save_run_config(
            output_dir,
            effective_batch_size=effective_batch_size,
            per_device_batch_size=per_device_batch_size,
            grad_accum=grad_accum,
            world_size=current_world_size,
        )

    # Restore curriculum state
    info.curriculum_state = restore_curriculum_state(checkpoint_path)

    # Restore loss history (truncate to checkpoint step)
    loss_entries = load_from_jsonl(output_dir, max_step=info.resume_step)
    if loss_entries:
        info.loss_history = loss_entries
        rank_print(f"[RESUME] Restored {len(loss_entries)} loss history entries "
                   f"(truncated to step {info.resume_step})")
        # Rewrite JSONL with truncated entries (remove post-crash entries)
        if is_main_process():
            try:
                rewrite_jsonl(output_dir, loss_entries)
            except Exception as e:
                rank_print(f"[RESUME][WARN] Failed to rewrite JSONL: {e}")

    # Restore stage eval history
    eval_path = os.path.join(output_dir, "stage_eval_history.json")
    if os.path.isfile(eval_path):
        try:
            import json
            with open(eval_path) as f:
                info.stage_eval_history = json.load(f)
            rank_print(f"[RESUME] Restored {len(info.stage_eval_history)} stage eval records")
        except Exception as e:
            rank_print(f"[RESUME][WARN] Failed to restore stage eval history: {e}")

    rank_print(f"[RESUME] grad_accum={grad_accum} | effective_batch={effective_batch_size} | "
               f"per_device={per_device_batch_size} | world_size={current_world_size}")

    return info


def apply_resume_to_callback(info: ResumeInfo, callback) -> None:
    if info.is_fresh or info.curriculum_state is None:
        return

    cs = info.curriculum_state

    # Restore callback state
    callback.stage_start_step = int(cs.get("stage_start_step", 0))
    callback.wall_time_offset = float(cs.get("wall_time_offset", 0.0))
    callback.samples_this_stage = int(cs.get("samples_this_stage", 0))
    callback.tokens_this_stage = int(cs.get("tokens_this_stage", 0))

    # Restore loss and eval histories
    callback.loss_history = info.loss_history
    callback.stage_eval_history = info.stage_eval_history

    # Mark next entry as resume point
    if info.loss_history:
        callback._mark_next_as_resume = True
        callback.resume_points.append(info.resume_step)
        callback._last_persist_step = info.loss_history[-1].get("step", 0)

        # Update wall_time_offset from last recorded entry
        last_wall_time = info.loss_history[-1].get("wall_time", 0.0)
        if last_wall_time > callback.wall_time_offset:
            callback.wall_time_offset = last_wall_time

        # Restore cumulative tokens
        last_cumulative = info.loss_history[-1].get("cumulative_tokens", 0)
        if last_cumulative:
            callback._cumulative_tokens = last_cumulative

    rank_print(f"[RESUME] Applied to callback: stage_start_step={callback.stage_start_step}, "
               f"wall_time_offset={callback.wall_time_offset:.1f}s")


def apply_resume_to_trainer(info: ResumeInfo, trainer) -> None:
    if info.is_fresh or info.curriculum_state is None:
        return

    cs = info.curriculum_state

    # Restore accuracy deques
    ftc = cs.get("first_token_correct")
    if ftc is not None:
        trainer.first_token_correct.extend(ftc)

    fwc = cs.get("full_word_correct")
    if fwc is not None:
        trainer.full_word_correct.extend(fwc)

    rl = cs.get("recent_losses")
    if rl is not None:
        trainer.recent_losses.extend(rl)

    rank_print(f"[RESUME] Applied to trainer: "
               f"first_token_correct={len(trainer.first_token_correct)}, "
               f"full_word_correct={len(trainer.full_word_correct)}")
