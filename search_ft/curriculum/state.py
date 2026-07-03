import json
import os
import re
from typing import Dict, Any, List, Optional

from ..distributed.utils import rank_print


# --- Curriculum state save/restore ---


def save_curriculum_state(
    dirpath: str,
    stage: int,
    stage_start_step: int,
    wall_time_offset: float = 0.0,
    first_token_correct=None,
    full_word_correct=None,
    recent_losses=None,
    samples_this_stage: int = 0,
    tokens_this_stage: int = 0,
) -> None:
    try:
        os.makedirs(dirpath, exist_ok=True)
        fp = os.path.join(dirpath, "curriculum_state.json")
        data = {
            "stage": int(stage),
            "stage_start_step": int(stage_start_step),
            "wall_time_offset": float(wall_time_offset),
            "samples_this_stage": int(samples_this_stage),
            "tokens_this_stage": int(tokens_this_stage),
        }
        if first_token_correct is not None:
            data["first_token_correct"] = list(first_token_correct)
        if full_word_correct is not None:
            data["full_word_correct"] = list(full_word_correct)
        if recent_losses is not None:
            data["recent_losses"] = list(recent_losses)
        with open(fp, "w") as f:
            json.dump(data, f)
    except Exception as e:
        rank_print(f"[CURRICULUM][WARN] Failed to save state: {e}")


def restore_curriculum_state(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    if not checkpoint_path:
        return None
    fp = os.path.join(checkpoint_path, "curriculum_state.json")
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp) as f:
            state = json.load(f)
        rank_print(f"[CURRICULUM] Restored state from {fp}: stage={state.get('stage')}, "
                   f"stage_start_step={state.get('stage_start_step')}")
        return state
    except Exception as e:
        rank_print(f"[CURRICULUM][WARN] Failed to restore from {fp}: {e}")
        return None


def get_checkpoint_step(checkpoint_path: str) -> int:
    if not checkpoint_path:
        return 0

    # Try path pattern
    match = re.search(r'checkpoint-(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))

    # Try trainer_state.json
    ts_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.isfile(ts_path):
        try:
            with open(ts_path) as f:
                ts = json.load(f)
            step = int(ts.get("global_step", 0))
            rank_print(f"[CURRICULUM] Extracted step {step} from trainer_state.json")
            return step
        except Exception:
            pass

    return 0


# --- Run config save/restore ---


def save_run_config(
    dirpath: str,
    effective_batch_size: int,
    per_device_batch_size: int,
    grad_accum: int,
    world_size: int,
) -> None:
    try:
        os.makedirs(dirpath, exist_ok=True)
        data = {
            "effective_batch_size": int(effective_batch_size),
            "per_device_batch_size": int(per_device_batch_size),
            "grad_accum": int(grad_accum),
            "world_size": int(world_size),
        }
        atomic_write_json(os.path.join(dirpath, "run_config.json"), data)
    except Exception as e:
        rank_print(f"[CONFIG][WARN] Failed to save run config: {e}")


def load_run_config(dirpath: str) -> Optional[Dict[str, Any]]:
    fp = os.path.join(dirpath, "run_config.json")
    if not os.path.isfile(fp):
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception:
        return None


# --- Loss history (JSONL) ---


def append_to_jsonl(dirpath: str, entries: List[Dict]) -> None:
    path = os.path.join(dirpath, "loss_history.jsonl")
    with open(path, "a") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(',', ':')) + "\n")
        f.flush()


def load_from_jsonl(dirpath: str, max_step: Optional[int] = None) -> Optional[List[Dict]]:
    path = os.path.join(dirpath, "loss_history.jsonl")
    if not os.path.isfile(path):
        return None
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if max_step is None or entry.get("step", 0) <= max_step:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries if entries else None


def rewrite_jsonl(dirpath: str, entries: List[Dict]) -> None:
    path = os.path.join(dirpath, "loss_history.jsonl")
    tmp_path = path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(',', ':')) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, path)


# --- Atomic file operations ---


def atomic_write_json(path: str, data: Any) -> None:
    tmp_path = path + f".tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
