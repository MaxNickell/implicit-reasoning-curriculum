import multiprocessing
import random
from typing import Dict, List, Any, Optional, Set

import torch
from torch.utils.data import Dataset

import generator as _gen_module

from ..curriculum.lookahead import alpha_for_lookahead
from ..distributed.utils import get_rank, rank_print
from .collate import pack_batch
from .nl_generator import NaturalLanguageGraphGenerator

INSTRUCTION_DATASET = "allenai/Dolci-Instruct-SFT"


def _tokenize_leading_space(tokenizer, s: str) -> List[int]:
    return tokenizer(" " + s, add_special_tokens=False)["input_ids"]


class PackedSequenceDataset(Dataset):

    INSTRUCTION_MAX_LEN = 2048

    def __init__(
        self,
        tokenizer,
        batch_size: int = 64,
        stage: int = 1,
        max_input_size: int = 576,
        max_lookahead: int = 96,
        base_lookahead: int = 1,
        lookahead_step: int = 1,
        reserved_inputs: Optional[Set[str]] = None,
        seed: Optional[int] = None,
        resume_step: int = 0,
        epoch_size: int = 10_000_000,
        fixed_vocab: bool = False,
        instruction_mix_ratio: float = 0.0,
    ):
        # No flash-attn check here: the model load with attn_implementation=flash_attention_{2,3}
        # has already failed if neither flash_attn nor flash_attn_3 is importable.
        self.tokenizer = tokenizer
        self.target_samples_per_batch = batch_size
        self._stage = multiprocessing.Value('i', stage)
        self.max_input_size = max_input_size
        self.max_lookahead = max_lookahead
        self.base_lookahead = base_lookahead
        self.lookahead_step = lookahead_step
        self.reserved_inputs = reserved_inputs or set()
        self.seed = seed
        self.resume_step = resume_step
        self.epoch_size = epoch_size
        self.fixed_vocab = fixed_vocab
        self.instruction_mix_ratio = instruction_mix_ratio
        self.use_chat_template = instruction_mix_ratio > 0

        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Per-worker state (lazily initialized)
        self._worker_generator = None
        self._worker_rng = None
        self._worker_id = None
        self._instruction_iterator = None

        rank_print(f"[DATASET] Packed Map-style | batch_size={batch_size} | "
                   f"instruction_mix={instruction_mix_ratio:.0%}")

    @property
    def stage(self):
        return self._stage.value

    @stage.setter
    def stage(self, value):
        self._stage.value = value

    def __len__(self):
        return self.epoch_size

    def _stage_target_lookahead(self) -> int:
        target_L = self.base_lookahead + (self.stage - 1) * self.lookahead_step
        return min(target_L, self.max_lookahead)

    def _stage_alpha(self) -> float:
        target_L = self._stage_target_lookahead()
        return alpha_for_lookahead(target_L, self.max_input_size)

    def _is_final_stage(self) -> bool:
        return self._stage_target_lookahead() >= self.max_lookahead

    def _get_worker_state(self, idx: int):
        worker_info = torch.utils.data.get_worker_info()
        current_worker_id = worker_info.id if worker_info else 0

        if self._worker_generator is None or self._worker_id != current_worker_id:
            rank = get_rank()
            worker_seed = (self.seed or 0) + rank * 9973 + current_worker_id * 7919

            self._worker_generator = NaturalLanguageGraphGenerator(
                self.max_input_size,
                seed=worker_seed
            )
            self._worker_rng = random.Random(worker_seed)
            self._worker_id = current_worker_id

        batch_seed = ((self.seed or 0) +
                      idx * 104729 +
                      self._worker_id * 7919 +
                      get_rank() * 999983)
        self._worker_rng.seed(batch_seed)
        _gen_module.set_seed(batch_seed & 0x7FFFFFFF)
        random.seed(batch_seed)

        return self._worker_generator, self._worker_rng

    def _get_instruction_iterator(self):
        if self._instruction_iterator is None:
            import os
            os.environ.setdefault("HF_DATASETS_DOWNLOAD_TIMEOUT", "120")
            from datasets import load_dataset

            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info else 0

            ds = load_dataset(
                INSTRUCTION_DATASET,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            seed = (self.seed or 0) + 77777 + worker_id
            ds = ds.shuffle(seed=seed, buffer_size=1000)
            self._instruction_iterator = iter(ds)
        return self._instruction_iterator

    def _generate_one_search_sample(self, rng: random.Random) -> Optional[Dict[str, Any]]:
        gen = self._worker_generator
        alpha = self._stage_alpha()

        ex = None
        for _ in range(100):
            batch = gen.generate_batch(
                batch_size=1,
                reserved_inputs=self.reserved_inputs,
                alpha=alpha,
                max_lookahead=self.max_lookahead,
                fixed_vocab=self.fixed_vocab,
            )
            if batch and batch[0] and batch[0].output_texts:
                candidate = batch[0]
                if candidate.input_text not in self.reserved_inputs:
                    ex = candidate
                    break

        if ex is None:
            return None

        prompt_text = ex.input_text
        chosen = rng.choice(ex.output_texts)

        if self.use_chat_template:
            msgs = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": chosen},
            ]
            full_ids = self.tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_ids = self.tokenizer.apply_chat_template(
                msgs[:1], tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = full_ids[:len(prefix_ids)]
            ans_ids = full_ids[len(prefix_ids):]
            input_ids = full_ids
            labels = [-100] * len(prompt_ids) + list(ans_ids)
        else:
            prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True, truncation=False)["input_ids"]
            ans_ids = _tokenize_leading_space(self.tokenizer, chosen)
            end_ids = self.tokenizer(". ", add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + ans_ids + end_ids
            labels = [-100] * len(prompt_ids) + ans_ids + end_ids

        first_union = sorted({
            tokens[0]
            for tokens in (_tokenize_leading_space(self.tokenizer, a) for a in ex.output_texts)
            if tokens
        })

        return {
            "input_ids": input_ids,
            "labels": labels,
            "prompt_len": len(prompt_ids),
            "valid_first_targets": first_union,
            "seq_len": len(input_ids),
        }

    def _generate_one_instruction_sample(self) -> Optional[Dict[str, Any]]:
        instruction_iter = self._get_instruction_iterator()
        max_len = self.INSTRUCTION_MAX_LEN

        for _ in range(20):
            try:
                example = next(instruction_iter)
            except StopIteration:
                self._instruction_iterator = None
                instruction_iter = self._get_instruction_iterator()
                example = next(instruction_iter)
            except Exception:
                import time
                time.sleep(1)
                continue

            if "messages" not in example or not example["messages"]:
                continue

            messages = example["messages"]
            if len(messages) < 2:
                continue

            msgs = [{"role": m["role"], "content": m["content"]} for m in messages
                    if m.get("role") and m.get("content")]
            if len(msgs) < 2:
                continue

            try:
                full_ids = self.tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=False,
                    enable_thinking=False,
                    truncation=True, max_length=max_len,
                )
            except Exception:
                continue

            if len(full_ids) < 4:
                continue

            # Labels: only train on assistant response tokens
            labels = [-100] * len(full_ids)
            for i in range(len(msgs)):
                if msgs[i]["role"] != "assistant":
                    continue
                if i > 0:
                    prefix_ids = self.tokenizer.apply_chat_template(
                        msgs[:i], tokenize=True, add_generation_prompt=True,
                        enable_thinking=False,
                        truncation=True, max_length=max_len,
                    )
                else:
                    prefix_ids = []
                through_ids = self.tokenizer.apply_chat_template(
                    msgs[:i + 1], tokenize=True, add_generation_prompt=False,
                    enable_thinking=False,
                    truncation=True, max_length=max_len,
                )
                tok_start = len(prefix_ids)
                tok_end = min(len(through_ids), len(full_ids))
                for t in range(tok_start, tok_end):
                    labels[t] = full_ids[t]

            if all(l == -100 for l in labels):
                continue

            return {
                "input_ids": full_ids,
                "labels": labels,
                "seq_len": len(full_ids),
                "prompt_len": 0,
                "valid_first_targets": [],
            }

        return None

    def _pack_batch(self, samples: List[Dict]) -> Dict[str, Any]:
        """One flat [1, T] micro-batch with per-sequence position_ids and int32 cu_seq_lens (collate.pack_batch)."""
        return pack_batch(samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        effective_idx = idx + self.resume_step
        gen, rng = self._get_worker_state(effective_idx)

        target_samples = self.target_samples_per_batch

        if self.instruction_mix_ratio > 0:
            n_instruction = int(target_samples * self.instruction_mix_ratio)
            n_search = target_samples - n_instruction
        else:
            n_instruction = 0
            n_search = target_samples

        all_samples = []

        # Generate search samples
        max_attempts = n_search * 10
        attempts = 0
        while len(all_samples) < n_search and attempts < max_attempts:
            attempts += 1
            sample = self._generate_one_search_sample(rng)
            if sample is None:
                continue
            all_samples.append(sample)

        # Generate instruction samples
        for _ in range(n_instruction):
            sample = self._generate_one_instruction_sample()
            if sample is not None:
                all_samples.append(sample)

        if not all_samples:
            raise RuntimeError(
                f"[DATASET] Failed to generate any samples for idx={idx}. "
                f"Stage={self.stage}, alpha={self._stage_alpha():.3f}"
            )

        return self._pack_batch(all_samples)
