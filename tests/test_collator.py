"""CPU-only check of the packed collator (search_ft/data/collate.py) against the contract transformers' packed
flash-attention path expects, and against the pre-port collator (release commit ed34885, inlined below).

For hand-made samples and, when the C++ generator extension is built, for a batch produced by
PackedSequenceDataset.__getitem__ with a tiny deterministic tokenizer:
  * input_ids / labels / position_ids are one row [1, T] with T = sum of the sequence lengths;
  * position_ids restart at 0 at every sequence start and increase by 1 inside each sequence;
  * cu_seq_lens is int32, starts at 0, is strictly increasing and ends at T; max_length is a python int equal to the
    longest sequence;
  * labels are the per-sample labels, unchanged (prompt rows -100, answer rows the token ids);
  * sequence_info carries the right start/end offsets, prompt_len and valid_first_targets;
  * input_ids, labels, position_ids, the cumulative lengths, max length and sequence_info equal the pre-port
    collator's output on the same samples.
Run from the repository root:  python tests/test_collator.py   (no GPU; the __getitem__ part needs the built
generator extension on the import path, e.g. after `python setup.py build_ext --inplace`, and is skipped otherwise)
"""
import os
import sys
from typing import Dict, List, Any

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import torch

from search_ft.data.collate import pack_batch


def _pre_port_pack_batch(samples: List[Dict]) -> Dict[str, Any]:
    """PackedSequenceDataset._pack_batch of the pre-port release (commit ed34885), verbatim."""
    all_input_ids = []
    all_labels = []
    all_position_ids = []
    cu_seqlens = [0]
    max_seqlen = 0
    all_sequence_info = []

    offset = 0
    for sample in samples:
        seq_len = sample["seq_len"]

        all_input_ids.extend(sample["input_ids"])
        all_labels.extend(sample["labels"])
        all_position_ids.extend(range(seq_len))

        offset += seq_len
        cu_seqlens.append(offset)

        max_seqlen = max(max_seqlen, seq_len)

        all_sequence_info.append({
            "row_idx": 0,
            "start_idx": cu_seqlens[-2],
            "end_idx": cu_seqlens[-1],
            "prompt_len": sample["prompt_len"],
            "valid_first_targets": sample["valid_first_targets"],
        })

    return {
        "input_ids": torch.tensor([all_input_ids], dtype=torch.long),
        "labels": torch.tensor([all_labels], dtype=torch.long),
        "position_ids": torch.tensor([all_position_ids], dtype=torch.long),
        "cu_seqlens_list": [cu_seqlens],
        "max_seqlen_list": [max_seqlen],
        "sequence_info": all_sequence_info,
        "num_sequences": len(samples),
        "_efficiency": 100.0,
    }


def make_samples(seed=0, n=5, V=1000):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(n):
        prompt_len = int(torch.randint(3, 12, (1,), generator=g))
        ans_len = int(torch.randint(1, 4, (1,), generator=g))
        ids = torch.randint(5, V, (prompt_len + ans_len,), generator=g).tolist()
        if i == n - 1:
            # instruction-style sample: prompt_len 0, labels on a middle span only, first label -100
            labels = [-100] * len(ids)
            for t in range(2, len(ids) - 1):
                labels[t] = ids[t]
            prompt_len_field, vft = 0, []
        else:
            labels = [-100] * prompt_len + ids[prompt_len:]
            prompt_len_field, vft = prompt_len, sorted({ids[prompt_len], ids[prompt_len] + 1})
        samples.append({
            "input_ids": ids,
            "labels": labels,
            "prompt_len": prompt_len_field,
            "valid_first_targets": vft,
            "seq_len": len(ids),
        })
    return samples


def check_contract(batch, samples):
    T = sum(s["seq_len"] for s in samples)
    S = len(samples)
    input_ids, labels, pos = batch["input_ids"], batch["labels"], batch["position_ids"]
    cu, max_len = batch["cu_seq_lens"], batch["max_length"]

    assert input_ids.shape == labels.shape == pos.shape == (1, T), (input_ids.shape, T)
    assert input_ids.dtype == labels.dtype == pos.dtype == torch.long
    # cu_seq_lens: int32, [S + 1], 0 ... T, strictly increasing
    assert isinstance(cu, torch.Tensor) and cu.dtype == torch.int32, cu.dtype
    assert cu.shape == (S + 1,) and int(cu[0]) == 0 and int(cu[-1]) == T, cu
    assert bool((cu[1:] > cu[:-1]).all()), cu
    assert cu.tolist() == [sum(s["seq_len"] for s in samples[:i]) for i in range(S + 1)]
    # max_length: python int (flash-attn requires an int, not a 0-d tensor), the longest sequence
    assert type(max_len) is int and max_len == max(s["seq_len"] for s in samples), (type(max_len), max_len)
    assert max_len == int(cu.diff().max())
    assert batch["num_sequences"] == S
    # per-sequence: position_ids restart at 0 and count up; ids/labels unchanged; sequence_info offsets
    for i, s in enumerate(samples):
        a, b = int(cu[i]), int(cu[i + 1])
        assert pos[0, a:b].tolist() == list(range(s["seq_len"])), (i, pos[0, a:b])
        assert input_ids[0, a:b].tolist() == s["input_ids"]
        assert labels[0, a:b].tolist() == s["labels"]
        assert labels[0, a:a + s["prompt_len"]].eq(-100).all()
        info = batch["sequence_info"][i]
        assert info["start_idx"] == a and info["end_idx"] == b
        assert info["prompt_len"] == s["prompt_len"] and info["valid_first_targets"] == s["valid_first_targets"]
    # position_ids restart exactly at the sequence starts and nowhere else
    starts = (pos[0] == 0).nonzero(as_tuple=True)[0].tolist()
    assert starts == cu[:-1].tolist(), (starts, cu)


def check_against_pre_port(batch, samples):
    pre = _pre_port_pack_batch(samples)
    assert torch.equal(batch["input_ids"], pre["input_ids"])
    assert torch.equal(batch["labels"], pre["labels"])
    assert torch.equal(batch["position_ids"], pre["position_ids"])
    assert batch["cu_seq_lens"].tolist() == pre["cu_seqlens_list"][0]
    assert batch["max_length"] == pre["max_seqlen_list"][0]
    assert batch["num_sequences"] == pre["num_sequences"]
    pre_info = [{k: v for k, v in d.items() if k != "row_idx"} for d in pre["sequence_info"]]
    assert batch["sequence_info"] == pre_info


def test_pack_batch_hand_made():
    for seed in (0, 1, 2):
        samples = make_samples(seed)
        batch = pack_batch(samples)
        check_contract(batch, samples)
        check_against_pre_port(batch, samples)
    # a single sequence
    samples = make_samples(3, n=1)
    batch = pack_batch(samples)
    check_contract(batch, samples)
    check_against_pre_port(batch, samples)
    print("test_pack_batch_hand_made: contract + pre-port equality on 3 random batches and a singleton  OK")


class _TinyTokenizer:
    """Deterministic whitespace tokenizer: every distinct whitespace-delimited piece gets the next id.
    Enough of the tokenizer interface for PackedSequenceDataset's search-sample path."""
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}

    def _ids(self, text):
        out = []
        for piece in text.split():
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            out.append(self.vocab[piece])
        return out

    def __call__(self, text, add_special_tokens=False, truncation=False, **kwargs):
        ids = self._ids(text)
        if add_special_tokens:
            ids = [self.vocab["<bos>"]] + ids
        return {"input_ids": ids}


def test_dataset_getitem():
    try:
        from search_ft.data.dataset import PackedSequenceDataset
    except ImportError as e:
        print(f"test_dataset_getitem: SKIPPED (generator extension not importable: {e})")
        return
    L = 8

    def make_ds():
        return PackedSequenceDataset(
            tokenizer=_TinyTokenizer(), batch_size=6, stage=1, max_input_size=6 * L, max_lookahead=L,
            base_lookahead=1, lookahead_step=8, seed=1234,
        )

    ds = make_ds()
    captured = []
    orig = ds._pack_batch
    ds._pack_batch = lambda samples: (captured.append(samples), orig(samples))[1]
    batch = ds[0]
    assert len(captured) == 1 and len(captured[0]) == 6, len(captured[0]) if captured else None
    samples = captured[0]
    assert all(s["prompt_len"] > 0 and s["valid_first_targets"] for s in samples)
    check_contract(batch, samples)
    check_against_pre_port(batch, samples)
    # A fresh dataset instance (a new dataloader worker, as after a resume) yields the same batch for the same index.
    # (Repeating an index on the SAME instance is not bit-identical: the name generator's Faker RNG is seeded once
    # per worker, not per batch; that is pre-existing behaviour, unchanged by the port.)
    batch2 = make_ds()[0]
    assert torch.equal(batch["input_ids"], batch2["input_ids"]) and torch.equal(batch["cu_seq_lens"], batch2["cu_seq_lens"])
    assert torch.equal(batch["labels"], batch2["labels"]) and batch["sequence_info"] == batch2["sequence_info"]
    print(f"test_dataset_getitem: {len(samples)} generated sequences, T={batch['input_ids'].shape[1]}, "
          f"max_length={batch['max_length']}  OK")


def main():
    test_pack_batch_hand_made()
    test_dataset_getitem()
    print("test_collator: ALL OK")


if __name__ == "__main__":
    main()
