from pathlib import Path
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from inference_utils import decode_generated_continuations, unpack_generation_inputs


class FakeTokenizer:
    def batch_decode(self, rows, skip_special_tokens=True):
        mapping = {7: "neutral", 8: "entailment"}
        return [" ".join(mapping.get(int(token), str(int(token))) for token in row) for row in rows]


def test_early_eos_decodes_only_continuation():
    prompt = torch.arange(20).reshape(1, 20)
    sequences = torch.cat([prompt, torch.tensor([[7, 8]])], dim=1)
    assert decode_generated_continuations(sequences, 20, FakeTokenizer()) == ["neutral entailment"]


def test_generate_output_is_supported():
    sequences = torch.tensor([[8, 8, 7]])
    generated = SimpleNamespace(sequences=sequences)
    assert decode_generated_continuations(generated, 2, FakeTokenizer()) == ["neutral"]


def test_prompt_label_cannot_pollute_prediction():
    sequences = torch.tensor([[8, 8, 7]])
    decoded = decode_generated_continuations(sequences, 2, FakeTokenizer())[0]
    assert decoded == "neutral"


def test_batch_slicing_matches_individual_slicing():
    batch = torch.tensor([[1, 2, 7], [3, 4, 8]])
    together = decode_generated_continuations(batch, 2, FakeTokenizer())
    apart = [decode_generated_continuations(row.unsqueeze(0), 2, FakeTokenizer())[0] for row in batch]
    assert together == apart


def test_unpack_generation_inputs_supports_tensor_and_mapping():
    tensor = torch.tensor([[1, 2, 3]])
    tensor_inputs, tensor_width = unpack_generation_inputs(tensor)
    assert set(tensor_inputs) == {"input_ids"}
    assert tensor_inputs["input_ids"] is tensor
    assert tensor_width == 3

    mapping = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    mapped_inputs, mapped_width = unpack_generation_inputs(mapping)
    assert set(mapped_inputs) == {"input_ids", "attention_mask"}
    assert mapped_width == 2
