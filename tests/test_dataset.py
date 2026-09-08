"""Unit tests for dataset prompt formatting and label mappings."""

from src.data.dataset import (
    ID2LABEL,
    LABEL2ID,
    LEDGAR_PROVISION_CLASSES,
    format_instruction_prompt,
)


def test_label_mapping_integrity():
    assert len(LEDGAR_PROVISION_CLASSES) == 100, "LEDGAR must have exactly 100 classes"
    assert len(ID2LABEL) == 100
    assert len(LABEL2ID) == 100

    # Ensure bijectivity
    for idx, name in ID2LABEL.items():
        assert LABEL2ID[name] == idx


def test_format_instruction_prompt_training():
    text = "This Agreement shall be governed by California law."
    category = "Governing Laws"

    prompt = format_instruction_prompt(text, category)

    assert "<|user|>" in prompt
    assert "<|end|>" in prompt
    assert "<|assistant|>" in prompt
    assert text in prompt
    assert category in prompt
    assert prompt.endswith("<|end|>")


def test_format_instruction_prompt_inference():
    text = "Each party shall indemnify the other."

    prompt = format_instruction_prompt(text, category_name=None)

    assert "<|user|>" in prompt
    assert "<|assistant|>\n" in prompt
    assert text in prompt
    # Inference prompt must leave assistant open for generation
    assert not prompt.endswith("<|end|>")
