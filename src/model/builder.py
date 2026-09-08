"""
Model builder and LoRA configuration factory for Phi-3-mini QLoRA.

Encapsulates:
- 4-bit BitsAndBytes quantization configuration (NF4, double quant, bfloat16/float16)
- PEFT LoRA target module configuration for Phi-3 attention and FFN projections
- Tokenizer setup with right-padding and eos_token alignment
"""

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_bnb_config(
    compute_dtype: str = "bfloat16",
    load_in_4bit: bool = True,
    use_double_quant: bool = True,
    quant_type: str = "nf4",
) -> BitsAndBytesConfig:
    """
    Construct BitsAndBytesConfig for 4-bit NormalFloat quantization.

    Args:
        compute_dtype: 'bfloat16' for Colab T4 / Ampere+, 'float16' for GTX 1050 Pascal.
        load_in_4bit: Whether to load base weights in 4-bit.
        use_double_quant: Nested quantization saving ~0.37 bits/param.
        quant_type: 'nf4' (optimal for Gaussian weights) or 'fp4'.
    """
    torch_dtype = getattr(torch, compute_dtype) if isinstance(compute_dtype, str) else compute_dtype

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_compute_dtype=torch_dtype,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    """
    Construct PEFT LoraConfig targeting Phi-3 projections.

    Default target modules for Phi-3-mini-4k-instruct:
    - Attention: 'o_proj', 'qkv_proj'
    - MLP/FFN: 'gate_up_proj', 'down_proj'
    """
    if target_modules is None:
        target_modules = ["o_proj", "qkv_proj", "gate_up_proj", "down_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


def load_tokenizer(model_id: str = "microsoft/Phi-3-mini-4k-instruct") -> Any:
    """
    Load and configure Phi-3 tokenizer with correct padding semantics.

    Ensures right-padding and assigns pad_token = eos_token.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qlora_model(
    model_id: str,
    bnb_config: BitsAndBytesConfig,
    lora_config: LoraConfig,
    device_map: str | dict[str, Any] = "auto",
) -> Any:
    """
    Instantiate 4-bit quantized base model and wrap with LoRA trainable adapters.

    Prints trainable parameter summary.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
