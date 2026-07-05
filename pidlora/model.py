"""4-bit Qwen2.5-3B-Instruct + LoRA setup, and the alpha-scaling helper used to set the
adapter's static output scale at load time (Section 7: `base_model.scaling[adapter] =
alpha / r`)."""
from __future__ import annotations

import contextlib
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ADAPTER_NAME = "default"


def load_model_and_tokenizer(
    model_name: str,
    lora_r: int,
    lora_alpha: float,
    target_modules: tuple[str, ...],
    device_map: str = "auto",
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    # Standard QLoRA step: casts layer norms to fp32 (bf16 layer norms are a known
    # source of instability under 4-bit) and calls enable_input_require_grads so LoRA
    # still gets gradients if gradient checkpointing is ever turned on for VRAM
    # headroom — without it that combination silently trains nothing.
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config, adapter_name=ADAPTER_NAME)
    return model, tokenizer


def _lora_layers(model):
    for module in model.modules():
        if hasattr(module, "scaling") and isinstance(getattr(module, "scaling"), dict):
            yield module


def set_lora_scaling(model, alpha: float, r: int, adapter_name: str = ADAPTER_NAME) -> None:
    """Set the adapter output scale on every LoRA layer: scaling = alpha / r.

    PEFT stores `scaling` as a per-layer {adapter_name: float} dict, not a single
    model-level attribute — this walks every LoRA layer so the doc's simplified
    `base_model.scaling[adapter] = alpha_new / r` actually takes effect everywhere.
    """
    new_scale = alpha / r
    for module in _lora_layers(model):
        if adapter_name in module.scaling:
            module.scaling[adapter_name] = new_scale


def get_lora_scaling(model, adapter_name: str = ADAPTER_NAME) -> Optional[float]:
    for module in _lora_layers(model):
        if adapter_name in module.scaling:
            return module.scaling[adapter_name]
    return None


@contextlib.contextmanager
def adapter_disabled(model):
    """Disable the LoRA adapter entirely — used to compute frozen base-model reference
    log-probs (Section 5) without loading a second copy of the model."""
    with model.disable_adapter():
        yield
