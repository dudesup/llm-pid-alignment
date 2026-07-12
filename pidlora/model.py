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
    use_4bit: bool = True,
):
    """use_4bit=False is for CPU smoke tests only (bitsandbytes 4-bit needs CUDA) — real
    T4 runs must keep it True, it's what the Section 5 VRAM budget assumes."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            dtype=torch.bfloat16,
        )
        # Standard QLoRA step: casts layer norms to fp32 (bf16 layer norms are a known
        # source of instability under 4-bit) and calls enable_input_require_grads so
        # LoRA still gets gradients if gradient checkpointing is ever turned on for
        # VRAM headroom — without it that combination silently trains nothing.
        base_model = prepare_model_for_kbit_training(base_model)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=torch.bfloat16,
        )

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


def _lora_layers_named(model):
    for name, module in model.named_modules():
        if hasattr(module, "scaling") and isinstance(getattr(module, "scaling"), dict):
            yield name, module


@torch.no_grad()
def lora_bA_frobenius_norms_by_layer(model, adapter_name: str = ADAPTER_NAME) -> dict[str, float]:
    """Per-layer ||B_l @ A_l||_F (design doc Section 9, v6): the suppression-vs-compensation
    tug-of-war (Section 8) need not be uniform across layers, and the global scalar
    (lora_bA_frobenius_norm) can read flat purely from cross-layer cancellation in the sum
    — this is the layer-resolved diagnostic supplement, logged at the 200-step cadence.

    Uses ||BA||_F^2 = tr((B^T B)(A A^T)) instead of materializing the full
    (out_features x in_features) delta matrix — both factors are r x r, so this is
    O(r^2 * (in+out)) per layer instead of O(r * in * out). Exact, not an approximation.
    """
    norms = {}
    for name, module in _lora_layers_named(model):
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue
        A = module.lora_A[adapter_name].weight  # (r, in_features)
        B = module.lora_B[adapter_name].weight  # (out_features, r)
        BtB = B.T @ B  # (r, r), symmetric
        AAt = A @ A.T  # (r, r), symmetric
        norm_sq = torch.sum(BtB * AAt).item()  # tr(BtB @ AAt) via elementwise sum (both symmetric)
        norms[name] = norm_sq ** 0.5
    return norms


def lora_bA_frobenius_norm(model, adapter_name: str = ADAPTER_NAME) -> float:
    """sqrt(sum over LoRA layers of ||B_l @ A_l||_F^2) — treats every layer's B@A as a
    block of one big block-diagonal matrix. Tracks adapter weight-space growth
    independent of the alpha/r output scale (design doc Section 8/9: the
    suppression-vs-compensation tug-of-war diagnostic, Figure 4b). Global counterpart of
    lora_bA_frobenius_norms_by_layer — matches global alpha's scale."""
    norms = lora_bA_frobenius_norms_by_layer(model, adapter_name)
    return sum(n * n for n in norms.values()) ** 0.5


@contextlib.contextmanager
def adapter_disabled(model):
    """Disable the LoRA adapter entirely — used to compute frozen base-model reference
    log-probs (Section 5) without loading a second copy of the model."""
    with model.disable_adapter():
        yield
