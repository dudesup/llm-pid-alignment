"""Tests for pidlora/model.py's ‖B·A‖ diagnostic (design doc Section 8/9, Figure 4b).

Uses a tiny hand-built nn.Module instead of Qwen2.5-3B — get_peft_model works on any
nn.Module, so this stays CPU-only and network-free.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from pidlora.model import ADAPTER_NAME, lora_bA_frobenius_norm, lora_bA_frobenius_norms_by_layer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.k_proj(self.q_proj(x))


def build_lora_model(r=2, target_modules=("q_proj", "k_proj")):
    model = TinyModel()
    lora_config = LoraConfig(r=r, lora_alpha=4, target_modules=list(target_modules), lora_dropout=0.0, bias="none")
    return get_peft_model(model, lora_config, adapter_name=ADAPTER_NAME)


class TestLoraBAFrobeniusNorm:
    def test_zero_at_fresh_init(self):
        """Standard LoRA init: B == 0, A ~ Kaiming — so B@A == 0 for every layer."""
        model = build_lora_model()
        assert lora_bA_frobenius_norm(model) == 0.0

    def test_matches_manual_computation_after_perturbing_B(self):
        model = build_lora_model(target_modules=("q_proj",))
        layer = model.base_model.model.q_proj
        with torch.no_grad():
            layer.lora_B[ADAPTER_NAME].weight.copy_(torch.randn_like(layer.lora_B[ADAPTER_NAME].weight))
        A = layer.lora_A[ADAPTER_NAME].weight
        B = layer.lora_B[ADAPTER_NAME].weight
        expected = torch.linalg.norm(B @ A).item()

        got = lora_bA_frobenius_norm(model)
        assert abs(got - expected) < 1e-5

    def test_aggregates_across_layers_as_sqrt_sum_of_squares(self):
        model = build_lora_model(target_modules=("q_proj", "k_proj"))
        q_layer = model.base_model.model.q_proj
        k_layer = model.base_model.model.k_proj
        with torch.no_grad():
            q_layer.lora_B[ADAPTER_NAME].weight.copy_(torch.randn_like(q_layer.lora_B[ADAPTER_NAME].weight))
            k_layer.lora_B[ADAPTER_NAME].weight.copy_(torch.randn_like(k_layer.lora_B[ADAPTER_NAME].weight))

        q_norm_sq = torch.linalg.norm(q_layer.lora_B[ADAPTER_NAME].weight @ q_layer.lora_A[ADAPTER_NAME].weight).item() ** 2
        k_norm_sq = torch.linalg.norm(k_layer.lora_B[ADAPTER_NAME].weight @ k_layer.lora_A[ADAPTER_NAME].weight).item() ** 2
        expected = (q_norm_sq + k_norm_sq) ** 0.5

        got = lora_bA_frobenius_norm(model)
        assert abs(got - expected) < 1e-4

    def test_respects_adapter_name(self):
        model = build_lora_model(target_modules=("q_proj",))
        assert lora_bA_frobenius_norm(model, adapter_name="nonexistent_adapter") == 0.0


class TestLoraBAFrobeniusNormsByLayer:
    def test_keys_are_the_target_layer_names(self):
        model = build_lora_model(target_modules=("q_proj", "k_proj"))
        norms = lora_bA_frobenius_norms_by_layer(model)
        assert set(norms) == {"base_model.model.q_proj", "base_model.model.k_proj"}

    def test_per_layer_values_match_manual_computation(self):
        model = build_lora_model(target_modules=("q_proj", "k_proj"))
        q_layer = model.base_model.model.q_proj
        k_layer = model.base_model.model.k_proj
        with torch.no_grad():
            q_layer.lora_B[ADAPTER_NAME].weight.copy_(torch.randn_like(q_layer.lora_B[ADAPTER_NAME].weight))
            # k_proj left at fresh init (B == 0) — layers should differ, not just scale together

        norms = lora_bA_frobenius_norms_by_layer(model)
        expected_q = torch.linalg.norm(q_layer.lora_B[ADAPTER_NAME].weight @ q_layer.lora_A[ADAPTER_NAME].weight).item()
        assert abs(norms["base_model.model.q_proj"] - expected_q) < 1e-5
        assert norms["base_model.model.k_proj"] == 0.0

    def test_global_scalar_is_sqrt_sum_of_squares_of_per_layer_values(self):
        """Regression guard on the aggregation decision itself (Section 9, v6/v7): the
        global scalar must stay derived from the per-layer breakdown, not drift into a
        separately-maintained computation that could silently disagree with it."""
        model = build_lora_model(target_modules=("q_proj", "k_proj"))
        with torch.no_grad():
            model.base_model.model.q_proj.lora_B[ADAPTER_NAME].weight.copy_(
                torch.randn_like(model.base_model.model.q_proj.lora_B[ADAPTER_NAME].weight)
            )
            model.base_model.model.k_proj.lora_B[ADAPTER_NAME].weight.copy_(
                torch.randn_like(model.base_model.model.k_proj.lora_B[ADAPTER_NAME].weight)
            )

        per_layer = lora_bA_frobenius_norms_by_layer(model)
        expected_global = sum(v * v for v in per_layer.values()) ** 0.5
        assert abs(lora_bA_frobenius_norm(model) - expected_global) < 1e-9
