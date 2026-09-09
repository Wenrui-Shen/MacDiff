"""Process-local compatibility for legacy AutoAWQ activation imports."""

import json
from contextlib import contextmanager
from pathlib import Path


def validate_expert_index(weight_map, config):
    """Require every split expert's packed weights, zeros and scales."""
    prefix = "model.language_model.layers."
    expected = set()
    for layer in range(config.num_hidden_layers):
        if layer in config.mlp_only_layers or (layer + 1) % config.decoder_sparse_step:
            continue
        for expert in range(config.num_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for field in ("qweight", "qzeros", "scales"):
                    expected.add(f"{prefix}{layer}.mlp.experts.{expert}.{projection}.{field}")
    actual = {key for key in weight_map if ".mlp.experts." in key}
    missing, unexpected = expected - actual, actual - expected
    if not expected or missing or unexpected:
        raise ValueError(
            "Unsupported/incomplete split-expert AWQ checkpoint: "
            f"missing={len(missing)} {sorted(missing)[:3]}, "
            f"unexpected={len(unexpected)} {sorted(unexpected)[:3]}"
        )
    return len(expected) // 9


def split_expert_block_class(original, mlp_class, torch):
    """Keep the original router math, exposing experts as quantizable Linears."""
    class SplitAWQMoeBlock(original):
        def __init__(self, config):
            # Do not construct the original dense, stacked expert parameters.
            torch.nn.Module.__init__(self)
            self.hidden_size = config.hidden_size
            self.num_experts = config.num_experts
            self.top_k = config.num_experts_per_tok
            self.gate = torch.nn.Linear(self.hidden_size, self.num_experts, bias=False)
            self.experts = torch.nn.ModuleList([
                mlp_class(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ])

        def forward(self, hidden_states):
            original_shape = hidden_states.shape
            flat = hidden_states.reshape(-1, self.hidden_size)
            router_logits = self.gate(flat)
            probabilities = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
            weights, selected = torch.topk(probabilities, self.top_k, dim=-1)
            weights = (weights / weights.sum(dim=-1, keepdim=True)).to(flat.dtype)
            result = torch.zeros_like(flat)
            # Run only selected experts, without materializing every expert's
            # output for every video token. Each expert retains its AWQ layers.
            for expert_id in torch.unique(selected).tolist():
                token_ids, slots = torch.where(selected == expert_id)
                values = self.experts[expert_id](flat[token_ids]).reshape(-1, self.hidden_size)
                values = values * weights[token_ids, slots, None]
                result.index_add_(0, token_ids, values.to(flat.dtype))
            return result.reshape(original_shape), router_logits

    return SplitAWQMoeBlock


@contextmanager
def split_awq_experts(model_path, model_config):
    """Scope the split-expert constructor to local Qwen3-VL-MoE AWQ loading."""
    quant = getattr(model_config, "quantization_config", {})
    if hasattr(quant, "to_dict"):
        quant = quant.to_dict()
    if (model_config.model_type != "qwen3_vl_moe"
            or not isinstance(quant, dict) or quant.get("quant_method") != "awq"):
        yield False
        return
    if quant.get("bits") != 4 or str(quant.get("version", "")).lower() != "gemm":
        raise ValueError("Split expert compatibility currently requires AWQ GEMM 4-bit")
    index_path = Path(model_path) / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    count = validate_expert_index(index["weight_map"], model_config.text_config)
    import torch
    import transformers
    if transformers.__version__ != "4.57.1":
        raise RuntimeError("Split expert compatibility is validated for Transformers 4.57.1 only")
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe as modeling

    original = modeling.Qwen3VLMoeTextSparseMoeBlock
    replacement = split_expert_block_class(original, modeling.Qwen3VLMoeTextMLP, torch)
    print(f"[load] Using split AWQ experts; checkpoint has {count} complete experts.", flush=True)
    modeling.Qwen3VLMoeTextSparseMoeBlock = replacement
    try:
        yield True
    finally:
        modeling.Qwen3VLMoeTextSparseMoeBlock = original


def prepare_awq_imports():
    """Restore the removed class name without changing activation registries.

    Transformers 4.51.3's PytorchGELUTanh.forward called functional.gelu with
    approximate='tanh'. AutoAWQ 0.2.9 still imports that name during startup.
    No installed package files, model modules, or weights are modified.
    """
    import torch
    import transformers.activations as activations

    if hasattr(activations, "PytorchGELUTanh"):
        return False

    class PytorchGELUTanh(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.gelu(value, approximate="tanh")

    activations.PytorchGELUTanh = PytorchGELUTanh
    return True


if __name__ == "__main__":
    print("Activation compatibility applied:", prepare_awq_imports(), flush=True)
    from awq.modules.linear.gemm import TRITON_AVAILABLE

    print("AWQ Triton:", TRITON_AVAILABLE, flush=True)
