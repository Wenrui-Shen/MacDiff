"""Checkpoint checks plus a small CPU routing-equivalence test (when Torch exists)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "vlm_pilot"))
from awq_compat import validate_expert_index, split_expert_block_class

try:
    import torch
except ImportError:
    torch = None


class ExpertIndexTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(num_hidden_layers=2, mlp_only_layers=[1],
                                      decoder_sparse_step=1, num_experts=3)
        self.keys = {f"model.language_model.layers.0.mlp.experts.{e}.{p}.{f}": "shard"
                     for e in range(3) for p in ("gate_proj", "up_proj", "down_proj")
                     for f in ("qweight", "qzeros", "scales")}

    def test_complete_experts(self):
        self.assertEqual(validate_expert_index(self.keys, self.config), 3)

    def test_missing_scale_rejected(self):
        self.keys.pop("model.language_model.layers.0.mlp.experts.1.up_proj.scales")
        with self.assertRaisesRegex(ValueError, "missing=1"):
            validate_expert_index(self.keys, self.config)

    def test_stacked_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            validate_expert_index({"model.language_model.layers.0.mlp.experts.gate_up_proj": "shard"}, self.config)


@unittest.skipIf(torch is None, "Torch unavailable; run on the inference server")
class RoutingTests(unittest.TestCase):
    def test_sparse_matches_dense_reference(self):
        class Expert(torch.nn.Module):
            def __init__(self, config, intermediate_size):
                super().__init__()
                self.gate_proj = torch.nn.Linear(config.hidden_size, intermediate_size, bias=False)
                self.up_proj = torch.nn.Linear(config.hidden_size, intermediate_size, bias=False)
                self.down_proj = torch.nn.Linear(intermediate_size, config.hidden_size, bias=False)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        config = SimpleNamespace(hidden_size=8, num_experts=4, num_experts_per_tok=2,
                                 moe_intermediate_size=6)
        cls = split_expert_block_class(torch.nn.Module, Expert, torch)
        torch.manual_seed(17)
        block = cls(config).eval()
        self.assertIn("experts.0.gate_proj.weight", block.state_dict())
        self.assertNotIn("experts.gate_up_proj", block.state_dict())
        for shape in ((2, 5, 8), (1, 1, 8), (7, 8)):
            x = torch.randn(shape)
            actual, logits = block(x)
            flat = x.reshape(-1, 8)
            weights, indices = torch.softmax(logits, -1).topk(2, -1)
            weights = weights / weights.sum(-1, keepdim=True)
            dense_weights = torch.zeros_like(logits).scatter(1, indices, weights)
            all_outputs = torch.stack([expert(flat) for expert in block.experts], dim=1)
            expected = (all_outputs * dense_weights.unsqueeze(-1)).sum(1).reshape(shape)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
