"""Process-local compatibility for legacy AutoAWQ activation imports."""


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
