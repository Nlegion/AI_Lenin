from src.core.settings.device import resolve_torch_device


def test_resolve_torch_device_cpu_forced():
    assert resolve_torch_device(preferred="cpu") == "cpu"


def test_resolve_torch_device_auto_prefers_cuda_when_available():
    import torch

    resolved = resolve_torch_device(preferred="auto", fallback_to_cpu=True)
    if torch.cuda.is_available():
        assert resolved == "cuda"
    else:
        assert resolved == "cpu"
