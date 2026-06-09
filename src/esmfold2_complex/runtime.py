"""Runtime helpers for ESMFold2 complex prediction.

This module owns device / precision / chunk-size / TF32 resolution and
configuration, all of which need a real ``torch`` import. The ``torch`` module
is passed in as an argument so callers control the import order and the same
helpers stay test-friendly.
"""

from __future__ import annotations


def resolve_device(requested: str, torch) -> str:
    """Resolve ``--device auto`` to ``cuda`` or ``cpu`` based on availability."""
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_runtime_device(requested: str, gpu_id: int, torch) -> str:
    """Resolve the final torch device string, validating ``--gpu-id`` against CUDA.

    Raises a clear error when ``--device cpu`` is combined with a non-zero
    ``--gpu-id``, when CUDA is requested but unavailable, or when ``--gpu-id``
    is out of range.
    """
    base_device = resolve_device(requested, torch)
    if base_device == "cpu":
        if requested == "cpu" and gpu_id != 0:
            raise ValueError("--gpu-id can only be used with --device auto or --device cuda")
        return "cpu"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no NVIDIA GPU is available to PyTorch")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid --gpu-id {gpu_id}; available CUDA device indices are 0 to "
            f"{torch.cuda.device_count() - 1}"
        )
    return f"cuda:{gpu_id}"


def cuda_bf16_supported(torch, gpu_id: int) -> bool:
    """Return ``True`` if the given CUDA device supports bfloat16 natively."""
    check_fn = getattr(torch.cuda, "is_bf16_supported", None)
    if check_fn is not None:
        try:
            return bool(check_fn(gpu_id))
        except TypeError:
            current_device = torch.cuda.current_device()
            if current_device != gpu_id:
                torch.cuda.set_device(gpu_id)
            try:
                return bool(check_fn())
            finally:
                if current_device != gpu_id:
                    torch.cuda.set_device(current_device)

    props = torch.cuda.get_device_properties(gpu_id)
    return bool(props.major >= 8)


def resolve_esmc_precision(device: str, requested: str, torch) -> str:
    """Resolve the ESMC precision setting, respecting device and explicit choice."""
    if device == "cpu":
        if requested not in {"auto", "fp32"}:
            raise ValueError("CPU mode only supports --esmc-precision fp32")
        return "fp32"

    if requested == "auto":
        gpu_id = torch.device(device).index or 0
        return "bf16" if cuda_bf16_supported(torch, gpu_id) else "fp32"
    return requested


def resolve_chunk_size(chunk_size: int) -> int | None:
    """Map the CLI ``--chunk-size`` value to a model setting (``None`` disables it)."""
    if chunk_size < 0:
        raise ValueError("--chunk-size must be >= 0")
    return None if chunk_size == 0 else chunk_size


def configure_torch_runtime(device: str, allow_tf32: bool, torch) -> None:
    """Apply TF32 and CUDA device selection ahead of model construction."""
    if not device.startswith("cuda"):
        return

    gpu_id = torch.device(device).index or 0
    torch.cuda.set_device(gpu_id)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def describe_runtime_device(device: str, torch) -> str:
    """Render a human-readable description of the runtime device for the banner."""
    if device == "cpu":
        return "cpu"

    gpu_id = torch.device(device).index or 0
    props = torch.cuda.get_device_properties(gpu_id)
    total_gb = props.total_memory / (1024**3)
    return f"cuda:{gpu_id} ({props.name}, {total_gb:.1f} GB)"
