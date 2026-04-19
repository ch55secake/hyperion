def get_device() -> str:
    """
    Detect whether a CUDA-capable GPU is available and return the appropriate device string.

    Returns ``"cuda"`` when PyTorch is installed and ``torch.cuda.is_available()`` is
    ``True``; otherwise returns ``"cpu"``.  The CPU fallback is automatic and silent so
    the rest of the codebase never needs to handle an import error.
    """
    try:
        import torch  # type: ignore[import-untyped]  # optional dependency

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
