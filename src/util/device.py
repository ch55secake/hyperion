"""
Device detection utility for GPU-accelerated model training.

Provides :func:`get_device`, a single helper that checks for a CUDA-capable GPU and
returns the appropriate device string.  All model implementations (XGBoost, LightGBM,
CatBoost) and the Optuna optimiser import this function so that GPU usage is enabled
automatically on machines with a compatible GPU and falls back to CPU everywhere else.
"""


def get_device() -> str:
    """
    Detect whether a CUDA-capable GPU is available and return the appropriate device string.

    Returns ``"cuda"`` when PyTorch is installed and ``torch.cuda.is_available()`` is
    ``True``; otherwise returns ``"cpu"``.  The CPU fallback is automatic and silent —
    both missing PyTorch (``ImportError``) and CUDA driver / runtime problems
    (``Exception``) are handled gracefully so the rest of the codebase never needs to
    deal with GPU detection failures.
    """
    try:
        import torch  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]  # optional dependency

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 — covers ImportError and CUDA runtime errors
        pass
    return "cpu"
