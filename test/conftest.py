"""Shared pytest fixtures and skip helpers for esmfold2_complex tests.

The CPU-only test suite must run without a GPU. Tests that require
ESMFold2 model loading or any torch GPU state should be guarded with
``@pytest.mark.requires_torch_gpu`` and the ``requires_torch_gpu``
fixture so they are skipped automatically when the local environment
does not provide CUDA.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "test_data"


def pytest_collection_modifyitems(config, items):
    """Skip ``requires_torch_gpu`` tests when CUDA is unavailable."""
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        return

    skip_marker = pytest.mark.skip(reason="CUDA not available in this environment")
    for item in items:
        if "requires_torch_gpu" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def wt_fasta() -> Path:
    """Absolute path to the two-chain ``wt.fasta`` fixture."""
    return TEST_DATA_DIR / "wt.fasta"


@pytest.fixture
def j13_fasta() -> Path:
    """Absolute path to the three-chain ``5j13.fasta`` fixture."""
    return TEST_DATA_DIR / "5j13.fasta"
