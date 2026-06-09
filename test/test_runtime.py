"""Tests for :mod:`esmfold2_complex.runtime` and env hand-off behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from esmfold2_complex import runtime


def test_runtime_module_no_longer_exposes_configure_env() -> None:
    assert not hasattr(runtime, "configure_env")


def test_huggingface_hub_reads_hf_home_from_process_environment(
    tmp_path,
) -> None:
    pytest.importorskip("huggingface_hub")
    hf_home = tmp_path / "hf-cache"
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(
        os.environ,
        HF_HOME=str(hf_home),
        PYTHONPATH=str(repo_root / "src"),
    )
    script = (
        "from esmfold2_complex.cli import build_parser; "
        "build_parser(); "
        "import huggingface_hub.constants as c; "
        "print(c.HF_HOME)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == str(hf_home)


def test_transformers_offline_env_flows_into_hf_hub_offline_flag() -> None:
    pytest.importorskip("huggingface_hub")
    repo_root = Path(__file__).resolve().parent.parent
    env = dict(
        os.environ,
        PYTHONPATH=str(repo_root / "src"),
        TRANSFORMERS_OFFLINE="1",
    )
    script = (
        "from esmfold2_complex.cli import build_parser; "
        "build_parser(); "
        "import huggingface_hub.constants as c; "
        "print(c.HF_HUB_OFFLINE)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout.strip() == "True"
