"""Tests for :mod:`esmfold2_complex.modeling` CCD/model parameter forwarding."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from esmfold2_complex.modeling import (
    load_esmfold2_model,
    make_input_builder,
    resolve_ccd_cache,
)


def test_resolve_ccd_cache_returns_local_directory_when_ccd_is_present(
    tmp_path: Path,
) -> None:
    ccd_path = tmp_path / "ccd.pkl"
    ccd_path.write_text("stub")

    resolved = resolve_ccd_cache(str(tmp_path))

    assert resolved == tmp_path


def test_resolve_ccd_cache_defers_repo_ids_to_upstream() -> None:
    assert resolve_ccd_cache("biohub/ESMFold2-Fast") is None


def test_resolve_ccd_cache_defers_local_directories_without_ccd_to_upstream(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert resolve_ccd_cache(str(model_dir)) is None


def test_make_input_builder_allows_upstream_ccd_resolution(monkeypatch) -> None:
    calls: list[Path | None] = []

    class FakeESMFold2InputBuilder:
        def __init__(self, ccd_cache: Path | None = None) -> None:
            calls.append(ccd_cache)

    esm_module = types.ModuleType("esm")
    models_module = types.ModuleType("esm.models")
    esmfold2_module = types.ModuleType("esm.models.esmfold2")
    esmfold2_module.ESMFold2InputBuilder = FakeESMFold2InputBuilder

    monkeypatch.setitem(sys.modules, "esm", esm_module)
    monkeypatch.setitem(sys.modules, "esm.models", models_module)
    monkeypatch.setitem(sys.modules, "esm.models.esmfold2", esmfold2_module)

    make_input_builder()

    assert calls == [None]


def test_load_esmfold2_model_defaults_local_files_only_to_false(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeLoadedModel:
        def __init__(self) -> None:
            self.device = None
            self.eval_called = False

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    class FakeESMFold2Model:
        @classmethod
        def from_pretrained(cls, model_name_or_path: str, **kwargs):
            calls.append((model_name_or_path, kwargs))
            return FakeLoadedModel()

    transformers_module = types.ModuleType("transformers")
    models_module = types.ModuleType("transformers.models")
    esmfold2_module = types.ModuleType("transformers.models.esmfold2")
    modeling_module = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2")
    modeling_module.ESMFold2Model = FakeESMFold2Model

    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.models", models_module)
    monkeypatch.setitem(sys.modules, "transformers.models.esmfold2", esmfold2_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.esmfold2.modeling_esmfold2",
        modeling_module,
    )

    model = load_esmfold2_model(
        "biohub/ESMFold2",
        esmc_precision="bf16",
        device="cuda:0",
        torch=None,
    )

    assert model.device == "cuda:0"
    assert model.eval_called is True
    assert calls == [
        (
            "biohub/ESMFold2",
            {
                "local_files_only": False,
                "esmc_precision": "bf16",
            },
        )
    ]


def test_load_esmfold2_model_allows_explicit_local_only_override(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeLoadedModel:
        def to(self, device: str):
            return self

        def eval(self):
            return self

    class FakeESMFold2Model:
        @classmethod
        def from_pretrained(cls, model_name_or_path: str, **kwargs):
            calls.append((model_name_or_path, kwargs))
            return FakeLoadedModel()

    transformers_module = types.ModuleType("transformers")
    models_module = types.ModuleType("transformers.models")
    esmfold2_module = types.ModuleType("transformers.models.esmfold2")
    modeling_module = types.ModuleType("transformers.models.esmfold2.modeling_esmfold2")
    modeling_module.ESMFold2Model = FakeESMFold2Model

    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "transformers.models", models_module)
    monkeypatch.setitem(sys.modules, "transformers.models.esmfold2", esmfold2_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers.models.esmfold2.modeling_esmfold2",
        modeling_module,
    )

    load_esmfold2_model(
        "biohub/ESMFold2",
        esmc_precision="bf16",
        device="cuda:0",
        torch=None,
        local_files_only=True,
    )

    assert calls[0][1]["local_files_only"] is True
