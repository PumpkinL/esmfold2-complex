"""Tests for :mod:`esmfold2_complex.inference` helper functions.

These tests cover seed planning, the pLDDT normalisation helper, the
per-chain pair summary, and the CSV-row builder. They do not exercise
``run_single_seed`` itself, which requires the real ESMFold2 model and a GPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from esmfold2_complex.contracts import ChainSpec, PairSummary, PredictionSummary
from esmfold2_complex.inference import (
    build_prediction_summary,
    build_run_seeds,
    generate_base_seed,
    normalize_plddt,
    summarize_pair_metrics,
)


def test_build_run_seeds_single_seed_returns_base_only() -> None:
    assert build_run_seeds(0, 1) == [0]


def test_build_run_seeds_expands_to_consecutive_range() -> None:
    assert build_run_seeds(7, 4) == [7, 8, 9, 10]


def test_build_run_seeds_is_deterministic() -> None:
    assert build_run_seeds(42, 5) == build_run_seeds(42, 5)


def test_build_run_seeds_uses_generated_base_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "esmfold2_complex.inference.generate_base_seed",
        lambda num_seeds: 11,
    )

    assert build_run_seeds(None, 3) == [11, 12, 13]


def test_build_run_seeds_rejects_zero() -> None:
    with pytest.raises(ValueError, match="--num-seeds"):
        build_run_seeds(0, 0)


def test_build_run_seeds_rejects_negative() -> None:
    with pytest.raises(ValueError, match="--num-seeds"):
        build_run_seeds(0, -1)


def test_generate_base_seed_stays_within_supported_range() -> None:
    seed = generate_base_seed(10)

    assert 0 <= seed <= (2**31 - 1) - 9


def test_generate_base_seed_rejects_oversized_num_seeds() -> None:
    with pytest.raises(ValueError, match="--num-seeds"):
        generate_base_seed((2**31) + 1)


def test_normalize_plddt_scales_zero_to_one_to_hundred() -> None:
    plddt = np.array([0.0, 0.5, 1.0])

    normalized = normalize_plddt(plddt)

    np.testing.assert_array_equal(normalized, np.array([0.0, 50.0, 100.0]))


def test_normalize_plddt_leaves_zero_to_hundred_unchanged() -> None:
    plddt = np.array([10.0, 50.0, 90.0, 100.0])

    normalized = normalize_plddt(plddt)

    np.testing.assert_array_equal(normalized, plddt)


def test_normalize_plddt_treats_above_one_as_already_normalised() -> None:
    plddt = np.array([1.5, 50.0, 100.0])

    normalized = normalize_plddt(plddt)

    np.testing.assert_array_equal(normalized, plddt)


def test_normalize_plddt_handles_empty_array() -> None:
    plddt = np.array([], dtype=float)

    normalized = normalize_plddt(plddt)

    assert normalized.size == 0


def test_normalize_plddt_preserves_dtype_kind() -> None:
    plddt = np.array([0.5, 0.9], dtype=np.float32)

    normalized = normalize_plddt(plddt)

    assert np.issubdtype(normalized.dtype, np.floating)


def _two_chain_specs() -> list[ChainSpec]:
    return [
        ChainSpec(
            chain_id="A",
            header="a",
            sequence="MQKLV",
            start=0,
            end=5,
            color="#000000",
        ),
        ChainSpec(
            chain_id="B",
            header="b",
            sequence="GHRSA",
            start=5,
            end=10,
            color="#111111",
        ),
    ]


def test_summarize_pair_metrics_one_entry_per_unordered_pair() -> None:
    specs = _two_chain_specs()

    summaries = summarize_pair_metrics(specs, pae=None, pair_chains_iptm=None)

    assert len(summaries) == 1
    assert summaries[0].chain_i == "A"
    assert summaries[0].chain_j == "B"


def test_summarize_pair_metrics_pulls_pair_iptm_from_matrix() -> None:
    specs = _two_chain_specs()
    pair_iptm = np.array([[0.0, 0.42], [0.42, 0.0]])

    summaries = summarize_pair_metrics(specs, pae=None, pair_chains_iptm=pair_iptm)

    assert summaries[0].pair_iptm == pytest.approx(0.42)


def test_summarize_pair_metrics_computes_block_mean_pae() -> None:
    specs = _two_chain_specs()
    pae = np.arange(100, dtype=float).reshape(10, 10)

    summaries = summarize_pair_metrics(specs, pae=pae, pair_chains_iptm=None)

    expected_block_mean = float(pae[0:5, 5:10].mean())
    assert summaries[0].mean_pae == pytest.approx(expected_block_mean)


def test_summarize_pair_metrics_three_chains_yields_three_pairs() -> None:
    specs = [
        ChainSpec(chain_id="A", header="", sequence="AAAA", start=0, end=4, color="#0"),
        ChainSpec(chain_id="B", header="", sequence="BBBB", start=4, end=8, color="#1"),
        ChainSpec(chain_id="C", header="", sequence="CCCC", start=8, end=12, color="#2"),
    ]

    summaries = summarize_pair_metrics(specs, pae=None, pair_chains_iptm=None)

    assert [(s.chain_i, s.chain_j) for s in summaries] == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]


def test_summarize_pair_metrics_returns_pair_summary_instances() -> None:
    specs = _two_chain_specs()

    summaries = summarize_pair_metrics(specs, pae=None, pair_chains_iptm=None)

    assert all(isinstance(summary, PairSummary) for summary in summaries)


def test_build_prediction_summary_reuses_supplied_aggregates() -> None:
    specs = _two_chain_specs()
    plddt = np.array([10.0, 20.0, 30.0, 40.0, 50.0] * 2)
    pair_summaries = [
        PairSummary(chain_i="A", chain_j="B", pair_iptm=0.1, mean_pae=20.0)
    ]

    summary = build_prediction_summary(
        seed=0,
        output_cif=Path("wt_pred_seed0.cif"),
        output_dir=Path("results/wt_pred_seed0"),
        report_path=Path("wt_pred_seed0_quality_report.txt"),
        chain_specs=specs,
        plddt=plddt,
        ptm=0.5,
        iptm=0.2,
        pair_summaries=pair_summaries,
        mean_inter_chain_pae=20.0,
        mean_pair_iptm=0.1,
    )

    assert isinstance(summary, PredictionSummary)
    assert summary.seed == 0
    assert summary.mean_plddt == pytest.approx(30.0)
    assert summary.ptm == 0.5
    assert summary.iptm == 0.2
    assert summary.mean_inter_chain_pae == 20.0
    assert summary.mean_pair_iptm == 0.1
    assert summary.plddt_class == "low"
    assert summary.iptm_class == "weak"
    assert summary.pae_class == "uncertain"
    assert summary.output_dir == Path("results/wt_pred_seed0")
    assert summary.num_chains == 2
    assert summary.total_residues == 10


def test_build_prediction_summary_computes_aggregates_when_omitted() -> None:
    specs = _two_chain_specs()
    plddt = np.full(10, 75.0)
    pair_summaries = [
        PairSummary(chain_i="A", chain_j="B", pair_iptm=0.7, mean_pae=15.0)
    ]

    summary = build_prediction_summary(
        seed=1,
        output_cif=Path("wt_pred_seed1.cif"),
        output_dir=Path("results/wt_pred_seed1"),
        report_path=Path("wt_pred_seed1_quality_report.txt"),
        chain_specs=specs,
        plddt=plddt,
        ptm=None,
        iptm=0.7,
        pair_summaries=pair_summaries,
    )

    assert summary.mean_inter_chain_pae == pytest.approx(15.0)
    assert summary.mean_pair_iptm == pytest.approx(0.7)
    assert summary.plddt_class == "good"
    assert summary.iptm_class == "moderate"
    assert summary.pae_class == "uncertain"


def test_build_prediction_summary_handles_empty_pair_summaries() -> None:
    specs = _two_chain_specs()
    plddt = np.array([80.0] * 10)

    summary = build_prediction_summary(
        seed=0,
        output_cif=Path("wt_pred_seed0.cif"),
        output_dir=Path("results/wt_pred_seed0"),
        report_path=Path("wt_pred_seed0_quality_report.txt"),
        chain_specs=specs,
        plddt=plddt,
        ptm=None,
        iptm=None,
        pair_summaries=[],
    )

    assert summary.mean_inter_chain_pae is None
    assert summary.mean_pair_iptm is None
    assert summary.pae_class == "unknown"
    assert summary.iptm_class == "unknown"


def test_build_prediction_summary_total_residues_equals_plddt_length() -> None:
    specs = _two_chain_specs()
    plddt = np.full(123, 50.0)

    summary = build_prediction_summary(
        seed=0,
        output_cif=Path("wt_pred_seed0.cif"),
        output_dir=Path("results/wt_pred_seed0"),
        report_path=Path("wt_pred_seed0_quality_report.txt"),
        chain_specs=specs,
        plddt=plddt,
        ptm=None,
        iptm=None,
        pair_summaries=[],
    )

    assert summary.total_residues == 123
