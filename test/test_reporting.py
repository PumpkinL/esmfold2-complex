"""Tests for :mod:`esmfold2_complex.reporting` CSV summary output."""

from __future__ import annotations

import csv
from pathlib import Path

from esmfold2_complex.contracts import PredictionSummary
from esmfold2_complex.reporting import write_summary_csv


def test_write_summary_csv_uses_output_dir_column(tmp_path: Path) -> None:
    summary_path = tmp_path / "input_seed_summary.csv"
    summary = PredictionSummary(
        seed=5,
        output_cif=Path("results/input_seed5/input_seed5.cif"),
        output_dir=Path("results/input_seed5"),
        report_path=Path("results/input_seed5/input_seed5_quality_report.txt"),
        mean_plddt=91.2,
        ptm=0.8,
        iptm=0.7,
        mean_inter_chain_pae=4.5,
        mean_pair_iptm=0.75,
        plddt_class="very high",
        iptm_class="moderate",
        pae_class="strong",
        num_chains=2,
        total_residues=123,
    )

    write_summary_csv(summary_path, [summary])

    with summary_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames is not None
    assert "output_dir" in reader.fieldnames
    assert "artifact_dir" not in reader.fieldnames
    assert rows == [
        {
            "seed": "5",
            "mean_plddt": "91.200",
            "ptm": "0.800",
            "iptm": "0.700",
            "mean_pair_iptm": "0.750",
            "mean_inter_chain_pae": "4.500",
            "plddt_class": "very high",
            "iptm_class": "moderate",
            "pae_class": "strong",
            "num_chains": "2",
            "total_residues": "123",
            "output_cif": "results/input_seed5/input_seed5.cif",
            "output_dir": "results/input_seed5",
            "report_path": "results/input_seed5/input_seed5_quality_report.txt",
        }
    ]
