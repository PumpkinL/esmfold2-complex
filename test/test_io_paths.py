"""Tests for :mod:`esmfold2_complex.io_paths` output-path resolution helpers."""

from __future__ import annotations

from pathlib import Path

from esmfold2_complex.io_paths import (
    resolve_output_dir,
    resolve_output_prefix,
    resolve_quality_report_path,
    resolve_seed_output_dir,
    resolve_seed_output_path,
    resolve_summary_csv_path,
)


def test_resolve_output_dir_defaults_to_fasta_parent(tmp_path: Path) -> None:
    fasta = tmp_path / "complex.fasta"
    fasta.touch()

    resolved = resolve_output_dir(fasta, None)

    assert resolved == tmp_path


def test_resolve_output_dir_returns_explicit_directory(tmp_path: Path) -> None:
    fasta = tmp_path / "complex.fasta"
    explicit = tmp_path / "outputs"

    resolved = resolve_output_dir(fasta, explicit)

    assert resolved == explicit


def test_resolve_output_dir_maps_legacy_file_like_path_to_parent(tmp_path: Path) -> None:
    fasta = tmp_path / "complex.fasta"
    explicit = tmp_path / "outputs" / "manual.cif"

    resolved = resolve_output_dir(fasta, explicit)

    assert resolved == tmp_path / "outputs"


def test_resolve_output_prefix_uses_fasta_stem_in_default_directory(tmp_path: Path) -> None:
    fasta = tmp_path / "complex.fasta"

    resolved = resolve_output_prefix(fasta, None)

    assert resolved == tmp_path / "complex"


def test_resolve_output_prefix_uses_fasta_stem_inside_explicit_directory(tmp_path: Path) -> None:
    fasta = tmp_path / "complex.fasta"
    output_dir = tmp_path / "results"

    resolved = resolve_output_prefix(fasta, output_dir)

    assert resolved == output_dir / "complex"


def test_resolve_seed_output_dir_uses_seed_named_subdirectory(tmp_path: Path) -> None:
    output_prefix = tmp_path / "results" / "input"

    resolved = resolve_seed_output_dir(output_prefix, 1813382118)

    assert resolved == tmp_path / "results" / "input_seed1813382118"


def test_resolve_seed_output_path_includes_seed_and_cif_suffix(tmp_path: Path) -> None:
    output_prefix = tmp_path / "results" / "input"

    resolved = resolve_seed_output_path(output_prefix, 1813382118)

    assert resolved == (
        tmp_path / "results" / "input_seed1813382118" / "input_seed1813382118.cif"
    )
    assert resolved.suffix == ".cif"


def test_resolve_summary_csv_path_uses_seed_summary(tmp_path: Path) -> None:
    output_prefix = tmp_path / "results" / "input"

    resolved = resolve_summary_csv_path(output_prefix)

    assert resolved == tmp_path / "results" / "input_seed_summary.csv"


def test_resolve_quality_report_path_stays_in_seed_directory(tmp_path: Path) -> None:
    seed_output = tmp_path / "results" / "input_seed5" / "input_seed5.cif"

    resolved = resolve_quality_report_path(seed_output)

    assert resolved == (
        tmp_path / "results" / "input_seed5" / "input_seed5_quality_report.txt"
    )


def test_seed_output_report_and_summary_share_output_directory(tmp_path: Path) -> None:
    output_prefix = tmp_path / "results" / "input"

    seed_output_dir = resolve_seed_output_dir(output_prefix, 1)
    seed_output = resolve_seed_output_path(output_prefix, 1)
    report_path = resolve_quality_report_path(seed_output)
    summary_csv = resolve_summary_csv_path(output_prefix)

    assert seed_output.parent == report_path.parent == seed_output_dir
    assert summary_csv.parent == tmp_path / "results"
