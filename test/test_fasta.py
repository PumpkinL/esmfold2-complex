"""Tests for :mod:`esmfold2_complex.fasta` parsing and chain-spec construction.

These tests exercise the FASTA reader against the two fixtures in
``test_data/`` and against small synthetic FASTA files written into a
``tmp_path``. They do not require GPU or any model state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from esmfold2_complex.contracts import ChainSpec
from esmfold2_complex.fasta import (
    build_chain_specs,
    chain_id_from_index,
    read_complex_fasta,
)


def test_chain_id_from_index_first_26_are_a_to_z() -> None:
    expected = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for index, letter in enumerate(expected):
        assert chain_id_from_index(index) == letter


def test_chain_id_from_index_uses_aa_ab_after_z() -> None:
    assert chain_id_from_index(26) == "AA"
    assert chain_id_from_index(27) == "AB"


def test_chain_id_from_index_handles_far_indices() -> None:
    assert chain_id_from_index(51) == "AZ"
    assert chain_id_from_index(52) == "BA"


def test_read_complex_fasta_reads_wt_fixture(wt_fasta: Path) -> None:
    chains = read_complex_fasta(wt_fasta)

    assert [chain[0] for chain in chains] == ["A", "B"]
    assert chains[0][1] == "A"
    assert chains[1][1] == "B"
    assert chains[0][2].isupper()
    assert chains[1][2].isupper()
    assert all(chains[i][2] == chains[i][2].replace(" ", "") for i in range(2))


def test_read_complex_fasta_reads_j13_fixture(j13_fasta: Path) -> None:
    chains = read_complex_fasta(j13_fasta)

    assert [chain[0] for chain in chains] == ["A", "B", "C"]


def test_read_complex_fasta_uppercases_lowercase_sequences(tmp_path: Path) -> None:
    fasta = tmp_path / "lower.fasta"
    fasta.write_text(">a\nmqklv\n")

    chains = read_complex_fasta(fasta)

    assert chains[0][2] == "MQKLV"


def test_read_complex_fasta_strips_internal_whitespace(tmp_path: Path) -> None:
    fasta = tmp_path / "whitespace.fasta"
    fasta.write_text(">a\nMQK LV\nL\n")

    chains = read_complex_fasta(fasta)

    assert chains[0][2] == "MQKLVL"


def test_read_complex_fasta_strips_header_whitespace(tmp_path: Path) -> None:
    fasta = tmp_path / "header.fasta"
    fasta.write_text(">   chain one description   \nMQKLV\n")

    chains = read_complex_fasta(fasta)

    assert chains[0][1] == "chain one description"


def test_read_complex_fasta_raises_on_empty_file(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.write_text("")

    with pytest.raises(ValueError):
        read_complex_fasta(fasta)


def test_read_complex_fasta_raises_on_empty_sequence_record(tmp_path: Path) -> None:
    fasta = tmp_path / "blank_record.fasta"
    fasta.write_text(">header\n\n>other\nMQKLV\n")

    with pytest.raises(ValueError, match="Empty sequence"):
        read_complex_fasta(fasta)


def test_build_chain_specs_assigns_contiguous_offsets() -> None:
    chains = [
        ("A", "first", "MQKLV"),
        ("B", "second", "GHRSA"),
    ]

    specs = build_chain_specs(chains)

    assert specs[0].start == 0
    assert specs[0].end == 5
    assert specs[1].start == 5
    assert specs[1].end == 10


def test_build_chain_specs_preserves_chain_id_and_header() -> None:
    chains = [
        ("A", "alpha chain", "MQKLV"),
        ("B", "beta chain", "GHRSA"),
    ]

    specs = build_chain_specs(chains)

    assert [spec.chain_id for spec in specs] == ["A", "B"]
    assert [spec.header for spec in specs] == ["alpha chain", "beta chain"]


def test_build_chain_specs_length_property_matches_sequence() -> None:
    chains = [
        ("A", "first", "MQKLV"),
        ("B", "second", "GHRSA"),
    ]

    specs = build_chain_specs(chains)

    assert [spec.length for spec in specs] == [5, 5]


def test_build_chain_specs_color_cycles_palette() -> None:
    """The color assignment uses ``CHAIN_COLORS[index % len(...)]``."""
    from esmfold2_complex.contracts import CHAIN_COLORS

    chains = [(chain_id_from_index(i), f"h{i}", "A") for i in range(len(CHAIN_COLORS) + 2)]

    specs = build_chain_specs(chains)

    for index, spec in enumerate(specs):
        assert spec.color == CHAIN_COLORS[index % len(CHAIN_COLORS)]


def test_build_chain_specs_display_name_falls_back_to_chain_id() -> None:
    chains = [("A", "", "MQKLV")]

    specs = build_chain_specs(chains)

    assert specs[0].display_name == "A"


def test_build_chain_specs_display_name_uses_header_when_present() -> None:
    chains = [("A", "alpha", "MQKLV")]

    specs = build_chain_specs(chains)

    assert specs[0].display_name == "alpha"


def test_build_chain_specs_returns_chain_spec_instances() -> None:
    chains = [("A", "alpha", "MQKLV")]

    specs = build_chain_specs(chains)

    assert all(isinstance(spec, ChainSpec) for spec in specs)


def test_wt_fixture_produces_expected_total_length(wt_fasta: Path) -> None:
    chains = read_complex_fasta(wt_fasta)
    specs = build_chain_specs(chains)

    total_residues = sum(spec.length for spec in specs)

    assert total_residues == sum(len(sequence) for _, _, sequence in chains)
    assert specs[-1].end == total_residues
