"""Shared dataclasses and metric utilities for ESMFold2 complex prediction.

This module centralises the data structures that flow between the parsing,
inference, artifact, and reporting layers. It also owns the metric
classification thresholds and small formatting helpers, so the rest of the
package can avoid duplicating the same magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHAIN_COLORS: list[str] = [
    "#2166AC",
    "#F4A9C4",
    "#E6550D",
    "#31A354",
    "#756BB1",
    "#4A90E2",
    "#F5A623",
    "#50E3C2",
    "#B15928",
    "#FB9A99",
]

PLDDT_COLORS: dict[str, str] = {
    "very_low": "#FF7D45",
    "low": "#FFDB13",
    "confident": "#65CBF3",
    "very_high": "#0053D6",
}


@dataclass(frozen=True)
class ChainSpec:
    """Metadata for a single chain in the complex."""

    chain_id: str
    header: str
    sequence: str
    start: int
    end: int
    color: str

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def display_name(self) -> str:
        return self.header or self.chain_id


@dataclass(frozen=True)
class PairSummary:
    """Pairwise iPTM / PAE summary for one chain pair."""

    chain_i: str
    chain_j: str
    pair_iptm: float | None
    mean_pae: float | None


@dataclass(frozen=True)
class PredictionSummary:
    """Combined per-seed metrics used for the CSV summary row."""

    seed: int
    output_cif: Path
    output_dir: Path
    report_path: Path
    mean_plddt: float
    ptm: float | None
    iptm: float | None
    mean_inter_chain_pae: float | None
    mean_pair_iptm: float | None
    plddt_class: str
    iptm_class: str
    pae_class: str
    num_chains: int
    total_residues: int


def plddt_hex(score: float) -> str:
    """Map a per-residue pLDDT score to its visualization color."""
    if score >= 90:
        return PLDDT_COLORS["very_high"]
    if score >= 70:
        return PLDDT_COLORS["confident"]
    if score >= 50:
        return PLDDT_COLORS["low"]
    return PLDDT_COLORS["very_low"]


def optional_metric(value: float | None, digits: int = 3) -> str:
    """Format an optional metric, using ``"NA"`` when the value is missing."""
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def to_optional_float(value) -> float | None:
    """Coerce a tensor/scalar/number into an optional Python float."""
    if value is None:
        return None
    return float(value)


def mean_optional(values: list[float]) -> float | None:
    """Mean of a list, returning ``None`` for an empty input."""
    if not values:
        return None
    return float(np.mean(values))


def classify_plddt(mean_plddt: float) -> str:
    """Bucket the mean pLDDT into a human-readable category."""
    if mean_plddt >= 90:
        return "very high"
    if mean_plddt >= 70:
        return "good"
    if mean_plddt >= 50:
        return "mixed"
    return "low"


def classify_iptm(iptm: float | None) -> str:
    """Bucket ipTM into a confidence category, or ``"unknown"`` when missing."""
    if iptm is None:
        return "unknown"
    if iptm >= 0.80:
        return "strong"
    if iptm >= 0.60:
        return "moderate"
    if iptm >= 0.40:
        return "tentative"
    return "weak"


def classify_pae(mean_pae: float | None) -> str:
    """Bucket mean PAE into a confidence category, or ``"unknown"`` when missing."""
    if mean_pae is None:
        return "unknown"
    if mean_pae <= 5:
        return "strong"
    if mean_pae <= 10:
        return "good"
    if mean_pae <= 20:
        return "uncertain"
    return "poor"
