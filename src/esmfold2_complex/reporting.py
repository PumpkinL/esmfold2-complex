"""Reporting layer: quality-report text and combined CSV summary writers.

This module owns the two summary outputs of a run:

- The per-seed quality report text file.
- The combined multi-seed CSV summary.

The :class:`PredictionSummary` and :class:`PairSummary` dataclasses live in
``contracts``; :func:`summarize_pair_metrics` (used to build the per-chain and
chain-pair sections of the report) is in ``inference``. Reporting only
formats those structures into the documented text and CSV layouts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from esmfold2_complex.contracts import (
    ChainSpec,
    PairSummary,
    PredictionSummary,
    classify_iptm,
    classify_pae,
    classify_plddt,
    mean_optional,
    optional_metric,
)
from esmfold2_complex.inference import mean_block, summarize_pair_metrics


def write_quality_report(
    report_path: Path,
    fasta_path: Path,
    cif_path: Path,
    seed: int,
    chain_specs: list[ChainSpec],
    plddt: np.ndarray,
    mean_plddt: float,
    ptm: float | None,
    iptm: float | None,
    pae: np.ndarray | None,
    pair_chains_iptm: np.ndarray | None,
    artifact_paths: dict[str, Path],
    pair_summaries: list[PairSummary] | None = None,
    mean_inter_chain_pae: float | None = None,
    mean_pair_iptm: float | None = None,
    summary_csv_path: Path | None = None,
) -> None:
    """Write the per-seed human-readable quality report.

    ``mean_plddt`` and the per-pair summaries (plus the aggregated
    ``mean_inter_chain_pae`` and ``mean_pair_iptm``) can be supplied by the
    caller when the same values are also needed for the CSV row. When any of
    them is ``None`` the function recomputes the missing value from
    ``plddt`` / ``pae`` / ``pair_chains_iptm`` so the report text remains
    consistent with the inputs.
    """
    if pair_summaries is None:
        pair_summaries = summarize_pair_metrics(chain_specs, pae, pair_chains_iptm)
    if mean_inter_chain_pae is None:
        mean_inter_chain_pae = mean_optional(
            [summary.mean_pae for summary in pair_summaries if summary.mean_pae is not None]
        )
    if mean_pair_iptm is None:
        mean_pair_iptm = mean_optional(
            [summary.pair_iptm for summary in pair_summaries if summary.pair_iptm is not None]
        )
    pair_lines = [
        (
            f"- {summary.chain_i}-{summary.chain_j}: "
            f"pair_iPTM={optional_metric(summary.pair_iptm)}  "
            f"mean_PAE={optional_metric(summary.mean_pae, digits=2)} A"
        )
        for summary in pair_summaries
    ]

    chain_lines: list[str] = []
    for chain in chain_specs:
        chain_plddt = plddt[chain.start : chain.end]
        low_conf_count = int(np.sum(chain_plddt < 50))
        intra_pae = None
        if pae is not None:
            intra_pae = mean_block(
                pae,
                chain.start,
                chain.end,
                chain.start,
                chain.end,
            )
        chain_lines.append(
            f"- {chain.chain_id} ({chain.display_name}): "
            f"len={chain.length}  "
            f"mean_pLDDT={chain_plddt.mean():.2f}  "
            f"residues_pLDDT<50={low_conf_count}/{chain.length}  "
            f"intra_chain_mean_PAE={optional_metric(intra_pae, digits=2)} A"
        )

    report = "\n".join(
        [
            "ESMFold2 quality report",
            "======================",
            "",
            "Summary",
            f"- input FASTA: {fasta_path}",
            f"- structure file: {cif_path}",
            f"- seed: {seed}",
            f"- chains: {len(chain_specs)}",
            f"- total residues: {len(plddt)}",
            f"- mean pLDDT: {mean_plddt:.2f} ({classify_plddt(mean_plddt)})",
            f"- pTM: {optional_metric(ptm)}",
            f"- ipTM: {optional_metric(iptm)} ({classify_iptm(iptm)})",
            f"- mean pair iPTM: {optional_metric(mean_pair_iptm)}",
            (
                f"- mean inter-chain PAE: {optional_metric(mean_inter_chain_pae, digits=2)} A "
                f"({classify_pae(mean_inter_chain_pae)})"
            ),
            *(
                [f"- combined CSV summary: {summary_csv_path}"]
                if summary_csv_path is not None
                else []
            ),
            "",
            "Quick take",
            (
                "- Use pLDDT to judge local residue confidence, ipTM / pair-iPTM to judge "
                "interfaces, and PAE to judge whether chain-chain placement is stable."
            ),
            "",
            "Per-chain summary",
            *chain_lines,
            "",
            "Chain-pair summary",
            *(pair_lines if pair_lines else ["- single-chain input; no interface pairs"]),
            "",
            "Files",
            *(f"- {name}: {path}" for name, path in artifact_paths.items()),
            "",
            "Interpretation guide",
            "- pLDDT: >90 very high, 70-90 good, 50-70 cautious, <50 low confidence",
            "- iPTM / pair-iPTM: >0.80 strong, 0.60-0.80 moderate, 0.40-0.60 tentative, <0.40 weak",
            "- PAE: lower is better; <5 strong, 5-10 good, 10-20 uncertain, >20 poor",
        ]
    )
    report_path.write_text(report + "\n")


def write_summary_csv(summary_path: Path, summaries: Sequence[PredictionSummary]) -> None:
    """Write the combined CSV summary (one row per seed run)."""
    fieldnames = [
        "seed",
        "mean_plddt",
        "ptm",
        "iptm",
        "mean_pair_iptm",
        "mean_inter_chain_pae",
        "plddt_class",
        "iptm_class",
        "pae_class",
        "num_chains",
        "total_residues",
        "output_cif",
        "output_dir",
        "report_path",
    ]

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "seed": summary.seed,
                    "mean_plddt": f"{summary.mean_plddt:.3f}",
                    "ptm": optional_metric(summary.ptm),
                    "iptm": optional_metric(summary.iptm),
                    "mean_pair_iptm": optional_metric(summary.mean_pair_iptm),
                    "mean_inter_chain_pae": optional_metric(
                        summary.mean_inter_chain_pae,
                        digits=3,
                    ),
                    "plddt_class": summary.plddt_class,
                    "iptm_class": summary.iptm_class,
                    "pae_class": summary.pae_class,
                    "num_chains": summary.num_chains,
                    "total_residues": summary.total_residues,
                    "output_cif": str(summary.output_cif),
                    "output_dir": str(summary.output_dir),
                    "report_path": str(summary.report_path),
                }
            )
