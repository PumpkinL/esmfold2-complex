"""Seed planning, fold execution, and per-seed result extraction.

The inference layer is intentionally small: it owns the run seed list, the
actual ``builder.fold(...)`` call, and the GPU→CPU conversion of ``plddt`` /
``pae`` / ``pair_chains_iptm`` plus the per-chain pair summary construction
that the artifact and reporting layers consume.

To keep the inference loop as tight as possible (so multiple folds can be
pipelined before any disk I/O), :func:`run_single_seed` does **not** touch
the filesystem. The CLI is responsible for creating the output directory and
writing the mmCIF text plus per-seed artifacts after the fold returns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from esmfold2_complex.contracts import (
    ChainSpec,
    PairSummary,
    classify_iptm,
    classify_pae,
    classify_plddt,
    mean_optional,
    to_optional_float,
)


@dataclass(frozen=True)
class FoldedResult:
    """Normalized per-seed result produced by :func:`run_single_seed`.

    All metrics are kept in a single object so the post-processing pass
    (artifacts, quality report, CSV row) can share them without recomputing
    reductions on the same numpy arrays.
    """

    seed: int
    output_cif: Path
    output_dir: Path
    mmcif_text: str
    plddt: np.ndarray
    pae: np.ndarray | None
    pair_chains_iptm: np.ndarray | None
    mean_plddt: float
    ptm: float | None
    iptm: float | None
    pair_summaries: list[PairSummary]


AUTO_SEED_MAX = 2**31 - 1


def generate_base_seed(num_seeds: int) -> int:
    """Generate a random base seed that leaves room for a full seed range."""
    max_start = AUTO_SEED_MAX - (num_seeds - 1)
    if max_start < 0:
        raise ValueError("--num-seeds is too large to derive a bounded seed range")
    return random.SystemRandom().randrange(0, max_start + 1)


def build_run_seeds(base_seed: int | None, num_seeds: int) -> list[int]:
    """Build the seed list for a run.

    When the user omits ``--seed``, a random base seed is generated at runtime.
    Multi-seed execution expands to a consecutive range starting from that base
    seed.
    """
    if num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")
    resolved_base_seed = (
        generate_base_seed(num_seeds) if base_seed is None else base_seed
    )
    return [resolved_base_seed + offset for offset in range(num_seeds)]


def normalize_plddt(plddt: np.ndarray) -> np.ndarray:
    """Ensure pLDDT values are in 0-100 range, regardless of 0-1 vs 0-100 input."""
    plddt = np.asarray(plddt, dtype=float)
    return plddt * 100.0 if plddt.size and plddt.max() <= 1.0 else plddt


def mean_block(
    matrix: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> float:
    return float(matrix[row_start:row_end, col_start:col_end].mean())


def summarize_pair_metrics(
    chain_specs: list[ChainSpec],
    pae: np.ndarray | None,
    pair_chains_iptm: np.ndarray | None,
) -> list[PairSummary]:
    """Build per-pair iPTM / PAE summaries for the chain-pair section."""
    pair_summaries: list[PairSummary] = []
    for i, chain_i in enumerate(chain_specs):
        for j, chain_j in enumerate(chain_specs):
            if j <= i:
                continue
            pair_iptm = None
            if pair_chains_iptm is not None:
                pair_iptm = float(pair_chains_iptm[i, j])
            pair_pae = None
            if pae is not None:
                pair_pae = mean_block(
                    pae,
                    chain_i.start,
                    chain_i.end,
                    chain_j.start,
                    chain_j.end,
                )
            pair_summaries.append(
                PairSummary(
                    chain_i=chain_i.chain_id,
                    chain_j=chain_j.chain_id,
                    pair_iptm=pair_iptm,
                    mean_pae=pair_pae,
                )
            )
    return pair_summaries


def build_prediction_summary(
    seed: int,
    output_cif: Path,
    output_dir: Path,
    report_path: Path,
    chain_specs: list[ChainSpec],
    plddt: np.ndarray,
    ptm: float | None,
    iptm: float | None,
    pair_summaries: list[PairSummary],
    mean_inter_chain_pae: float | None = None,
    mean_pair_iptm: float | None = None,
) -> Any:
    """Build a :class:`PredictionSummary` for the CSV row.

    ``mean_inter_chain_pae`` and ``mean_pair_iptm`` are normally supplied
    by the caller so the same values can be reused in the quality report
    without a second ``mean_optional`` pass. When omitted, they are computed
    from ``pair_summaries`` here.
    """
    from esmfold2_complex.contracts import PredictionSummary

    if mean_inter_chain_pae is None:
        mean_inter_chain_pae = mean_optional(
            [summary.mean_pae for summary in pair_summaries if summary.mean_pae is not None]
        )
    if mean_pair_iptm is None:
        mean_pair_iptm = mean_optional(
            [summary.pair_iptm for summary in pair_summaries if summary.pair_iptm is not None]
        )
    return PredictionSummary(
        seed=seed,
        output_cif=output_cif,
        output_dir=output_dir,
        report_path=report_path,
        mean_plddt=float(plddt.mean()),
        ptm=ptm,
        iptm=iptm,
        mean_inter_chain_pae=mean_inter_chain_pae,
        mean_pair_iptm=mean_pair_iptm,
        plddt_class=classify_plddt(float(plddt.mean())),
        iptm_class=classify_iptm(iptm),
        pae_class=classify_pae(mean_inter_chain_pae),
        num_chains=len(chain_specs),
        total_residues=len(plddt),
    )


def run_single_seed(
    *,
    seed: int,
    seed_output_path: Path,
    output_dir: Path,
    model: Any,
    builder: Any,
    structure_input: Any,
    chain_specs: list[ChainSpec],
    num_loops: int,
    num_sampling_steps: int,
    torch: Any,
) -> FoldedResult:
    """Run a single ``builder.fold(...)`` call and return normalized outputs.

    The model output is normalized into :class:`FoldedResult` so the artifact
    and reporting layers only need to deal with ``numpy`` arrays and Python
    types. Scalar reductions (``mean_plddt``) and per-pair summaries are
    computed once here and reused by the report and CSV row.

    The caller owns the output directory creation; this function deliberately
    avoids touching the filesystem so multiple fold calls can be pipelined
    before any disk I/O happens.
    """
    with torch.inference_mode():
        result = builder.fold(
            model,
            structure_input,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            complex_id=seed_output_path.stem,
        )

    mmcif_text = result.complex.to_mmcif()

    plddt_tensor = result.plddt.detach()
    plddt = normalize_plddt(plddt_tensor.float().cpu().numpy())
    pae = result.pae.detach().cpu().numpy() if result.pae is not None else None
    pair_chains_iptm = (
        result.pair_chains_iptm.detach().cpu().numpy()
        if result.pair_chains_iptm is not None
        else None
    )

    # Compute mean_pLDDT from the normalised numpy array (0-100 range) so it
    # stays consistent with the values the artifact, report, and CSV writers
    # read from the same ``plddt`` array.
    mean_plddt = float(plddt.mean())
    pair_summaries = summarize_pair_metrics(chain_specs, pae, pair_chains_iptm)

    return FoldedResult(
        seed=seed,
        output_cif=seed_output_path,
        output_dir=output_dir,
        mmcif_text=mmcif_text,
        plddt=plddt,
        pae=pae,
        pair_chains_iptm=pair_chains_iptm,
        mean_plddt=mean_plddt,
        ptm=to_optional_float(result.ptm),
        iptm=to_optional_float(result.iptm),
        pair_summaries=pair_summaries,
    )
