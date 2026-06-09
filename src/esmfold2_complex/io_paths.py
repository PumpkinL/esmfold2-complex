"""Output path resolution helpers for ESMFold2 complex prediction.

These functions encapsulate the current naming policy:

- Default root output directory: the input FASTA parent directory
- Shared output prefix: ``<root_output_dir>/<fasta_stem>``
- Per-seed output directory: ``<root_output_dir>/<fasta_stem>_seed<seed>/``
- Per-seed output: ``<fasta_stem>_seed<seed>/<fasta_stem>_seed<seed>.cif``
- Per-seed artifacts: ``<fasta_stem>_seed<seed>/<fasta_stem>_seed<seed>_*.{png,html,txt}``
- Combined CSV: ``<root_output_dir>/<fasta_stem>_seed_summary.csv``
"""

from __future__ import annotations

from pathlib import Path


def resolve_output_dir(fasta_path: Path, output_path: Path | None) -> Path:
    """Resolve the root directory that will receive all run outputs.

    ``--output`` now names the target directory directly. For migration
    convenience, a legacy mmCIF path such as ``results/out.cif`` still maps to
    its parent directory, ``results/``.
    """
    if output_path is None:
        return fasta_path.parent
    if output_path.suffix.lower() == ".cif":
        return output_path.parent
    return output_path


def resolve_output_prefix(fasta_path: Path, output_path: Path | None) -> Path:
    """Return the shared file prefix based on the FASTA stem."""
    return resolve_output_dir(fasta_path, output_path) / fasta_path.stem


def resolve_seed_output_dir(output_prefix: Path, seed: int) -> Path:
    """Return the per-seed subdirectory inside the root output directory."""
    return output_prefix.parent / f"{output_prefix.name}_seed{seed}"


def resolve_seed_output_path(output_prefix: Path, seed: int) -> Path:
    """Return the per-seed mmCIF path inside the per-seed subdirectory."""
    seed_output_dir = resolve_seed_output_dir(output_prefix, seed)
    return seed_output_dir / f"{seed_output_dir.name}.cif"


def resolve_summary_csv_path(output_prefix: Path) -> Path:
    """Return the combined CSV summary path inside the root output directory."""
    return output_prefix.with_name(f"{output_prefix.name}_seed_summary.csv")


def resolve_quality_report_path(seed_output_path: Path) -> Path:
    """Return the per-seed quality report path inside the same subdirectory."""
    return seed_output_path.parent / f"{seed_output_path.stem}_quality_report.txt"
