"""FASTA parsing and chain-spec construction for ESMFold2 complex prediction.

This module wraps the upstream ``esm.utils.parsing.read_sequences`` reader with
the normalisation rules documented in
``.codex/doc/phase1-reference-contract.md`` (uppercasing, whitespace stripping, empty
sequence rejection) and assigns chain IDs using the canonical
A, B, C, ... ordering.
"""

from __future__ import annotations

from pathlib import Path

from esmfold2_complex.contracts import CHAIN_COLORS, ChainSpec


def chain_id_from_index(index: int) -> str:
    """Map a zero-based chain index to a letter ID using A, B, ..., Z, AA, AB, ..."""
    letters: list[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("A") + remainder))
        if value == 0:
            break
        value -= 1
    return "".join(reversed(letters))


def read_complex_fasta(fasta_path: Path) -> list[tuple[str, str, str]]:
    """Read a complex FASTA file.

    Returns a list of ``(chain_id, header, sequence)`` tuples. The chain ID is
    generated from the FASTA order; the header is the original FASTA header
    stripped of surrounding whitespace; the sequence has whitespace removed and
    is uppercased.
    """
    from esm.utils.parsing import read_sequences

    chains: list[tuple[str, str, str]] = []
    for index, entry in enumerate(read_sequences(fasta_path)):
        sequence = "".join(entry.sequence.split()).upper()
        if not sequence:
            raise ValueError(f"Empty sequence found in FASTA entry: {entry.header!r}")
        chains.append((chain_id_from_index(index), entry.header.strip(), sequence))

    if not chains:
        raise ValueError(f"No FASTA sequences found in {fasta_path}")

    return chains


def build_chain_specs(
    chains: list[tuple[str, str, str]],
) -> list[ChainSpec]:
    """Convert ``(chain_id, header, sequence)`` tuples into ``ChainSpec`` objects.

    The start/end offsets describe the chain's residue range in the concatenated
    complex, which is needed for downstream PAE / pLDDT block indexing.
    """
    chain_specs: list[ChainSpec] = []
    start = 0
    for index, (chain_id, header, sequence) in enumerate(chains):
        end = start + len(sequence)
        chain_specs.append(
            ChainSpec(
                chain_id=chain_id,
                header=header,
                sequence=sequence,
                start=start,
                end=end,
                color=CHAIN_COLORS[index % len(CHAIN_COLORS)],
            )
        )
        start = end
    return chain_specs
