"""Model loading and input construction for ESMFold2 complex prediction.

This module wraps the upstream ``ESMFold2Model`` loader and only performs a
small amount of local-path normalization for CCD data. When no local CCD
directory is supplied, the underlying ``esm`` package keeps full control over
how ``ccd.pkl`` is resolved.
"""

from __future__ import annotations

from pathlib import Path


def resolve_ccd_cache(model_name_or_path: str) -> Path | None:
    """Return a local CCD directory when one is bundled with the model path.

    Upstream ``esm`` already defines the fallback chain for ``ccd.pkl``:
    ``ESMCFOLD_CCD_PATH`` first, then its default Hugging Face cache/download
    behavior. This helper therefore only detects the one case the wrapper can
    resolve locally without second-guessing upstream: a user-supplied local
    model directory that already contains ``ccd.pkl``.
    """
    model_path = Path(model_name_or_path)
    if model_path.exists() and (model_path / "ccd.pkl").exists():
        return model_path
    return None


def build_structure_input(chains: list[tuple[str, str, str]]):
    """Build an ``esm`` ``StructurePredictionInput`` from FASTA chain tuples.

    The first element of each tuple is used as the protein id; only chain ID,
    header, and sequence are consumed here.
    """
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(
        sequences=[
            ProteinInput(id=chain_id, sequence=sequence)
            for chain_id, _, sequence in chains
        ]
    )


def load_esmfold2_model(
    model_name_or_path: str,
    esmc_precision: str,
    device: str,
    torch,
    local_files_only: bool = False,
):
    """Load ``ESMFold2Model`` and move it to the requested device in eval mode."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    model = ESMFold2Model.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        esmc_precision=esmc_precision,
    ).to(device).eval()
    return model


def make_input_builder(ccd_cache: Path | None = None):
    """Construct an ``ESMFold2InputBuilder`` with optional local CCD override."""
    from esm.models.esmfold2 import ESMFold2InputBuilder

    return ESMFold2InputBuilder(ccd_cache=ccd_cache)
