"""ESMFold2 complex prediction package.

Public modules:

- :mod:`esmfold2_complex.cli` — console entry point and argument parser.
- :mod:`esmfold2_complex.contracts` — shared dataclasses and metric helpers.
- :mod:`esmfold2_complex.fasta` — FASTA parsing and chain-spec construction.
- :mod:`esmfold2_complex.runtime` — device, precision, and chunk-size helpers.
- :mod:`esmfold2_complex.modeling` — local ESMFold2 model loading.
- :mod:`esmfold2_complex.inference` — seed loop and fold execution.
- :mod:`esmfold2_complex.artifacts` — HTML and matplotlib artifacts.
- :mod:`esmfold2_complex.reporting` — quality report text and CSV summary.
- :mod:`esmfold2_complex.io_paths` — output naming policy.
"""

from __future__ import annotations

__version__ = "0.1.0"
