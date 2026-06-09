# esmfold2-complex

`esmfold2-complex` is a lightweight wrapper around ESMFold2 for fast protein
complex prediction from FASTA input. Give it a protein complex FASTA file, and
the CLI writes predicted structures together with confidence plots and quality
reports for each seed.

## Overview

- reads one FASTA file per run, with one FASTA record mapped to one protein
  chain
- runs local ESMFold2 inference through a simple command-line interface
- writes per-seed mmCIF structures, confidence artifacts, and text reports
- collects all seed-level metrics into one summary CSV at the root output
  directory

## Requirements

- Python 3.10+
- `esm` 3.0.0+

## Installation

Clone the repository and install the package in your environment.

```bash
git clone <repository-url>
cd esmfold2-complex

# Option 1: install with uv
uv sync
source .venv/bin/activate

# Option 2: install with venv + pip
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

After installation, the `esmfold2-complex` command is available in the
environment. If you prefer not to activate the `uv` environment, you can also
run commands as `uv run esmfold2-complex ...`.

## Usage

### Input

The CLI takes one positional FASTA file:

```bash
esmfold2-complex path/to/complex.fasta
```

Each FASTA record is treated as one protein chain in the complex.

### Common examples

```bash
# Use the default local or cached model setting
esmfold2-complex path/to/complex.fasta

# Write all outputs under a dedicated root directory
esmfold2-complex path/to/complex.fasta -o results

# Point explicitly to a local model directory
esmfold2-complex path/to/complex.fasta \
  --model /path/to/local/ESMFold2 \
  -o results

# Run multiple consecutive seeds starting from a fixed base seed
esmfold2-complex path/to/complex.fasta \
  --seed 5 \
  --num-seeds 3 \
  -o results
```

If `--seed` is omitted, the CLI generates a random base seed at runtime and
prints the planned seed list before inference starts.

### Main options

| Option | Default | Description |
| --- | --- | --- |
| `fasta` | required | Input FASTA file; each FASTA record is one chain |
| `-o`, `--output` | input FASTA directory | Root output directory |
| `--model` | `biohub/ESMFold2` | Local model directory or cached Hugging Face repo id |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--gpu-id` | `0` | CUDA device index when using GPU |
| `--esmc-precision` | `auto` | `auto`, `fp32`, `bf16`, or `fp8` |
| `--chunk-size` | `64` | Chunk size for memory-heavy ESMFold2 blocks; `0` disables chunking |
| `--allow-tf32` | enabled | Enable TF32 kernels on supported NVIDIA GPUs |
| `--num-loops` | `10` | Refinement loops |
| `--num-sampling-steps` | `100` | Diffusion sampling steps |
| `--seed` | random at runtime | Base seed for the first run |
| `--num-seeds` | `1` | Number of consecutive seed-specific predictions |

CCD lookup follows upstream `esm` behavior. If `--model` points to a local
directory that already contains `ccd.pkl`, the wrapper uses that directory.
Otherwise it defers to upstream `esm` resolution, including
`ESMCFOLD_CCD_PATH` and the default upstream cache/download path for
`biohub/ESMFold2`.

## Output layout

`--output` names the root output directory. Each seed writes into its own
`<fasta_stem>_seedN/` subdirectory, and the combined CSV summary stays at the
root.

For example:

```bash
esmfold2-complex path/to/input.fasta -o results --seed 5 --num-seeds 2
```

produces:

```text
results/
├── input_seed5/
│   ├── input_seed5.cif
│   ├── input_seed5_plddt.png
│   ├── input_seed5_pae.png
│   ├── input_seed5_pair_iptm.png
│   ├── input_seed5_structure_views.html
│   └── input_seed5_quality_report.txt
├── input_seed6/
│   ├── input_seed6.cif
│   ├── input_seed6_plddt.png
│   ├── input_seed6_pae.png
│   ├── input_seed6_pair_iptm.png
│   ├── input_seed6_structure_views.html
│   └── input_seed6_quality_report.txt
└── input_seed_summary.csv
```

The summary CSV reports seed-level paths and quality metrics such as
`mean_plddt`, `ptm`, `iptm`, and `mean_inter_chain_pae`.

## Notes

- This project currently supports only the local ESM runtime stack used by this
  wrapper. In practice, `--model` should resolve to a local directory or an
  already cached checkpoint.
- `--model` selects model weights only; it does not force `ccd.pkl` to come
  from the same Hugging Face repo id.
- Biohub-hosted integration has not been integrated into the current workflow.
- The current CLI contract was checked against `python -m esmfold2_complex.cli
  --help`: `--output` is a root directory, each seed writes to its own
  subdirectory, and the summary CSV stays at the root.

## License

MIT
