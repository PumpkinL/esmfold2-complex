"""Command-line interface for ESMFold2 complex prediction.

The argument parser stays broadly aligned with the reference script in
``pipeline_example/predict_complex_structure.py`` for names, short forms,
choices, ``--num-seeds`` / ``--num_seeds`` alias, and
``--allow-tf32`` / ``--no-allow-tf32`` ``BooleanOptionalAction``. The main
intentional behavior change is seed planning: omitting ``--seed`` now picks a
random base seed at runtime, and multi-seed execution expands to a consecutive
range from that base seed.

``main()`` wires the modular pipeline together:

1. Resolve device / precision / chunk-size / TF32 and apply TF32 settings.
2. Read the FASTA, build chain specs.
3. Resolve any bundled local CCD cache override and load the local
   ``ESMFold2Model``.
4. **Phase A — inference loop**: for each planned seed, call
   :func:`esmfold2_complex.inference.run_single_seed` and stash the
   ``FoldedResult``. No per-seed disk I/O happens here so the GPU is not
   blocked by mmCIF writes or visualisation.
5. **Phase B — post-processing loop**: write mmCIFs, render the
   per-seed artifact PNGs/HTML, emit the per-seed quality report, and
   assemble the :class:`PredictionSummary` rows.
6. Write the combined CSV summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from esmfold2_complex.contracts import mean_optional, optional_metric
from esmfold2_complex.io_paths import (
    resolve_output_dir,
    resolve_output_prefix,
    resolve_quality_report_path,
    resolve_seed_output_dir,
    resolve_seed_output_path,
    resolve_summary_csv_path,
)
from esmfold2_complex.runtime import (
    configure_torch_runtime,
    describe_runtime_device,
    resolve_chunk_size,
    resolve_esmc_precision,
    resolve_runtime_device,
)

DEFAULT_MODEL = "biohub/ESMFold2"


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ESMFold2 complex CLI."""
    parser = argparse.ArgumentParser(
        prog="esmfold2-complex",
        description=(
            "Fold a protein complex from FASTA using local ESMFold2 and save a "
            "small set of confidence visualizations."
        ),
    )
    parser.add_argument(
        "fasta",
        type=Path,
        help="Input FASTA file. Each FASTA record is one protein chain.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Root output directory. Each seed writes to "
            "<fasta_stem>_seedN/ inside this directory, and the summary CSV "
            "stays at the root. Default: the input FASTA directory."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local model directory or cached Hugging Face repo id.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Run on CPU or CUDA. Use --gpu-id to pick a CUDA device. Default: auto",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="CUDA device index to use when --device is auto or cuda. Default: 0",
    )
    parser.add_argument(
        "--esmc-precision",
        choices=["auto", "fp32", "bf16", "fp8"],
        default="auto",
        help=(
            "Precision for the ESMC language model on GPU. "
            "Default: auto (bf16 when supported, else fp32)"
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help=(
            "Chunk size for ESMFold2 L^2 memory-heavy blocks. "
            "Use 0 to disable chunking. Default: 64"
        ),
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 matmul/cuDNN kernels on supported NVIDIA GPUs. Default: enabled",
    )
    parser.add_argument(
        "--num-loops",
        type=int,
        default=10,
        help="Refinement loops. Default: 10",
    )
    parser.add_argument(
        "--num-sampling-steps",
        type=int,
        default=100,
        help="Diffusion sampling steps. Default: 100",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Base seed for the first run. Multi-seed runs expand "
            "consecutively from this value. If omitted, a random base seed is "
            "generated at runtime."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        dest="num_seeds",
        type=int,
        default=1,
        help=(
            "Number of seed-specific predictions to run. When greater than 1, "
            "seeds expand consecutively from the base seed and each seed "
            "writes into its own <fasta_stem>_seedN/ subdirectory. Default: 1"
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Console entry point: drive the modular ESMFold2 complex pipeline."""
    args = parse_args(argv)

    import torch

    from esmfold2_complex.artifacts import write_seed_artifacts
    from esmfold2_complex.fasta import build_chain_specs, read_complex_fasta
    from esmfold2_complex.inference import (
        build_prediction_summary,
        build_run_seeds,
        run_single_seed,
    )
    from esmfold2_complex.modeling import (
        build_structure_input,
        load_esmfold2_model,
        make_input_builder,
        resolve_ccd_cache,
    )
    from esmfold2_complex.reporting import write_quality_report, write_summary_csv

    device = resolve_runtime_device(args.device, args.gpu_id, torch)
    esmc_precision = resolve_esmc_precision(device, args.esmc_precision, torch)
    chunk_size = resolve_chunk_size(args.chunk_size)
    configure_torch_runtime(device, args.allow_tf32, torch)

    output_dir = resolve_output_dir(args.fasta, args.output)
    output_prefix = resolve_output_prefix(args.fasta, args.output)
    summary_csv_path = resolve_summary_csv_path(output_prefix)
    chains = read_complex_fasta(args.fasta)
    chain_specs = build_chain_specs(chains)
    ccd_cache = resolve_ccd_cache(args.model)
    run_seeds = build_run_seeds(args.seed, args.num_seeds)

    print(f"Loaded {len(chains)} chains from {args.fasta}")
    for chain in chain_specs:
        print(f"  {chain.chain_id}: {chain.display_name} ({chain.length} aa)")
    print(f"Using device: {describe_runtime_device(device, torch)}")
    if device.startswith("cuda"):
        print(f"ESMC precision: {esmc_precision}")
        print(
            "Chunk size: "
            f"{chunk_size if chunk_size is not None else 'disabled'}"
        )
        print(f"TF32: {'enabled' if args.allow_tf32 else 'disabled'}")
    print(f"Loading model: {args.model}")
    print(f"Planned seeds ({len(run_seeds)}): {', '.join(str(seed) for seed in run_seeds)}")
    if device == "cpu":
        print("CPU mode is supported but can be very slow for real complexes.")

    model = load_esmfold2_model(args.model, esmc_precision, device, torch)
    model.set_chunk_size(chunk_size)
    builder = make_input_builder(ccd_cache)
    structure_input = build_structure_input(chains)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_writes: list[tuple] = []

    # Phase A — inference loop. Run every fold back-to-back and only touch
    # the GPU; defer all per-seed disk I/O and visualisation work to the
    # second loop so the GPU is not blocked by py3Dmol/matplotlib rendering
    # and synchronous file writes.
    for run_index, seed in enumerate(run_seeds, start=1):
        seed_output_dir = resolve_seed_output_dir(output_prefix, seed)
        seed_output_path = resolve_seed_output_path(output_prefix, seed)
        print(
            f"[{run_index}/{len(run_seeds)}] Running seed {seed} -> "
            f"{seed_output_path.name}"
        )
        folded = run_single_seed(
            seed=seed,
            seed_output_path=seed_output_path,
            output_dir=seed_output_dir,
            model=model,
            builder=builder,
            structure_input=structure_input,
            chain_specs=chain_specs,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            torch=torch,
        )
        pending_writes.append(
            (
                seed_output_path,
                seed_output_dir,
                folded,
            )
        )

    # Phase B — post-processing loop. Now that no further GPU work is
    # outstanding, write mmCIFs, render plots/HTML, emit quality reports,
    # and assemble the CSV summaries.
    summaries: list = []
    for run_index, (seed_output_path, seed_output_dir, folded) in enumerate(
        pending_writes, start=1
    ):
        print(
            f"[{run_index}/{len(pending_writes)}] Writing artifacts for seed "
            f"{folded.seed}"
        )
        seed_output_dir.mkdir(parents=True, exist_ok=True)
        seed_output_path.write_text(folded.mmcif_text)

        artifact_paths = write_seed_artifacts(
            seed_output_path=seed_output_path,
            output_dir=seed_output_dir,
            mmcif_text=folded.mmcif_text,
            plddt=folded.plddt,
            pae=folded.pae,
            pair_chains_iptm=folded.pair_chains_iptm,
            chain_specs=chain_specs,
        )
        quality_report_path = resolve_quality_report_path(seed_output_path)
        artifact_paths["quality_report_txt"] = quality_report_path

        mean_inter_chain_pae = mean_optional(
            [
                summary.mean_pae
                for summary in folded.pair_summaries
                if summary.mean_pae is not None
            ]
        )
        mean_pair_iptm = mean_optional(
            [
                summary.pair_iptm
                for summary in folded.pair_summaries
                if summary.pair_iptm is not None
            ]
        )
        write_quality_report(
            quality_report_path,
            args.fasta,
            seed_output_path,
            folded.seed,
            chain_specs,
            folded.plddt,
            folded.mean_plddt,
            folded.ptm,
            folded.iptm,
            folded.pae,
            folded.pair_chains_iptm,
            artifact_paths,
            pair_summaries=folded.pair_summaries,
            mean_inter_chain_pae=mean_inter_chain_pae,
            mean_pair_iptm=mean_pair_iptm,
            summary_csv_path=summary_csv_path,
        )

        summary = build_prediction_summary(
            seed=folded.seed,
            output_cif=seed_output_path,
            output_dir=seed_output_dir,
            report_path=quality_report_path,
            chain_specs=chain_specs,
            plddt=folded.plddt,
            ptm=folded.ptm,
            iptm=folded.iptm,
            pair_summaries=folded.pair_summaries,
            mean_inter_chain_pae=mean_inter_chain_pae,
            mean_pair_iptm=mean_pair_iptm,
        )
        summaries.append(summary)

        print(f"  saved structure: {seed_output_path}")
        print(f"  saved outputs dir: {seed_output_dir}")
        print(f"  mean pLDDT: {summary.mean_plddt:.2f}")
        print(f"  pTM: {optional_metric(summary.ptm)}")
        print(f"  ipTM: {optional_metric(summary.iptm)}")

    write_summary_csv(summary_csv_path, summaries)
    print(f"Saved combined CSV summary to {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
