"""Tests for :mod:`esmfold2_complex.cli` argument parsing.

These tests pin the current packaged CLI surface (names, short forms, defaults,
choices, and aliases). They do not exercise the end-to-end pipeline, which
requires a real model and a GPU.
"""

from __future__ import annotations

import argparse

import pytest

from esmfold2_complex.cli import build_parser, parse_args


EXPECTED_ARGUMENTS: set[str] = {
    "fasta",
    "output",
    "model",
    "device",
    "gpu_id",
    "esmc_precision",
    "chunk_size",
    "allow_tf32",
    "num_loops",
    "num_sampling_steps",
    "seed",
    "num_seeds",
}


def test_parser_exposes_every_expected_argument() -> None:
    parser = build_parser()

    actual = {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}

    assert EXPECTED_ARGUMENTS.issubset(actual)


def test_parser_exposes_positional_fasta() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.fasta == pytest.approx(__import__("pathlib").Path("input.fasta"))


def test_short_output_flag_resolves_to_output_dest() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "-o", "results"])

    assert args.output == pytest.approx(__import__("pathlib").Path("results"))


def test_default_model_is_biohub_esmfold2() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.model == "biohub/ESMFold2"


def test_default_device_is_auto() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.device == "auto"


def test_default_gpu_id_is_zero() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.gpu_id == 0


def test_default_esmc_precision_is_auto() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.esmc_precision == "auto"


def test_default_chunk_size_is_64() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.chunk_size == 64


def test_default_allow_tf32_is_true() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.allow_tf32 is True


def test_no_allow_tf32_flag_disables_tf32() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--no-allow-tf32"])

    assert args.allow_tf32 is False


def test_allow_tf32_flag_still_works_alongside_negative_form() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--allow-tf32"])

    assert args.allow_tf32 is True


def test_default_num_loops_is_10() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.num_loops == 10


def test_default_num_sampling_steps_is_100() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.num_sampling_steps == 100


def test_default_seed_is_omitted() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.seed is None


def test_default_num_seeds_is_one() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta"])

    assert args.num_seeds == 1


def test_num_seeds_dash_form_populates_dest() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--num-seeds", "3"])

    assert args.num_seeds == 3


def test_num_seeds_underscore_alias_populates_same_dest() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--num_seeds", "4"])

    assert args.num_seeds == 4


def test_device_choices_are_auto_cpu_cuda() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--device", "cpu"])

    assert args.device == "cpu"

    args = parser.parse_args(["input.fasta", "--device", "cuda"])

    assert args.device == "cuda"


def test_device_rejects_unknown_choice() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.fasta", "--device", "tpu"])


def test_esmc_precision_choices() -> None:
    parser = build_parser()

    for choice in ("auto", "fp32", "bf16", "fp8"):
        args = parser.parse_args(["input.fasta", "--esmc-precision", choice])
        assert args.esmc_precision == choice


def test_esmc_precision_rejects_unknown_choice() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.fasta", "--esmc-precision", "int8"])


def test_hf_home_option_is_removed() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["input.fasta", "--hf-home", "/tmp/cache"])


def test_gpu_id_override() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--gpu-id", "2"])

    assert args.gpu_id == 2


def test_chunk_size_zero_disables_chunking() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--chunk-size", "0"])

    assert args.chunk_size == 0


def test_num_loops_and_sampling_steps_override() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "input.fasta",
            "--num-loops",
            "3",
            "--num-sampling-steps",
            "8",
        ]
    )

    assert args.num_loops == 3
    assert args.num_sampling_steps == 8


def test_seed_override() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--seed", "42"])

    assert args.seed == 42


def test_model_override() -> None:
    parser = build_parser()

    args = parser.parse_args(["input.fasta", "--model", "/tmp/my_model"])

    assert args.model == "/tmp/my_model"


def test_parse_args_is_a_thin_wrapper_around_build_parser() -> None:
    args = parse_args(["input.fasta"])

    assert isinstance(args, argparse.Namespace)
    assert args.fasta == pytest.approx(__import__("pathlib").Path("input.fasta"))
    assert args.seed is None
    assert args.num_seeds == 1
    assert args.allow_tf32 is True


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower()
    assert "--num-seeds" in captured.out
    assert "--num_seeds" in captured.out
    assert "--no-allow-tf32" in captured.out
    assert "--esmc-precision" in captured.out
    assert "Root output directory." in captured.out
    assert "<fasta_stem>_seedN/" in captured.out
    assert "summary CSV" in captured.out
