"""CLI entrypoints for YAML-driven Pilgrim Lightning experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .yaml_config import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser.

    Returns:
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(
        prog="pilgrim-lightning",
        description="Run Lightning-based Pilgrim train/inference experiments from YAML.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for cmd in ("run", "train", "infer", "benchmark-algraphgpt"):
        p = sub.add_parser(cmd)
        p.add_argument(
            "--config",
            required=True,
            type=Path,
            help="Path to YAML run configuration.",
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    CLI main function.

    Args:
        argv: Optional CLI argument list.

    Returns:
        Process exit code.

    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if str(args.command) == "benchmark-algraphgpt":
        try:
            from .algraphgpt_timing import run_algraphgpt_timing_from_yaml
        except ModuleNotFoundError as exc:
            parser.error(
                "Missing runtime dependency. Install `lightning` and related "
                "packages before running benchmark command."
            )
            raise exc
        result = run_algraphgpt_timing_from_yaml(args.config)
        print(
            f"benchmark_run_dir={result.run_dir}\n"
            f"benchmark_summary={result.summary_path}"
        )
        return 0

    try:
        from .pipeline import run_from_config
    except ModuleNotFoundError as exc:
        parser.error(
            "Missing runtime dependency. Install `lightning` and `aim` before "
            "running train/infer commands."
        )
        raise exc

    config = load_yaml_config(args.config)
    result = run_from_config(config, mode=str(args.command))
    print_summary(result, mode=str(args.command))
    return 0


def print_summary(result: dict[str, Any], *, mode: str) -> None:
    """
    Print compact run summary.

    Args:
        result: Result dictionary from pipeline.
        mode: Executed command mode.

    Returns:
        None.

    """
    train_results = result.get("train_results", {})
    inference_result = result.get("inference_result")

    if mode in {"run", "train"}:
        print(f"trained_models={len(train_results)}")
        for n, item in sorted(train_results.items()):
            print(
                f"n={int(n)} elapsed_s={item.elapsed_seconds:.2f} "
                f"model_path={item.model_path}"
            )

    if mode in {"run", "infer"} and inference_result is not None:
        print(
            "inference_stats="
            f"attempted:{inference_result.stats.attempted} "
            f"accepted:{inference_result.stats.accepted} "
            f"oom:{inference_result.stats.oom_count}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
