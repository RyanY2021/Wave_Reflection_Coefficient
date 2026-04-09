"""Command-line entry point for reflection coefficient analysis."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute wave tank reflection coefficient.")
    parser.add_argument("--input", required=True, help="Path to experiment data file.")
    parser.add_argument("--output", default="results/", help="Output directory.")
    parser.add_argument("--method", choices=["goda", "least_squares"], default="least_squares")
    args = parser.parse_args()

    print(f"[run_analysis] input={args.input} method={args.method} output={args.output}")
    # TODO: wire up io -> preprocessing -> method -> save results


if __name__ == "__main__":
    main()
