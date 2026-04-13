"""Command-line entry point for reflection coefficient analysis.

Three path inputs can be overridden independently (each is remembered in
``~/.reflection_coefficient.json`` the next time you supply it):

* ``--tank-config``   — path to the tank_config.json file
* ``--metadata-dir``  — folder containing rw.csv / wn.csv / js.csv
* ``--data-dir``      — folder holding raw ``<TEST_ID>.txt`` files
                        (flat or with per-scheme subfolders)

The only mandatory per-run choice is the wave scheme (``rw`` / ``wn`` / ``js``),
which is prompted for if ``--scheme`` is omitted.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from reflection_coefficient.io import (
    USER_CONFIG_PATH,
    list_tests,
    load_probe_data,
    resolve_data_dir,
    resolve_metadata_dir,
    resolve_tank_config,
    save_paths,
)
from reflection_coefficient.pipeline import IrregularResult, RegularResult, analyse

SCHEME_LABELS = {
    "rw": "REGULAR WAVE",
    "wn": "WHITE-NOISE IRREGULAR WAVE",
    "js": "JONSWAP IRREGULAR WAVE",
}


def _prompt_choice(question: str, choices: tuple[str, ...]) -> str:
    menu = "  ".join(f"[{i + 1}] {c}" for i, c in enumerate(choices))
    while True:
        raw = input(f"{question}\n  {menu}\n> ").strip().lower()
        if raw in choices:
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"Please enter one of {choices} or 1..{len(choices)}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute wave tank reflection coefficient.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tank-config", type=Path, default=None,
        help=f"Tank config JSON file. Persisted in {USER_CONFIG_PATH} when set.",
    )
    parser.add_argument(
        "--metadata-dir", type=Path, default=None,
        help="Folder with rw.csv / wn.csv / js.csv. Persisted when set.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help=(
            "Folder containing raw <TEST_ID>.txt files (flat or with "
            "rw/wn/js subfolders). Persisted when set."
        ),
    )
    parser.add_argument(
        "--scheme", choices=["rw", "wn", "js"], default=None,
        help="Wave scheme (rw/wn/js). Prompted if omitted.",
    )
    parser.add_argument(
        "--test", default="all",
        help="Test id (e.g. RW005) or 'all' for every test found.",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--method", choices=["goda", "least_squares"], default="least_squares")
    parser.add_argument(
        "--list", action="store_true",
        help="List tests discovered for the chosen scheme and exit.",
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="Print the resolved tank_config / metadata_dir / data_dir and exit.",
    )
    args = parser.parse_args()

    # Persist any explicitly provided paths before resolving (so the saved
    # values are available to later invocations regardless of errors below).
    if any(v is not None for v in (args.tank_config, args.metadata_dir, args.data_dir)):
        save_paths(
            tank_config=args.tank_config,
            metadata_dir=args.metadata_dir,
            data_dir=args.data_dir,
        )
        print(f"[run_analysis] updated saved paths in {USER_CONFIG_PATH}")

    tank_cfg = resolve_tank_config(args.tank_config)
    meta_dir = resolve_metadata_dir(args.metadata_dir)
    data_dir = resolve_data_dir(args.data_dir)

    if args.show_paths:
        print(f"  tank_config  = {tank_cfg}")
        print(f"  metadata_dir = {meta_dir}")
        print(f"  data_dir     = {data_dir}")
        return

    for label, p in [("tank_config", tank_cfg), ("metadata_dir", meta_dir), ("data_dir", data_dir)]:
        if not p.exists():
            print(f"[run_analysis] {label} does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    scheme = args.scheme or _prompt_choice(
        "Select wave scheme for this run:", ("rw", "wn", "js")
    )

    banner = (
        f" {SCHEME_LABELS[scheme]} | data_dir={data_dir} "
        f"| metadata_dir={meta_dir} "
    )
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    available = list_tests(scheme, data_dir=data_dir, metadata_dir=meta_dir)
    print(f"[run_analysis] {len(available)} test(s) found for {scheme}")

    if args.list:
        for tid in available:
            print(" ", tid)
        return

    if args.test == "all":
        selected = available
    elif args.test in available:
        selected = [args.test]
    else:
        print(f"Test {args.test!r} not found under {data_dir}.", file=sys.stderr)
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)
    regular_results: list[RegularResult] = []
    for tid in selected:
        t, eta1, eta2, eta3, meta = load_probe_data(
            tid, campaign=scheme,
            tank_config=tank_cfg, metadata_dir=meta_dir, data_dir=data_dir,
        )
        print(f"[run_analysis] {tid}: N={len(t)}, fs≈{1/(t[1]-t[0]):.1f} Hz")
        try:
            result = analyse(t, eta1, eta2, eta3, meta, method=args.method)
        except Exception as exc:
            print(f"  !! {tid}: {exc}", file=sys.stderr)
            continue
        _report(result)
        if isinstance(result, RegularResult):
            regular_results.append(result)

    if scheme == "rw" and len(regular_results) >= 2:
        _write_kr_vs_freq(regular_results, args.output, args.method)


def _report(result: RegularResult | IrregularResult) -> None:
    if isinstance(result, RegularResult):
        print(
            f"  {result.test_id} [{result.method}] "
            f"f={result.f_Hz:.3f} Hz  H_I={result.H_I:.4f} m  "
            f"H_R={result.H_R:.4f} m  Kr={result.Kr:.3f}"
            + ("" if result.singularity_ok else "  [SINGULARITY]")
        )
    else:
        d = result.diagnostics
        print(
            f"  {result.test_id} [{result.method}] "
            f"Hm0_I={result.Hm0_I:.4f} m  Hm0_R={result.Hm0_R:.4f} m  "
            f"Tp_I={result.Tp_I:.3f} s  Kr={result.Kr_overall:.3f}  "
            f"(bins={d['n_bins_valid']}, min D/sin²={d['D_or_sin2_min']:.3f})"
        )


def _write_kr_vs_freq(results: list[RegularResult], out_dir: Path, method: str) -> None:
    """Aggregate regular-wave runs into a Kr(f) table (+ plot if matplotlib)."""
    rows = sorted(results, key=lambda r: r.f_Hz)
    csv_path = out_dir / f"rw_kr_vs_freq_{method}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "f_Hz", "k_rad_m", "L_m", "H_I_m", "H_R_m", "Kr", "singularity_ok"])
        for r in rows:
            w.writerow([
                r.test_id, f"{r.f_Hz:.6f}", f"{r.k:.6f}", f"{r.wavelength_m:.6f}",
                f"{r.H_I:.6f}", f"{r.H_R:.6f}", f"{r.Kr:.6f}", int(r.singularity_ok),
            ])
    print(f"[run_analysis] wrote {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    fs = [r.f_Hz for r in rows]
    krs = [r.Kr for r in rows]
    ax.plot(fs, krs, marker="o", linestyle="-")
    for r in rows:
        if not r.singularity_ok:
            ax.plot([r.f_Hz], [r.Kr], marker="x", color="red")
    ax.set_xlabel("f [Hz]")
    ax.set_ylabel(r"$K_r$")
    ax.set_title(f"Regular-wave reflection coefficient ({method})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    png_path = out_dir / f"rw_kr_vs_freq_{method}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[run_analysis] wrote {png_path}")


if __name__ == "__main__":
    main()
