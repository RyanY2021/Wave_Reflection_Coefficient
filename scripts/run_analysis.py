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
from datetime import datetime
from pathlib import Path

from reflection_coefficient.calibration import recalibrate_probes
from reflection_coefficient.io import (
    USER_CONFIG_PATH,
    list_tests,
    load_probe_data,
    load_probes_config,
    resolve_data_dir,
    resolve_drops,
    resolve_freq_source,
    resolve_goda_pair,
    resolve_metadata_dir,
    resolve_method,
    resolve_probes_config,
    resolve_recalibrate,
    resolve_tank_config,
    resolve_window,
    save_drops,
    save_freq_source,
    save_goda_pair,
    save_method,
    save_paths,
    save_recalibrate,
    save_window,
)
from reflection_coefficient.irregular_report import write_irregular_report
from reflection_coefficient.pipeline import IrregularResult, RegularResult, analyse
from reflection_coefficient.rw_report import write_rw_report

SCHEME_LABELS = {
    "rw": "REGULAR WAVE",
    "wn": "WHITE-NOISE IRREGULAR WAVE",
    "js": "JONSWAP IRREGULAR WAVE",
}


class _Tee:
    """Duplicate writes to an underlying stream and a log file."""

    def __init__(self, stream, log_fh):
        self._stream = stream
        self._log = log_fh

    def write(self, s):
        self._stream.write(s)
        self._log.write(s)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _prune_logs(log_dir: Path, keep: int) -> None:
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    for old in logs[:-keep] if keep > 0 else logs:
        try:
            old.unlink()
        except OSError:
            pass


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
        "--probes-config", type=Path, default=None,
        help=(
            "Per-probe linear re-calibration JSON "
            "(default: experiment_data/probes.json). Persisted when set. "
            "Consumed only when --recalibrate is on."
        ),
    )
    parser.add_argument(
        "--recalibrate", default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Apply per-probe linear re-calibration from probes.json after "
            "loading. Use --no-recalibrate to disable. Persisted when set "
            "(default: off)."
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
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Parent output dir (default: <project>/results). A timestamped subfolder is created per run.",
    )
    parser.add_argument(
        "--method", choices=["goda", "least_squares"], default=None,
        help="Separation method. Persisted when set; defaults to last choice (or least_squares).",
    )
    parser.add_argument(
        "--window", choices=["none", "hann"], default=None,
        help="Spectral window for irregular-wave FFT. Persisted when set (default: hann).",
    )
    parser.add_argument(
        "--bandwidth", type=float, default=None,
        help="Target resolution bandwidth in Hz (only with --window hann). Persisted when set (default: 0.04).",
    )
    parser.add_argument(
        "--head-drop", type=float, default=None,
        help="Seconds to drop from the start of the clean analysis window. Persisted when set (default: 3.0).",
    )
    parser.add_argument(
        "--tail-drop", type=float, default=None,
        help="Seconds to drop from the end of the clean analysis window. Persisted when set (default: 3.0).",
    )
    parser.add_argument(
        "--window-mode", choices=["canonical", "noref"], default="canonical",
        help=(
            "Time-window convention. 'canonical' (default) uses the standard "
            "post-reflection clip. 'noref' uses the pre-reflection window — "
            "from when the incident wave has reached every probe to just "
            "before the first reflection returns to wp3 — so the separation "
            "is applied to incident-only signal. A baseline sanity check: "
            "ideally Kr ~ 0. Not persisted."
        ),
    )
    parser.add_argument(
        "--freq-source", choices=["bin", "target"], default=None,
        help=(
            "Regular-wave only. 'bin' (default) picks the nearest FFT bin to "
            "meta.f_Hz. 'target' evaluates a single-point DFT at exactly "
            "meta.f_Hz, bypassing bin quantisation (reduces leakage bias on "
            "short clips). Persisted when set."
        ),
    )
    parser.add_argument(
        "--goda-pair", choices=["13", "12", "23"], default=None,
        help=(
            "Goda method only. Which probe pair to use for the two-probe "
            "separation: '13' (wp1 & wp3, default — widest spacing), '12' "
            "(wp1 & wp2) or '23' (wp2 & wp3, spacing = X13 − X12). Changing "
            "the spacing moves the kΔ = nπ singularities in frequency. "
            "Ignored when --method least_squares. Persisted when set."
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List tests discovered for the chosen scheme and exit.",
    )
    parser.add_argument(
        "--show-paths", action="store_true",
        help="Print the resolved tank_config / metadata_dir / data_dir and exit.",
    )
    parser.add_argument(
        "--no-log", action="store_true",
        help="Disable writing a per-run log file under <project>/log/.",
    )
    parser.add_argument(
        "--log-keep", type=int, default=10,
        help="Number of recent log files to retain (default: 10; 0 = keep all).",
    )
    args = parser.parse_args()

    log_fh = None
    if not args.no_log and not args.show_paths and not args.list:
        project_root = Path(__file__).resolve().parents[1]
        log_dir = project_root / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
        log_fh = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, log_fh)
        sys.stderr = _Tee(sys.stderr, log_fh)
        print(f"[run_analysis] log file: {log_path}")
        _prune_logs(log_dir, args.log_keep)

    try:
        _run(args)
    finally:
        if log_fh is not None:
            sys.stdout = sys.stdout._stream  # type: ignore[attr-defined]
            sys.stderr = sys.stderr._stream  # type: ignore[attr-defined]
            log_fh.close()


def _run(args) -> None:

    # Persist any explicitly provided paths before resolving (so the saved
    # values are available to later invocations regardless of errors below).
    if any(v is not None for v in (
        args.tank_config, args.metadata_dir, args.data_dir, args.probes_config,
    )):
        save_paths(
            tank_config=args.tank_config,
            metadata_dir=args.metadata_dir,
            data_dir=args.data_dir,
            probes_config=args.probes_config,
        )
        print(f"[run_analysis] updated saved paths in {USER_CONFIG_PATH}")

    if args.method is not None:
        save_method(args.method)
    args.method = resolve_method(args.method)

    if args.freq_source is not None:
        save_freq_source(args.freq_source)
    args.freq_source = resolve_freq_source(args.freq_source)

    if args.goda_pair is not None:
        save_goda_pair(args.goda_pair)
    args.goda_pair = resolve_goda_pair(args.goda_pair)

    if args.window is not None or args.bandwidth is not None:
        save_window(window=args.window, bandwidth_Hz=args.bandwidth)
    args.window, args.bandwidth = resolve_window(args.window, args.bandwidth)

    if args.head_drop is not None or args.tail_drop is not None:
        save_drops(head_drop_s=args.head_drop, tail_drop_s=args.tail_drop)
    args.head_drop, args.tail_drop = resolve_drops(args.head_drop, args.tail_drop)

    if args.recalibrate is not None:
        save_recalibrate(args.recalibrate)
    args.recalibrate = resolve_recalibrate(args.recalibrate)

    tank_cfg = resolve_tank_config(args.tank_config)
    meta_dir = resolve_metadata_dir(args.metadata_dir)
    data_dir = resolve_data_dir(args.data_dir)
    probes_cfg_path = resolve_probes_config(args.probes_config)

    if args.show_paths:
        print(f"  tank_config   = {tank_cfg}")
        print(f"  metadata_dir  = {meta_dir}")
        print(f"  data_dir      = {data_dir}")
        print(f"  probes_config = {probes_cfg_path}")
        print(f"  method        = {args.method}")
        bw_txt = f"{args.bandwidth:g} Hz" if args.bandwidth is not None else "—"
        print(f"  window        = {args.window}  (bandwidth: {bw_txt})")
        print(
            f"  drops         = head {args.head_drop:g} s, "
            f"tail {args.tail_drop:g} s"
        )
        print(f"  recalibrate   = {'on' if args.recalibrate else 'off'}")
        print(f"  freq_source   = {args.freq_source}")
        print(f"  goda_pair     = {args.goda_pair}")
        return

    for label, p in [("tank_config", tank_cfg), ("metadata_dir", meta_dir), ("data_dir", data_dir)]:
        if not p.exists():
            print(f"[run_analysis] {label} does not exist: {p}", file=sys.stderr)
            sys.exit(1)

    probes_cfg = None
    if args.recalibrate:
        if not probes_cfg_path.exists():
            print(
                f"[run_analysis] --recalibrate requires a probes config at "
                f"{probes_cfg_path} (run scripts/init_project.py to scaffold it).",
                file=sys.stderr,
            )
            sys.exit(1)
        probes_cfg = load_probes_config(probes_cfg_path)

    scheme = args.scheme or _prompt_choice(
        "Select wave scheme for this run:", ("rw", "wn", "js")
    )

    bw_txt = f"{args.bandwidth:g} Hz" if args.bandwidth is not None else "—"
    pair_txt = (
        f" | goda_pair={args.goda_pair}" if args.method == "goda" else ""
    )
    banner = (
        f" {SCHEME_LABELS[scheme]} | method={args.method}{pair_txt} "
        f"| window={args.window} (bw {bw_txt}) "
        f"| drops head {args.head_drop:g}s tail {args.tail_drop:g}s "
        f"| recalibrate={'on' if args.recalibrate else 'off'} "
        f"| mode={args.window_mode} | freq={args.freq_source} "
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
        if scheme != "rw":
            print(
                f"[run_analysis] --test all is only supported for --scheme rw; "
                f"specify a single {scheme.upper()}### test id for irregular schemes.",
                file=sys.stderr,
            )
            sys.exit(2)
        selected = available
    elif args.test in available:
        selected = [args.test]
    else:
        print(f"Test {args.test!r} not found under {data_dir}.", file=sys.stderr)
        sys.exit(2)

    _project_root = Path(__file__).resolve().parents[1]
    output_parent = args.output if args.output is not None else _project_root / "results"
    run_dir = output_parent / datetime.now().strftime("%Y%m%d_%H%M%S")

    def _ensure_run_dir() -> Path:
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run_analysis] output dir: {run_dir}")
        return run_dir

    regular_results: list[RegularResult] = []
    regular_metas: list = []
    for tid in selected:
        t, eta1, eta2, eta3, meta = load_probe_data(
            tid, campaign=scheme,
            tank_config=tank_cfg, metadata_dir=meta_dir, data_dir=data_dir,
        )
        if probes_cfg is not None:
            eta1, eta2, eta3 = recalibrate_probes(eta1, eta2, eta3, probes_cfg)
        print(f"[run_analysis] {tid}: N={len(t)}, fs≈{1/(t[1]-t[0]):.1f} Hz")
        try:
            result = analyse(
                t, eta1, eta2, eta3, meta,
                method=args.method,
                window=args.window, bandwidth_Hz=args.bandwidth,
                head_drop_s=args.head_drop, tail_drop_s=args.tail_drop,
                window_mode=args.window_mode,
                freq_source=args.freq_source,
                goda_pair=args.goda_pair,
            )
        except Exception as exc:
            print(f"  !! {tid}: {exc}", file=sys.stderr)
            continue
        _report(result)
        if isinstance(result, RegularResult):
            regular_results.append(result)
            regular_metas.append(meta)
        elif isinstance(result, IrregularResult):
            out = _ensure_run_dir()
            html_path = write_irregular_report(
                result, meta, out, args.method, timestamp=out.name,
                window_mode=args.window_mode,
            )
            print(f"[run_analysis] wrote {html_path}")

    if scheme == "rw" and len(regular_results) >= 2:
        out = _ensure_run_dir()
        csv_path = _write_kr_vs_freq(
            regular_results, out, args.method, window_mode=args.window_mode,
        )
        html_path = write_rw_report(
            list(zip(regular_results, regular_metas)),
            out, args.method,
            csv_path=csv_path, timestamp=out.name,
            window_mode=args.window_mode,
        )
        print(f"[run_analysis] wrote {html_path}")


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


def _write_kr_vs_freq(
    results: list[RegularResult], out_dir: Path, method: str,
    window_mode: str = "canonical",
) -> Path:
    """Aggregate regular-wave runs into a Kr(f) CSV table and return its path."""
    rows = sorted(results, key=lambda r: r.f_Hz)
    suffix = "" if window_mode == "canonical" else f"_{window_mode}"
    csv_path = out_dir / f"rw_kr_vs_freq_{method}{suffix}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "f_Hz", "k_rad_m", "L_m", "H_I_m", "H_R_m", "Kr", "singularity_ok"])
        for r in rows:
            w.writerow([
                r.test_id, f"{r.f_Hz:.6f}", f"{r.k:.6f}", f"{r.wavelength_m:.6f}",
                f"{r.H_I:.6f}", f"{r.H_R:.6f}", f"{r.Kr:.6f}", int(r.singularity_ok),
            ])
    print(f"[run_analysis] wrote {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
