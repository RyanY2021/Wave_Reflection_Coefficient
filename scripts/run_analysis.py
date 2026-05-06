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
from reflection_coefficient.cn_correction import (
    fit_cn_from_records,
    load_cn_config,
    save_cn_config,
)
from reflection_coefficient.io import (
    USER_CONFIG_PATH,
    TestMeta,
    list_tests,
    load_probe_data,
    load_probes_config,
    resolve_cn_alpha_mode,
    resolve_cn_apply,
    resolve_cn_config,
    resolve_cn_mode,
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
    save_cn_alpha_mode,
    save_cn_apply,
    save_cn_mode,
    save_drops,
    save_freq_source,
    save_goda_pair,
    save_method,
    save_paths,
    save_recalibrate,
    save_window,
)
from reflection_coefficient.irregular_report import write_irregular_report
from reflection_coefficient.pipeline import (
    IrregularResult,
    RegularResult,
    analyse,
    extract_regular_bins,
)
from reflection_coefficient.rw_report import singularity_metric, write_rw_report

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
            "(default: on)."
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
            "Regular-wave only. 'target' (default) evaluates a single-point "
            "DFT at exactly meta.f_Hz, bypassing bin quantisation — assumes "
            "the wave maker holds the commanded frequency across the test. "
            "'bin' snaps to the nearest FFT bin (the legacy behaviour) and "
            "is mainly useful as a diagnostic comparison. Persisted when set."
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
        "--cn-config", type=Path, default=None,
        help=(
            "Per-probe complex correction JSON (default: "
            "experiment_data/probes_refined.json). Persisted when set. "
            "Consumed when --cn-apply is on; written by --cn-fit."
        ),
    )
    parser.add_argument(
        "--cn-fit", action="store_true",
        help=(
            "Fit per-probe complex correction C_n(f) = alpha * exp(-i k Δx "
            "- i ω Δt) from the noref window across the selected tests, then "
            "write the result to --cn-config (default probes_refined.json). "
            "Requires --window-mode noref. For --scheme rw, fitting needs "
            "--test all so the (k, ω) regression is well-conditioned. "
            "Not persisted (action-only)."
        ),
    )
    parser.add_argument(
        "--cn-apply", default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Apply the per-probe complex correction from --cn-config to FFT "
            "bins before separation. Use --no-cn-apply to disable. Persisted "
            "when set (default: off — opt-in, since a stale fit silently "
            "corrupts separation)."
        ),
    )
    parser.add_argument(
        "--cn-mode", choices=["amp", "phase", "both"], default=None,
        help=(
            "Which component of C_n to apply: 'amp' (alpha only, Variant B), "
            "'phase' (Δx + Δt only, Variant C), or 'both' (Variant D, full "
            "fit). Persisted when set (default: both). Ignored when "
            "--cn-apply is off."
        ),
    )
    parser.add_argument(
        "--cn-alpha-mode", choices=["scalar", "dynamic"], default=None,
        help=(
            "How alpha is evaluated when applying C_n. 'dynamic' (default) "
            "linearly interpolates the per-bin alpha table stored in "
            "probes_refined.json (per_bin.alpha) over frequency, falling "
            "back to the scalar masked-mean alpha for bins outside the "
            "table's frequency support. 'scalar' uses the masked-mean "
            "alpha at every frequency (legacy behaviour). The scalar "
            "fallback's frequency mask is editable in probes_refined.json "
            "under fit_mask. Persisted when set; ignored when --cn-apply "
            "is off or --cn-mode phase."
        ),
    )
    parser.add_argument(
        "--cn-fit-freq", choices=["bin", "target"], default="target",
        help=(
            "How to pick the per-record frequency at which C_n is fitted. "
            "'target' (default) evaluates a single-point DFT at exactly "
            "meta.f_Hz from rw.csv — leakage-free, independent of the noref "
            "window length. 'bin' snaps to the nearest FFT bin (same as "
            "--freq-source bin); on short noref windows df = 1/T is coarse "
            "and the snapped frequency drifts from the target, which biases "
            "the per-bin C_n^obs and corrupts the phase regression. Only "
            "consulted when --cn-fit is on; not persisted."
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
        args.tank_config, args.metadata_dir, args.data_dir,
        args.probes_config, args.cn_config,
    )):
        save_paths(
            tank_config=args.tank_config,
            metadata_dir=args.metadata_dir,
            data_dir=args.data_dir,
            probes_config=args.probes_config,
            cn_config=args.cn_config,
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

    if args.cn_apply is not None:
        save_cn_apply(args.cn_apply)
    args.cn_apply = resolve_cn_apply(args.cn_apply)

    if args.cn_mode is not None:
        save_cn_mode(args.cn_mode)
    args.cn_mode = resolve_cn_mode(args.cn_mode)

    if args.cn_alpha_mode is not None:
        save_cn_alpha_mode(args.cn_alpha_mode)
    args.cn_alpha_mode = resolve_cn_alpha_mode(args.cn_alpha_mode)

    tank_cfg = resolve_tank_config(args.tank_config)
    meta_dir = resolve_metadata_dir(args.metadata_dir)
    data_dir = resolve_data_dir(args.data_dir)
    probes_cfg_path = resolve_probes_config(args.probes_config)
    cn_config_path = resolve_cn_config(args.cn_config)

    if args.show_paths:
        print(f"  tank_config   = {tank_cfg}")
        print(f"  metadata_dir  = {meta_dir}")
        print(f"  data_dir      = {data_dir}")
        print(f"  probes_config = {probes_cfg_path}")
        print(f"  cn_config     = {cn_config_path}")
        print(f"  method        = {args.method}")
        bw_txt = f"{args.bandwidth:g} Hz" if args.bandwidth is not None else "—"
        print(f"  window        = {args.window}  (bandwidth: {bw_txt})")
        print(
            f"  drops         = head {args.head_drop:g} s, "
            f"tail {args.tail_drop:g} s"
        )
        print(f"  recalibrate   = {'on' if args.recalibrate else 'off'}")
        print(f"  cn_apply      = {'on' if args.cn_apply else 'off'} "
              f"(mode: {args.cn_mode}, alpha: {args.cn_alpha_mode})")
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

    # cn-fit / cn-apply mutually exclusive within one invocation: do one or
    # the other, not both. The fit consumes raw bins; applying first would
    # absorb the existing correction into the new fit and produce identity.
    if args.cn_fit and args.cn_apply:
        print(
            "[run_analysis] --cn-fit and --cn-apply cannot be combined in one "
            "run. Fit produces a new probes_refined.json; apply consumes it. "
            "Run them as two separate commands.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.cn_fit and args.window_mode != "noref":
        print(
            "[run_analysis] --cn-fit requires --window-mode noref "
            "(the fit assumes a_R = 0; the canonical window has reflection).",
            file=sys.stderr,
        )
        sys.exit(1)

    cn_config_data = None
    if args.cn_apply:
        if not cn_config_path.exists():
            print(
                f"[run_analysis] --cn-apply requires a cn config at "
                f"{cn_config_path} (run scripts/init_project.py to scaffold "
                f"a placeholder, or --cn-fit to produce a real one).",
                file=sys.stderr,
            )
            sys.exit(1)
        cn_config_data = load_cn_config(cn_config_path)
        fitted_with = cn_config_data.get("fit_meta", {}).get(
            "fitted_with_recalibrate"
        )
        if fitted_with is not None and bool(fitted_with) != bool(args.recalibrate):
            print(
                f"[run_analysis] WARNING: --cn-apply config was fitted with "
                f"recalibrate={'on' if fitted_with else 'off'} but current "
                f"--recalibrate is {'on' if args.recalibrate else 'off'}. "
                f"Results may be biased. Re-fit with the matching setting.",
                file=sys.stderr,
            )

    scheme = args.scheme or _prompt_choice(
        "Select wave scheme for this run:", ("rw", "wn", "js")
    )

    if args.cn_fit and scheme != "rw":
        print(
            "[run_analysis] --cn-fit currently supports --scheme rw only. "
            "Single-record irregular fitting is a planned follow-up.",
            file=sys.stderr,
        )
        sys.exit(1)

    bw_txt = f"{args.bandwidth:g} Hz" if args.bandwidth is not None else "—"
    pair_txt = (
        f" | goda_pair={args.goda_pair}" if args.method == "goda" else ""
    )
    cn_txt = (
        f" | cn=fit(freq={args.cn_fit_freq})"
        if args.cn_fit
        else f" | cn=on({args.cn_mode},α={args.cn_alpha_mode})"
        if args.cn_apply
        else ""
    )
    banner = (
        f" {SCHEME_LABELS[scheme]} | method={args.method}{pair_txt} "
        f"| window={args.window} (bw {bw_txt}) "
        f"| drops head {args.head_drop:g}s tail {args.tail_drop:g}s "
        f"| recalibrate={'on' if args.recalibrate else 'off'}{cn_txt} "
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

    if args.cn_fit and len(selected) < 4:
        print(
            f"[run_analysis] WARNING: --cn-fit selected {len(selected)} test(s); "
            f"the (k, ω) regression is rank-2 so at least ~4 distinct frequencies "
            f"are needed for a reliable Δx and Δt. With fewer, alpha is fine but "
            f"the phase fit will be ill-conditioned. Pass --test all on a richer "
            f"campaign for a robust fit.",
            file=sys.stderr,
        )

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
    cn_records: list[dict] = []
    cn_test_ids: list[str] = []
    cn_geometry: tuple[float, float] | None = None
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
                cn_config=cn_config_data,
                cn_mode=args.cn_mode,
                cn_alpha_mode=args.cn_alpha_mode,
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

        # Accumulate FFT bins for the cn-fit branch (regular path only).
        # The fit is decoupled from the persisted --freq-source: the noref
        # window is short, so a bin-snapped frequency can sit far from the
        # rw.csv target. Default cn_fit_freq='target' fixes the fit to the
        # metadata frequency via a single-point DFT.
        if args.cn_fit and scheme == "rw":
            try:
                f_used, k_val, b1, b2, b3 = extract_regular_bins(
                    t, eta1, eta2, eta3, meta,
                    head_drop_s=args.head_drop, tail_drop_s=args.tail_drop,
                    window_mode=args.window_mode,
                    freq_source=args.cn_fit_freq,
                )
            except Exception as exc:
                print(f"  !! {tid} (cn_fit): {exc}", file=sys.stderr)
            else:
                cn_records.append({
                    "f": [f_used], "k": [k_val],
                    "B1": [b1], "B2": [b2], "B3": [b3],
                })
                cn_test_ids.append(tid)
                cn_geometry = (float(meta.X12_m), float(meta.X13_m))

    if scheme == "rw" and len(regular_results) >= 2:
        out = _ensure_run_dir()
        csv_path = _write_kr_vs_freq(
            list(zip(regular_results, regular_metas)),
            out, args.method, window_mode=args.window_mode,
        )
        html_path = write_rw_report(
            list(zip(regular_results, regular_metas)),
            out, args.method,
            csv_path=csv_path, timestamp=out.name,
            window_mode=args.window_mode,
        )
        print(f"[run_analysis] wrote {html_path}")

    if args.cn_fit:
        if not cn_records or cn_geometry is None:
            print(
                "[run_analysis] --cn-fit: no usable records collected.",
                file=sys.stderr,
            )
            sys.exit(1)
        X12, X13 = cn_geometry
        # Preserve any user-edited fit_mask in the existing JSON so a re-fit
        # honours their narrowed band. If the file is absent or unreadable,
        # fall back to "no mask" (data range) — the fit code defaults safely.
        existing_fit_mask: dict | None = None
        if cn_config_path.exists():
            try:
                existing_cfg = load_cn_config(cn_config_path)
            except Exception:
                existing_cfg = None
            if existing_cfg is not None:
                existing_fit_mask = existing_cfg.get("fit_mask")
        try:
            cn_dict = fit_cn_from_records(
                cn_records, X12=X12, X13=X13,
                existing_fit_mask=existing_fit_mask,
            )
        except Exception as exc:
            print(f"[run_analysis] --cn-fit failed: {exc}", file=sys.stderr)
            sys.exit(1)
        fit_meta = {
            "fitted_from_test_ids": cn_test_ids,
            "fitted_from_window_mode": args.window_mode,
            "fitted_with_recalibrate": bool(args.recalibrate),
            "fitted_freq_source": args.cn_fit_freq,
            "geometry": {"X12_m": X12, "X13_m": X13},
            "n_records": len(cn_records),
        }
        save_cn_config(cn_config_path, cn_dict, fit_meta=fit_meta)
        print(f"[run_analysis] wrote {cn_config_path}")
        fm = cn_dict.get("fit_mask", {})
        if fm:
            preserved = " (preserved from existing file)" if existing_fit_mask else ""
            print(
                f"  [cn_fit] scalar-α mask: "
                f"f∈[{fm.get('f_min_Hz', '?'):.4f}, "
                f"{fm.get('f_max_Hz', '?'):.4f}] Hz{preserved}"
            )
        for key in ("wp2", "wp3"):
            entry = cn_dict[key]
            d = entry.get("fit_diagnostics", {})
            print(
                f"  [cn_fit] {key} α_scalar={entry['alpha']:.4f} "
                f"Δx={entry['delta_x_m']*1000:+.2f} mm "
                f"Δt={entry['delta_t_s']*1e6:+.1f} µs "
                f"(n_bins={d.get('n_bins_used', '?')}, "
                f"in_mask={d.get('n_bins_in_mask', '?')}, "
                f"residual={d.get('residual_rms_rad', float('nan')):.3f} rad)"
            )


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
    pairs: list[tuple[RegularResult, TestMeta]], out_dir: Path, method: str,
    window_mode: str = "canonical",
) -> Path:
    """Aggregate regular-wave runs into a Kr(f) CSV table and return its path."""
    rows = sorted(pairs, key=lambda p: p[0].f_Hz)
    suffix = "" if window_mode == "canonical" else f"_{window_mode}"
    csv_path = out_dir / f"rw_kr_vs_freq_{method}{suffix}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "f_Hz", "k_rad_m", "L_m",
                    "a_I_m", "a_R_m", "Kr",
                    "singularity_metric", "singularity_threshold"])
        for r, meta in rows:
            sing, thr, _ = singularity_metric(r, meta, method)
            w.writerow([
                r.test_id, f"{r.f_Hz:.6f}", f"{r.k:.6f}", f"{r.wavelength_m:.6f}",
                f"{r.a_I:.6f}", f"{r.a_R:.6f}", f"{r.Kr:.6f}",
                f"{sing:.6f}", f"{thr:g}",
            ])
    print(f"[run_analysis] wrote {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
