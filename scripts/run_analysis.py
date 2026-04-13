"""Command-line entry point for reflection coefficient analysis.

Each run must declare three things up front so it is unambiguous whether the
analysis is for regular or irregular waves, and against which probe array:

* **scheme**   — ``rw`` (regular wave) or ``wn`` (irregular / white-noise)
* **array**    — ``middle_tank`` or ``near_beach``
* **data root** — folder containing ``tank_config.json``, ``metadata/`` and the
  ``{rw,wn}/{middle_tank,near_beach}/`` data directories

Any of the three may be supplied on the command line; anything omitted is
prompted for interactively so no run can start without explicit scheme
selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reflection_coefficient.io import (
    ENV_VAR,
    list_tests,
    load_probe_data,
    resolve_data_root,
    validate_data_root,
)

SCHEME_LABELS = {
    "rw": "REGULAR WAVE",
    "wn": "WHITE-NOISE IRREGULAR WAVE",
    "jonswap": "JONSWAP IRREGULAR WAVE",
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


def _resolve_selection(args: argparse.Namespace) -> tuple[Path, str, str]:
    # Data root first, because the error message it produces is the most
    # useful signal if the user's path is wrong.
    try:
        data_root = resolve_data_root(args.data_root)
        validate_data_root(data_root)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    scheme = args.scheme or _prompt_choice(
        "Select wave scheme for this run:", ("rw", "wn", "jonswap")
    )
    array = args.array or _prompt_array(data_root, scheme)
    return data_root, scheme, array


def _prompt_array(data_root: Path, scheme: str) -> str:
    campaign_dir = data_root / scheme
    found = sorted(p.name for p in campaign_dir.iterdir() if p.is_dir()) \
        if campaign_dir.is_dir() else []
    if not found:
        raise SystemExit(
            f"No probe-array subfolders found under {campaign_dir}. "
            "Create one (any name) and place your test .txt files in it."
        )
    return _prompt_choice("Select probe array folder:", tuple(found))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute wave tank reflection coefficient.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help=(
            f"Root of the experiment_data tree. Falls back to ${ENV_VAR} then "
            "the in-repo default."
        ),
    )
    parser.add_argument(
        "--scheme", choices=["rw", "wn", "jonswap"], default=None,
        help="Wave scheme: 'rw' = regular wave, 'wn' = white-noise irregular, "
             "'jonswap' = JONSWAP irregular. Prompted if omitted.",
    )
    parser.add_argument(
        "--array", default=None,
        help="Name of the probe-array subfolder under <data_root>/<scheme>/. "
             "Prompted from the available folders if omitted.",
    )
    parser.add_argument(
        "--test", default="all",
        help="Test id (e.g. RW005) or 'all' for every test found.",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--method", choices=["goda", "least_squares"], default="least_squares")
    parser.add_argument(
        "--list", action="store_true",
        help="List tests discovered under the chosen data root and exit.",
    )
    args = parser.parse_args()

    data_root, scheme, array = _resolve_selection(args)

    banner = f" {SCHEME_LABELS[scheme]} | array={array} | root={data_root} "
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))

    available = list_tests(scheme, array, data_root)
    print(f"[run_analysis] {len(available)} test(s) found for {scheme}/{array}")

    if args.list:
        for tid in available:
            print(" ", tid)
        return

    if args.test == "all":
        selected = available
    elif args.test in available:
        selected = [args.test]
    else:
        print(f"Test {args.test!r} not found under {data_root}.", file=sys.stderr)
        sys.exit(2)

    args.output.mkdir(parents=True, exist_ok=True)
    for tid in selected:
        t, eta1, eta2, eta3, meta = load_probe_data(
            tid, array=array, campaign=scheme, data_root=data_root,
        )
        print(f"[run_analysis] {tid}: N={len(t)}, fs≈{1/(t[1]-t[0]):.1f} Hz")
        # TODO: wire up preprocessing -> method -> save results


if __name__ == "__main__":
    main()
