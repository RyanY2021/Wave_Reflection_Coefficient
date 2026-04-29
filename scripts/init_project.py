"""CLI: create the input scaffold (tank_config.json + metadata CSVs + data dirs).

Run once before the first analysis. Paths follow the same resolution rules as
``run_analysis.py`` (CLI arg > stored user config > built-in default), and any
explicit path supplied here is persisted for later runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reflection_coefficient.init_project import init_project
from reflection_coefficient.io import (
    USER_CONFIG_PATH,
    resolve_cn_config,
    resolve_data_dir,
    resolve_metadata_dir,
    resolve_probes_config,
    resolve_tank_config,
    save_paths,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tank-config", type=Path, default=None)
    p.add_argument("--metadata-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--probes-config", type=Path, default=None)
    p.add_argument("--cn-config", type=Path, default=None)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing files (default: skip if present).")
    args = p.parse_args()

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
        print(f"[init_project] updated saved paths in {USER_CONFIG_PATH}")

    print(f"  tank_config   = {resolve_tank_config(args.tank_config)}")
    print(f"  metadata_dir  = {resolve_metadata_dir(args.metadata_dir)}")
    print(f"  data_dir      = {resolve_data_dir(args.data_dir)}")
    print(f"  probes_config = {resolve_probes_config(args.probes_config)}")
    print(f"  cn_config     = {resolve_cn_config(args.cn_config)}")

    for line in init_project(
        tank_config=args.tank_config,
        metadata_dir=args.metadata_dir,
        data_dir=args.data_dir,
        probes_config=args.probes_config,
        cn_config=args.cn_config,
        force=args.force,
    ):
        print(" ", line)

    print(
        "\nNext steps:\n"
        "  1. Fill probe_geometry fields in tank_config.json.\n"
        "  2. (Optional) Fill scale_*/offset_* for each probe in probes.json\n"
        "     and enable correction with: run_analysis.py --recalibrate\n"
        "  3. Add per-test rows to metadata/{rw,wn,js}.csv.\n"
        "  4. Drop raw <TEST_ID>.txt files into data_dir/ (or rw/ wn/ js/ subfolders).\n"
        "  5. Run: python scripts/run_analysis.py --scheme rw --test all\n"
        "  6. (Optional) Fit per-probe complex correction C_n on the noref window:\n"
        "     run_analysis.py --scheme rw --test all --window-mode noref --cn-fit\n"
        "     then apply with --cn-apply on the canonical window."
    )


if __name__ == "__main__":
    main()
