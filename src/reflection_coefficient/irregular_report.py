"""Self-contained HTML report for a single irregular-wave test (wn / js).

Unlike the regular-wave report (which aggregates a family of single-frequency
tests), each irregular test produces its own spectra and ``Kr(f)`` curve, so
the HTML is keyed to a single ``IrregularResult`` + its ``TestMeta``.

Outputs written to ``out_dir``:

* ``<test_id>_spectra_<method>.png`` — incident vs reflected spectral density
* ``<test_id>_kr_f_<method>.png``   — Kr(f) across the analysis band
* ``<test_id>_singularity_<method>.png`` — D or sin²(kΔ) vs f
* ``<test_id>_spectrum_<method>.csv`` — smoothed S_I, S_R, Kr(f) table
* ``<test_id>_report_<method>.html`` — self-contained report
"""

from __future__ import annotations

import base64
import csv
import html
import io as _io
import math
from pathlib import Path

import numpy as np

from .analysis import group_velocity, solve_dispersion
from .io import TestMeta
from .pipeline import IrregularResult, clip_bounds

# Re-use the stylesheet + tank SVG + geometry cards from rw_report.
from .rw_report import _CSS, _geometry_cards, _tank_svg  # noqa: F401


def _window_info(result: IrregularResult, meta: TestMeta) -> dict:
    depth = meta.water_depth_m
    g = meta.gravity_m_s2
    f_peak = result.diagnostics["f_peak_used_Hz"]
    # Metadata band (WN) takes precedence; fall back to ±peak-factor (JS/§7).
    f_lo = meta.f_min_Hz if meta.f_min_Hz else 0.5 * f_peak
    f_hi = meta.f_max_Hz if meta.f_max_Hz else 2.5 * f_peak
    cg_slow = group_velocity(f_lo, depth, g=g)
    cg_fast = group_velocity(f_hi, depth, g=g)
    t_start, _ = clip_bounds(cg_slow, meta.x_paddle_to_wp1_m,
                             meta.x_wp3_to_struct_m, meta.X13_m)
    _, t_end = clip_bounds(cg_fast, meta.x_paddle_to_wp1_m,
                           meta.x_wp3_to_struct_m, meta.X13_m)
    _, L_peak = solve_dispersion(f_peak, depth, g=g)
    return {
        "f_peak": f_peak, "f_lo": f_lo, "f_hi": f_hi,
        "cg_slow": cg_slow, "cg_fast": cg_fast,
        "t_start": float(t_start), "t_end": float(t_end),
        "L_peak": L_peak,
    }


def _png_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = _io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _plot_spectra(result: IrregularResult, win: dict, out_path: Path) -> str:
    import matplotlib.pyplot as plt
    f = result.f_smooth
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f, result.S_I_smooth, label="Incident $S_I$", color="#3266ad")
    ax.plot(f, result.S_R_smooth, label="Reflected $S_R$", color="#D85A30")
    ax.axvspan(win["f_lo"], win["f_hi"], color="#3266ad", alpha=0.06,
               label=f"analysis band {win['f_lo']:.2f}–{win['f_hi']:.2f} Hz")
    ax.axvline(win["f_peak"], linestyle="--", color="#534AB7",
               linewidth=1, label=f"f_peak = {win['f_peak']:.2f} Hz")
    pad = 0.1 * (win["f_hi"] - win["f_lo"])
    ax.set_xlim(max(0.0, win["f_lo"] - pad), win["f_hi"] + pad)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("S(f) [m²/Hz]")
    ax.set_title("Smoothed incident & reflected spectra")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    b64 = _png_b64(fig)
    out_path.write_bytes(base64.b64decode(b64))
    return b64


def _plot_kr_f(result: IrregularResult, win: dict, out_path: Path) -> str:
    import matplotlib.pyplot as plt
    f = result.f_smooth
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f, result.Kr_f, color="#0F6E56", marker=".", linestyle="-")
    ax.axvspan(win["f_lo"], win["f_hi"], color="#3266ad", alpha=0.06)
    ax.axvline(win["f_peak"], linestyle="--", color="#534AB7", linewidth=1)
    pad = 0.1 * (win["f_hi"] - win["f_lo"])
    ax.set_xlim(max(0.0, win["f_lo"] - pad), win["f_hi"] + pad)
    ax.axhline(result.Kr_overall, linestyle=":", color="#A32D2D",
               linewidth=1, label=f"$K_r$ overall = {result.Kr_overall:.3f}")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(r"$K_r(f)$")
    ax.set_ylim(bottom=0)
    ax.set_title("Reflection coefficient vs frequency")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    b64 = _png_b64(fig)
    out_path.write_bytes(base64.b64decode(b64))
    return b64


def _plot_singularity(result: IrregularResult, meta: TestMeta,
                      method: str, out_path: Path, win: dict) -> str:
    import matplotlib.pyplot as plt
    from .analysis import solve_dispersion_array
    f = result.f
    k = solve_dispersion_array(f, meta.water_depth_m, g=meta.gravity_m_s2)
    if method == "goda":
        metric = np.sin(k * meta.X13_m) ** 2
        threshold = 0.05
        label = r"$\sin^2(k X_{13})$ (Goda)"
    else:
        sb = np.sin(k * meta.X12_m)
        sg = np.sin(k * meta.X13_m)
        sgb = np.sin(k * meta.X13_m - k * meta.X12_m)
        metric = 2.0 * (sb * sb + sg * sg + sgb * sgb)
        threshold = 0.1
        label = r"$D$ (Mansard–Funke)"
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(f, metric, color="#5f5e5a", linewidth=1)
    ax.axvspan(win["f_lo"], win["f_hi"], color="#3266ad", alpha=0.06)
    ax.axhline(threshold, linestyle="--", color="#A32D2D",
               linewidth=1, label=f"threshold = {threshold}")
    pad = 0.1 * (win["f_hi"] - win["f_lo"])
    ax.set_xlim(max(0.0, win["f_lo"] - pad), win["f_hi"] + pad)
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(label)
    ax.set_title("Singularity metric")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    b64 = _png_b64(fig)
    out_path.write_bytes(base64.b64decode(b64))
    return b64


def _write_spectrum_csv(result: IrregularResult, out_path: Path) -> Path:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f_Hz", "S_I_m2_Hz", "S_R_m2_Hz", "Kr_f"])
        for fi, si, sr, kr in zip(
            result.f_smooth, result.S_I_smooth, result.S_R_smooth, result.Kr_f
        ):
            w.writerow([
                f"{fi:.6f}",
                f"{si:.6e}" if np.isfinite(si) else "",
                f"{sr:.6e}" if np.isfinite(sr) else "",
                f"{kr:.6f}" if np.isfinite(kr) else "",
            ])
    return out_path


def _summary_cards(result: IrregularResult, win: dict) -> str:
    d = result.diagnostics
    cards = [
        ("f<sub>peak</sub>", f"{win['f_peak']:.3f} Hz"),
        ("L at f<sub>peak</sub>", f"{win['L_peak']:.2f} m"),
        ("Clip t<sub>start</sub> / t<sub>end</sub>",
         f"{win['t_start']:.1f} / {win['t_end']:.1f} s"),
        ("H<sub>m0,I</sub>", f"{result.Hm0_I:.4f} m"),
        ("H<sub>m0,R</sub>", f"{result.Hm0_R:.4f} m"),
        ("T<sub>p,I</sub>", f"{result.Tp_I:.3f} s" if np.isfinite(result.Tp_I) else "—"),
        ("K<sub>r,overall</sub>", f"{result.Kr_overall:.3f}"),
        ("Valid bins / min D(sin²)",
         f"{d['n_bins_valid']} / {d['D_or_sin2_min']:.3f}"),
        ("Window / resolution bw",
         f"{d.get('window', '—')} / {d.get('bandwidth_Hz', float('nan')):.4f} Hz "
         f"({d.get('n_bands', '?')} bins)"),
    ]
    parts = []
    for label, value in cards:
        parts.append(
            f'<div style="background:var(--color-background-secondary);'
            f'border-radius:var(--border-radius-md);padding:0.75rem;">'
            f'<p style="font-size:11px;color:var(--color-text-secondary);margin:0;">{label}</p>'
            f'<p style="font-size:16px;font-weight:500;margin:2px 0 0;">{value}</p></div>'
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));'
        'gap:10px;margin-bottom:1.5rem;">' + "".join(parts) + "</div>"
    )


def _img_tag(b64: str, alt: str) -> str:
    return (
        f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" '
        f'style="max-width:100%;height:auto;border-radius:var(--border-radius-md);'
        f'background:var(--color-background-secondary);padding:0.5rem;"/>'
    )


def write_irregular_report(
    result: IrregularResult,
    meta: TestMeta,
    out_dir: Path,
    method: str,
    timestamp: str | None = None,
) -> Path:
    """Write the single-test irregular-wave report and supporting files.

    Returns the path to the generated HTML file.
    """
    win = _window_info(result, meta)
    stem = f"{result.test_id}_{method}"

    spec_png = out_dir / f"{stem}_spectra.png"
    kr_png = out_dir / f"{stem}_kr_f.png"
    sing_png = out_dir / f"{stem}_singularity.png"
    csv_path = out_dir / f"{stem}_spectrum.csv"
    b64_spec = _plot_spectra(result, win, spec_png)
    b64_kr = _plot_kr_f(result, win, kr_png)
    b64_sing = _plot_singularity(result, meta, method, sing_png, win)
    _write_spectrum_csv(result, csv_path)

    header = (
        f"<h1>Irregular-wave reflection report — {html.escape(result.test_id)} "
        f"({html.escape(meta.campaign)}, method: {html.escape(method)})</h1>"
    )
    if timestamp:
        header += (
            f'<p style="color:var(--color-text-secondary);margin-bottom:1rem;">'
            f'Generated {html.escape(timestamp)}</p>'
        )

    layout_block = (
        '<div style="background:var(--color-background-secondary);'
        'border-radius:var(--border-radius-lg);padding:1.25rem;margin-bottom:1.5rem;">'
        '<p style="font-size:13px;color:var(--color-text-secondary);margin:0 0 10px;">'
        f'Tank layout (depth {meta.water_depth_m:g} m)</p>'
        f'{_tank_svg(meta)}</div>'
    )
    # _geometry_cards expects per-test t_gen rows; reuse with a single synthetic row.
    geo_rows = [{"t_gen": meta.t_gen_s}]

    body = "\n".join([
        header,
        layout_block,
        _geometry_cards(meta, geo_rows),
        "<h2>Summary</h2>",
        _summary_cards(result, win),
        "<h2>Incident & reflected spectra</h2>",
        _img_tag(b64_spec, "spectra"),
        "<h2>K<sub>r</sub>(f)</h2>",
        _img_tag(b64_kr, "Kr vs f"),
        "<h2>Singularity metric</h2>",
        _img_tag(b64_sing, "singularity metric"),
    ])

    html_doc = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f'<title>{html.escape(result.test_id)} reflection report</title>'
        f'<style>{_CSS}</style></head><body>{body}</body></html>'
    )
    out_path = out_dir / f"{stem}_report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
