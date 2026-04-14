"""Self-contained HTML report for a regular-wave (``--scheme rw``) run.

Renders the tank layout, geometry summary, per-test time-window table (f, T,
L, c, cg, t_gen, t_start, t_end, T_usable, N_per, Kr, singularity flag) and
embeds the Kr-vs-frequency curve PNG produced by ``run_analysis.py``.

Inputs are deliberately minimal: a list of :class:`RegularResult` from
:mod:`reflection_coefficient.pipeline`, one representative :class:`TestMeta`
for tank geometry, and optional paths to the CSV / PNG generated alongside.
"""

from __future__ import annotations

import base64
import csv
import html
import math
from pathlib import Path

from .analysis import group_velocity, solve_dispersion
from .io import TestMeta
from .pipeline import RegularResult, clip_bounds


def _row_for(result: RegularResult, meta: TestMeta) -> dict:
    depth = meta.water_depth_m
    g = meta.gravity_m_s2
    x_paddle = meta.x_paddle_to_wp1_m
    x_struct = meta.x_wp3_to_struct_m
    X13 = meta.X13_m

    f = result.f_Hz
    T = 1.0 / f
    k, L = solve_dispersion(f, depth, g=g)
    omega = 2.0 * math.pi * f
    c = omega / k
    cg = group_velocity(f, depth, g=g)
    t_start, t_end_raw = clip_bounds(cg, x_paddle, x_struct, X13)
    t_gen = meta.t_gen_s

    if t_gen is None:
        t_end = t_end_raw
        status = "OK"
    elif t_start >= t_gen:
        t_end = t_start
        status = "NO_WINDOW"
    elif t_end_raw > t_gen:
        t_end = t_gen
        status = "CLAMPED"
    else:
        t_end = t_end_raw
        status = "OK"
    t_use = max(t_end - t_start, 0.0)
    n_per = t_use / T if t_use > 0 else 0.0
    if 0 < n_per < 5:
        status += "_FEW"
    if not result.singularity_ok:
        status += "+SING"
    return {
        "test_id": result.test_id,
        "f": f, "T": T, "L": L, "c": c, "cg": cg,
        "t_gen": t_gen, "t_start": t_start, "t_end": t_end,
        "t_use": t_use, "n_per": n_per,
        "H_I": result.H_I, "H_R": result.H_R, "Kr": result.Kr,
        "status": status, "singularity_ok": result.singularity_ok,
    }


def _status_colors(status: str) -> tuple[str, str]:
    if status.startswith("OK") and "_FEW" not in status and "+SING" not in status:
        return "var(--color-background-success)", "#0F6E56"
    if status.startswith("CLAMPED") and "_FEW" not in status and "+SING" not in status:
        return "var(--color-background-warning)", "#BA7517"
    return "var(--color-background-danger)", "#A32D2D"


_CSS = """
:root {
  --color-background-primary:#ffffff;--color-background-secondary:#f5f5f2;
  --color-background-success:#EAF3DE;--color-background-warning:#FAEEDA;
  --color-background-danger:#FCEBEB;--color-text-primary:#1a1a1a;
  --color-text-secondary:#5f5e5a;--color-border-tertiary:rgba(0,0,0,0.15);
  --color-border-secondary:rgba(0,0,0,0.3);
  --font-sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;
  --font-mono:'SF Mono','Consolas','Courier New',monospace;
  --border-radius-md:8px;--border-radius-lg:12px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --color-background-primary:#1a1a1a;--color-background-secondary:#2c2c2a;
    --color-background-success:#173404;--color-background-warning:#412402;
    --color-background-danger:#501313;--color-text-primary:#e8e8e8;
    --color-text-secondary:#b4b2a9;--color-border-tertiary:rgba(255,255,255,0.15);
    --color-border-secondary:rgba(255,255,255,0.3);
  }
}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:var(--font-sans);color:var(--color-text-primary);
     background:var(--color-background-primary);padding:1.5rem;line-height:1.7;}
h1{font-size:22px;font-weight:500;margin-bottom:1rem;}
h2{font-size:16px;font-weight:500;margin:1.5rem 0 0.5rem;}
.mono{font-family:var(--font-mono);}
"""


def _tank_svg(meta: TestMeta) -> str:
    tank_len = meta.tank_length_m or 0.0
    x_p = meta.x_paddle_to_wp1_m or 0.0
    X12 = meta.X12_m or 0.0
    X13 = meta.X13_m or 0.0
    if tank_len <= 0:
        return ""
    def px(x_m: float) -> float:
        return 20.0 + 660.0 * (x_m / tank_len)
    wp1 = px(x_p)
    wp2 = px(x_p + X12)
    wp3 = px(x_p + X13)
    return f"""
<svg viewBox="0 0 700 70" style="width:100%;height:auto;" xmlns="http://www.w3.org/2000/svg">
  <rect x="20" y="25" width="660" height="20" rx="2" fill="none"
        stroke="var(--color-border-secondary)" stroke-width="1"/>
  <rect x="20" y="25" width="6" height="20" rx="1" fill="#3266ad"/>
  <rect x="674" y="25" width="6" height="20" rx="1" fill="#73726c"/>
  <text x="23" y="18" font-size="10" fill="var(--color-text-secondary)">Paddle (x=0)</text>
  <text x="620" y="18" font-size="10" fill="var(--color-text-secondary)">Structure (x={tank_len:g})</text>
  <line x1="{wp1:.1f}" y1="25" x2="{wp1:.1f}" y2="45" stroke="#D85A30" stroke-width="2"/>
  <text x="{wp1:.1f}" y="58" font-size="9" fill="#D85A30" text-anchor="middle">WP1 ({x_p:.2f})</text>
  <line x1="{wp2:.1f}" y1="25" x2="{wp2:.1f}" y2="45" stroke="#1D9E75" stroke-width="2"/>
  <text x="{wp2:.1f}" y="66" font-size="9" fill="#1D9E75" text-anchor="middle">WP2 ({x_p + X12:.2f})</text>
  <line x1="{wp3:.1f}" y1="25" x2="{wp3:.1f}" y2="45" stroke="#534AB7" stroke-width="2"/>
  <text x="{wp3 + 18:.1f}" y="58" font-size="9" fill="#534AB7" text-anchor="start">WP3 ({x_p + X13:.2f})</text>
</svg>
"""


def _geometry_cards(meta: TestMeta, rows: list[dict]) -> str:
    x_struct = meta.x_wp3_to_struct_m
    t_gens = sorted({r["t_gen"] for r in rows if r["t_gen"] is not None})
    t_gen_txt = " / ".join(f"{v:g}" for v in t_gens) + " s" if t_gens else "—"
    cards = [
        ("x<sub>paddle→wp1</sub>", f"{meta.x_paddle_to_wp1_m:g} m"),
        ("x<sub>wp3→struct</sub>", f"{x_struct:g} m" if x_struct is not None else "—"),
        ("X<sub>12</sub> / X<sub>13</sub>", f"{meta.X12_m:g} / {meta.X13_m:g}"),
        ("depth", f"{meta.water_depth_m:g} m"),
        ("t<sub>gen</sub>", t_gen_txt),
    ]
    parts = []
    for label, value in cards:
        parts.append(
            f'<div style="background:var(--color-background-secondary);'
            f'border-radius:var(--border-radius-md);padding:0.75rem;">'
            f'<p style="font-size:11px;color:var(--color-text-secondary);margin:0;">{label}</p>'
            f'<p style="font-size:16px;font-weight:500;margin:2px 0 0;">{value}</p>'
            f'</div>'
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(5,minmax(0,1fr));'
        'gap:10px;margin-bottom:1.5rem;">' + "".join(parts) + "</div>"
    )


def _table(rows: list[dict]) -> str:
    headers = [
        "test_id", "f (Hz)", "T (s)", "L (m)", "c (m/s)", "c<sub>g</sub> (m/s)",
        "t<sub>gen</sub> (s)", "t<sub>start</sub> (s)", "t<sub>end</sub> (s)",
        "T<sub>usable</sub> (s)", "N<sub>per</sub>",
        "H<sub>I</sub> (m)", "H<sub>R</sub> (m)", "K<sub>r</sub>", "Status",
    ]
    th = "".join(
        f'<th style="padding:6px 6px;text-align:right;font-weight:500;font-size:11px;'
        f'color:var(--color-text-secondary);">{h}</th>' for h in headers
    )
    body = []
    for r in rows:
        bg, fg = _status_colors(r["status"])
        t_gen_cell = f"{r['t_gen']:g}" if r["t_gen"] is not None else "—"
        cells = [
            html.escape(r["test_id"]),
            f"{r['f']:.3f}", f"{r['T']:.3f}", f"{r['L']:.3f}",
            f"{r['c']:.3f}", f"{r['cg']:.3f}", t_gen_cell,
            f"{r['t_start']:.1f}",
            f"{r['t_end']:.1f}" if r["t_use"] > 0 else "—",
            f"{r['t_use']:.1f}" if r["t_use"] > 0 else "—",
            f"{r['n_per']:.1f}" if r["n_per"] > 0 else "—",
            f"{r['H_I']:.4f}", f"{r['H_R']:.4f}", f"{r['Kr']:.3f}",
        ]
        tds = "".join(
            f'<td style="padding:4px 6px;text-align:right;font-family:var(--font-mono);'
            f'font-size:12px;">{c}</td>' for c in cells
        )
        tds += (
            f'<td style="padding:4px 8px;text-align:left;">'
            f'<span style="font-size:11px;padding:2px 7px;border-radius:var(--border-radius-md);'
            f'background:{bg};color:{fg};">{html.escape(r["status"])}</span></td>'
        )
        body.append(
            f'<tr style="border-bottom:0.5px solid var(--color-border-tertiary);">{tds}</tr>'
        )
    return (
        '<div style="overflow-x:auto;margin-bottom:1.5rem;">'
        '<table style="width:100%;border-collapse:collapse;font-size:12.5px;white-space:nowrap;">'
        f'<thead><tr style="border-bottom:2px solid var(--color-border-secondary);">{th}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _gantt_png(rows: list[dict]) -> str:
    """Return an <img> tag with a base64-embedded Gantt chart of time windows.

    Mirrors the Chart.js visual in ``docs/rw_test_matrix/*.html``: per frequency
    a grey 'wait-for-reflection' bar from 0→t_start and a coloured 'usable'
    bar from t_start→t_end, with dashed vertical t_gen limit(s).
    """
    try:
        import io as _io
        import matplotlib.pyplot as plt
    except ImportError:
        return ""
    if not rows:
        return ""

    labels = [f"{r['f']:.2f}" for r in rows]
    y = list(range(len(rows)))
    fig_h = max(3.0, 0.28 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))

    for i, r in enumerate(rows):
        if r["t_use"] <= 0:
            continue
        ax.barh(i, r["t_start"], left=0,
                color=(136/255, 135/255, 128/255, 0.2),
                edgecolor=(136/255, 135/255, 128/255, 0.4), linewidth=0.5)
        clamped = r["status"].startswith("CLAMPED")
        face = (186/255, 117/255, 23/255, 0.45) if clamped else (50/255, 102/255, 173/255, 0.45)
        edge = "#BA7517" if clamped else "#3266ad"
        ax.barh(i, r["t_end"] - r["t_start"], left=r["t_start"],
                color=face, edgecolor=edge, linewidth=0.8)

    t_gens = sorted({r["t_gen"] for r in rows if r["t_gen"] is not None})
    colors = {180: "#E24B4A", 240: "#534AB7"}
    for tg in t_gens:
        ax.axvline(tg, linestyle="--", linewidth=1.3,
                   color=colors.get(int(tg), "#888"))
        ax.text(tg, -0.8, f"{tg:g}s", color=colors.get(int(tg), "#888"),
                fontsize=9, ha="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlim(0, max([r["t_end"] for r in rows] + t_gens + [1.0]) * 1.05)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    legend = (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;margin:8px 0 0;'
        'font-size:12px;color:var(--color-text-secondary);">'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(136,135,128,0.3);margin-right:4px;"></span>Wait for reflections</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(50,102,173,0.45);margin-right:4px;"></span>Usable (OK)</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(186,117,23,0.45);margin-right:4px;"></span>Usable (clamped)</span>'
        '<span><span style="display:inline-block;width:2px;height:10px;'
        'background:#E24B4A;margin-right:4px;"></span>180 s limit</span>'
        '<span><span style="display:inline-block;width:2px;height:10px;'
        'background:#534AB7;margin-right:4px;"></span>240 s limit</span>'
        '</div>'
    )
    return (
        f'<img src="data:image/png;base64,{data}" alt="Time window Gantt" '
        f'style="max-width:100%;height:auto;border-radius:var(--border-radius-md);'
        f'background:var(--color-background-secondary);padding:0.5rem;"/>{legend}'
    )


def _embedded_png(png_path: Path | None) -> str:
    if png_path is None or not png_path.exists():
        return '<p style="color:var(--color-text-secondary);">(no curve image available)</p>'
    data = base64.b64encode(png_path.read_bytes()).decode("ascii")
    return (
        f'<img src="data:image/png;base64,{data}" alt="Kr vs frequency" '
        f'style="max-width:100%;height:auto;border-radius:var(--border-radius-md);'
        f'background:var(--color-background-secondary);padding:0.5rem;"/>'
    )


def _csv_block(csv_path: Path | None) -> str:
    if csv_path is None or not csv_path.exists():
        return ""
    with csv_path.open() as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return ""
    header, *data = rows
    th = "".join(
        f'<th style="padding:4px 8px;text-align:left;font-size:11px;'
        f'color:var(--color-text-secondary);">{html.escape(h)}</th>' for h in header
    )
    trs = []
    for row in data:
        tds = "".join(
            f'<td style="padding:3px 8px;font-family:var(--font-mono);font-size:12px;">'
            f'{html.escape(c)}</td>' for c in row
        )
        trs.append(
            f'<tr style="border-bottom:0.5px solid var(--color-border-tertiary);">{tds}</tr>'
        )
    return (
        '<div style="overflow-x:auto;">'
        '<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="border-bottom:2px solid var(--color-border-secondary);">{th}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table></div>'
    )


def write_rw_report(
    pairs: list[tuple[RegularResult, TestMeta]],
    out_dir: Path,
    method: str,
    csv_path: Path | None = None,
    png_path: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Write ``rw_report_<method>.html`` into ``out_dir`` and return its path.

    ``pairs`` is a list of ``(RegularResult, TestMeta)`` — each meta supplies
    the per-test ``t_gen_s`` plus the (shared) tank geometry. The first meta
    is used for the tank-layout SVG and geometry cards.
    """
    if not pairs:
        raise ValueError("write_rw_report: no results to report")
    meta = pairs[0][1]
    rows = [_row_for(r, m) for r, m in pairs]
    rows.sort(key=lambda r: r["f"])

    header = (
        f"<h1>Regular-wave reflection report — method: {html.escape(method)}</h1>"
    )
    if timestamp:
        header += (
            f'<p style="color:var(--color-text-secondary);margin-bottom:1rem;">'
            f'Generated {html.escape(timestamp)} — {len(rows)} test(s)</p>'
        )

    layout_block = (
        '<div style="background:var(--color-background-secondary);'
        'border-radius:var(--border-radius-lg);padding:1.25rem;margin-bottom:1.5rem;">'
        '<p style="font-size:13px;color:var(--color-text-secondary);margin:0 0 10px;">'
        f'Tank layout (depth {meta.water_depth_m:g} m)</p>'
        f'{_tank_svg(meta)}</div>'
    )

    body = "\n".join([
        header,
        layout_block,
        _geometry_cards(meta, rows),
        "<h2>Per-test time window & reflection</h2>",
        _table(rows),
        "<h2>Time-window breakdown</h2>",
        _gantt_png(rows),
        "<h2>K<sub>r</sub> vs frequency</h2>",
        _embedded_png(png_path),
        "<h2>Result CSV</h2>",
        _csv_block(csv_path),
    ])

    html_doc = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f'<title>RW reflection report ({html.escape(method)})</title>'
        f'<style>{_CSS}</style></head><body>{body}</body></html>'
    )
    out_path = out_dir / f"rw_report_{method}.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
