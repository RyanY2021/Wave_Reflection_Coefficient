"""Self-contained HTML report for a regular-wave (``--scheme rw``) run.

Renders the tank layout, geometry summary, per-test time-window table (f, T,
L, c, cg, t_gen, t_start, t_end, T_usable, N_per, Kr, singularity flag) and
embeds the Kr-vs-frequency curve PNG produced by ``run_analysis.py``.

Inputs are deliberately minimal: a list of :class:`RegularResult` from
:mod:`reflection_coefficient.pipeline`, one representative :class:`TestMeta`
for tank geometry, and optional paths to the CSV / PNG generated alongside.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path

from .io import TestMeta
from .pipeline import RegularResult


def _row_for(result: RegularResult, meta: TestMeta) -> dict:
    """Build a report row from a pipeline result.

    All time-window quantities come from the pipeline (``analyse_regular``)
    to guarantee the HTML matches the window the Kr was computed over.
    """
    f = result.f_Hz
    T = 1.0 / f
    omega = 2.0 * math.pi * f
    c = omega / result.k
    t_ana = max(result.t_analysis_end_s - result.t_analysis_start_s, 0.0)
    n_per = t_ana / T if t_ana > 0 else 0.0
    status = "CLAMPED" if result.runtime_capped else "OK"
    if 0 < n_per < 5:
        status += "_FEW"
    if not result.singularity_ok:
        status += "+SING"
    return {
        "test_id": result.test_id,
        "f": f, "T": T, "L": result.wavelength_m,
        "c": c, "cg": result.cg_m_s,
        "t_gen": meta.t_gen_s,
        "t_start": result.t_start_s,
        "t_end": result.t_end_s,
        "t_ana_start": result.t_analysis_start_s,
        "t_ana_end": result.t_analysis_end_s,
        "head_drop": result.head_drop_s,
        "tail_drop": result.tail_drop_s,
        "t_use": t_ana, "n_per": n_per,
        "H_I": result.H_I, "H_R": result.H_R, "Kr": result.Kr,
        "status": status, "singularity_ok": result.singularity_ok,
    }


def _status_colors(status: str) -> tuple[str, str]:
    if status.startswith("OK") and "_FEW" not in status and "+SING" not in status:
        return "var(--color-background-success)", "#0F6E56"
    if status.startswith("CLAMPED") and "_FEW" not in status and "+SING" not in status:
        return "var(--color-background-warning)", "#BA7517"
    return "var(--color-background-danger)", "#A32D2D"


CHARTJS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"

_CHART_DEFAULTS = (
    "if(window.Chart){"
    "Chart.defaults.font.family=getComputedStyle(document.body).fontFamily;"
    "Chart.defaults.color=getComputedStyle(document.body).color;"
    "Chart.defaults.borderColor=getComputedStyle(document.body).getPropertyValue('--color-border-tertiary');"
    "}"
)

_CSS = """
@font-face {
  font-family:'ReportDigits';
  src:local('Times New Roman'),local('TimesNewRomanPSMT'),local('Times');
  unicode-range:U+0030-0039;
}
:root {
  --color-background-primary:#ffffff;--color-background-secondary:#f5f5f2;
  --color-background-success:#EAF3DE;--color-background-warning:#FAEEDA;
  --color-background-danger:#FCEBEB;--color-text-primary:#1a1a1a;
  --color-text-secondary:#5f5e5a;--color-border-tertiary:rgba(0,0,0,0.15);
  --color-border-secondary:rgba(0,0,0,0.3);
  --font-sans:'ReportDigits','Galaxie Copernicus','Copernicus',Georgia,'Times New Roman',serif;
  --font-mono:'ReportDigits','SF Mono','Consolas','Courier New',monospace;
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
        "head drop (s)", "tail drop (s)",
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
            f"{r['head_drop']:.1f}", f"{r['tail_drop']:.1f}",
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


def _gantt_canvas(rows: list[dict]) -> str:
    """Interactive Gantt chart of per-test time windows (Chart.js)."""
    if not rows:
        return ""
    data = [
        {
            "f": r["f"], "t_gen": r["t_gen"],
            "t_start": r["t_start"], "t_end": r["t_end"],
            "t_ana_start": r["t_ana_start"], "t_ana_end": r["t_ana_end"],
            "head_drop": r["head_drop"], "tail_drop": r["tail_drop"],
            "t_use": r["t_use"], "n_per": r["n_per"],
            "status": r["status"],
        }
        for r in rows
    ]
    payload = json.dumps(data)
    height = max(320, 22 * len(rows) + 80)
    legend = (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;margin:8px 0 0;'
        'font-size:12px;color:var(--color-text-secondary);">'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(136,135,128,0.3);margin-right:4px;"></span>Wait for reflections</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(186,117,23,0.25);margin-right:4px;"></span>Head / tail drop</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(50,102,173,0.6);margin-right:4px;"></span>Analysis window (OK)</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(186,117,23,0.6);margin-right:4px;"></span>Analysis window (clamped)</span>'
        '<span><span style="display:inline-block;width:2px;height:10px;'
        'background:#E24B4A;margin-right:4px;"></span>180 s limit</span>'
        '<span><span style="display:inline-block;width:2px;height:10px;'
        'background:#534AB7;margin-right:4px;"></span>240 s limit</span>'
        '</div>'
    )
    return f"""
<div style="position:relative;width:100%;height:{height}px;
     background:var(--color-background-secondary);
     border-radius:var(--border-radius-md);padding:0.75rem;">
  <canvas id="ganttChart"></canvas>
</div>{legend}
<script>(function(){{
  const rows = {payload};
  const labels = rows.map(r => r.f.toFixed(2));
  const waitData = rows.map(r => r.t_use > 0 ? [0, r.t_start] : null);
  const headData = rows.map(r => r.t_use > 0 && r.head_drop > 0 ? [r.t_start, r.t_ana_start] : null);
  const anaData  = rows.map(r => r.t_use > 0 ? [r.t_ana_start, r.t_ana_end] : null);
  const tailData = rows.map(r => r.t_use > 0 && r.tail_drop > 0 ? [r.t_ana_end, r.t_end] : null);
  const anaFills = rows.map(r => r.status.startsWith('OK') ? 'rgba(50,102,173,0.6)' : 'rgba(186,117,23,0.6)');
  const anaEdges = rows.map(r => r.status.startsWith('OK') ? '#3266ad' : '#BA7517');
  const tGens = Array.from(new Set(rows.map(r => r.t_gen).filter(v => v!=null))).sort((a,b)=>a-b);
  const limitColors = {{180:'#E24B4A', 240:'#534AB7'}};
  const maxX = Math.max(...rows.map(r=>r.t_end||0), ...tGens, 1) * 1.05;
  new Chart(document.getElementById('ganttChart'), {{
    type:'bar',
    data:{{labels, datasets:[
      {{label:'Wait', data:waitData,
        backgroundColor:'rgba(136,135,128,0.2)', borderColor:'rgba(136,135,128,0.4)',
        borderWidth:1, borderSkipped:false}},
      {{label:'Head drop', data:headData,
        backgroundColor:'rgba(186,117,23,0.25)', borderColor:'rgba(186,117,23,0.6)',
        borderWidth:1, borderSkipped:false}},
      {{label:'Analysis', data:anaData, backgroundColor:anaFills, borderColor:anaEdges,
        borderWidth:1, borderSkipped:false}},
      {{label:'Tail drop', data:tailData,
        backgroundColor:'rgba(186,117,23,0.25)', borderColor:'rgba(186,117,23,0.6)',
        borderWidth:1, borderSkipped:false}}
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:false, indexAxis:'y',
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{
          title:(items)=>'f = '+items[0].label+' Hz',
          label:(ctx)=>{{
            const r = rows[ctx.dataIndex];
            if(!ctx.raw) return ctx.dataset.label+': —';
            const [a,b] = ctx.raw;
            const name = ctx.dataset.label;
            if(name==='Analysis')
              return ['Analysis: '+a.toFixed(1)+' → '+b.toFixed(1)+' s',
                      'T_use = '+r.t_use.toFixed(1)+' s  ('+r.n_per.toFixed(1)+' periods)',
                      'Status: '+r.status];
            if(name==='Head drop')
              return 'Head drop: '+a.toFixed(1)+' → '+b.toFixed(1)+' s  ('+r.head_drop.toFixed(1)+' s)';
            if(name==='Tail drop')
              return 'Tail drop: '+a.toFixed(1)+' → '+b.toFixed(1)+' s  ('+r.tail_drop.toFixed(1)+' s)';
            return 'Wait: 0 → '+b.toFixed(1)+' s';
          }}
        }}}}
      }},
      scales:{{
        x:{{title:{{display:true,text:'Time (s)'}}, min:0, max:maxX}},
        y:{{title:{{display:true,text:'Frequency (Hz)'}}, ticks:{{autoSkip:false, font:{{size:11}}}}}}
      }}
    }},
    plugins:[{{
      id:'tGenLimits',
      afterDraw(chart){{
        const ctx = chart.ctx, xS = chart.scales.x, area = chart.chartArea;
        tGens.forEach(v => {{
          const x = xS.getPixelForValue(v);
          ctx.save();
          ctx.strokeStyle = limitColors[v] || '#888';
          ctx.lineWidth = 1.5; ctx.setLineDash([5,3]);
          ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.bottom); ctx.stroke();
          ctx.restore();
          ctx.fillStyle = limitColors[v] || '#888';
          ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
          ctx.fillText(v+' s', x+4, area.top+10);
        }});
      }}
    }}]
  }});
}})();</script>
"""


def _kr_chart_canvas(rows: list[dict]) -> str:
    """Interactive Kr-vs-f line chart."""
    data = [
        {"f": r["f"], "Kr": r["Kr"], "ok": bool(r["singularity_ok"]),
         "test_id": r["test_id"], "H_I": r["H_I"], "H_R": r["H_R"]}
        for r in rows
    ]
    payload = json.dumps(data)
    return f"""
<div style="position:relative;width:100%;height:420px;
     background:var(--color-background-secondary);
     border-radius:var(--border-radius-md);padding:0.75rem;">
  <canvas id="krChart"></canvas>
</div>
<script>(function(){{
  const rows = {payload};
  const points = rows.map(r => ({{x:r.f, y:r.Kr}}));
  const flagged = rows.filter(r => !r.ok).map(r => ({{x:r.f, y:r.Kr}}));
  new Chart(document.getElementById('krChart'), {{
    type:'line',
    data:{{datasets:[
      {{label:'K_r', data:points, borderColor:'#3266ad', backgroundColor:'#3266ad',
        pointRadius:4, pointHoverRadius:6, tension:0.1, fill:false}},
      {{label:'Singularity flagged', data:flagged, type:'scatter',
        borderColor:'#A32D2D', backgroundColor:'#A32D2D',
        pointStyle:'crossRot', pointRadius:8, pointHoverRadius:10, showLine:false}}
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{boxWidth:12}}}},
        tooltip:{{callbacks:{{
          label:(ctx)=>{{
            const r = rows[ctx.dataIndex];
            return [r.test_id+':  K_r = '+r.Kr.toFixed(3),
                    'H_I = '+r.H_I.toFixed(4)+' m,  H_R = '+r.H_R.toFixed(4)+' m',
                    r.ok ? '' : '⚠ singularity flagged'].filter(Boolean);
          }}
        }}}}
      }},
      scales:{{
        x:{{type:'linear', title:{{display:true,text:'f (Hz)'}}}},
        y:{{title:{{display:true,text:'K_r'}}, beginAtZero:true}}
      }}
    }}
  }});
}})();</script>
"""


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
        _gantt_canvas(rows),
        "<h2>K<sub>r</sub> vs frequency</h2>",
        _kr_chart_canvas(rows),
        "<h2>Result CSV</h2>",
        _csv_block(csv_path),
    ])

    html_doc = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f'<title>RW reflection report ({html.escape(method)})</title>'
        f'<style>{_CSS}</style>'
        f'<script src="{CHARTJS_CDN}"></script>'
        f'</head><body>'
        f'<script>{_CHART_DEFAULTS}</script>'
        f'{body}</body></html>'
    )
    out_path = out_dir / f"rw_report_{method}.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
