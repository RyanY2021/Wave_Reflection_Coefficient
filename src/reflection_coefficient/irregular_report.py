"""Self-contained HTML report for a single irregular-wave test (wn / js).

Unlike the regular-wave report (which aggregates a family of single-frequency
tests), each irregular test produces its own spectra and ``Kr(f)`` curve, so
the HTML is keyed to a single ``IrregularResult`` + its ``TestMeta``.

Outputs written to ``out_dir``:

* ``<test_id>_<method>_spectrum.csv`` — smoothed S_I, S_R, Kr(f) table
* ``<test_id>_<method>_report.html`` — self-contained report with
  interactive Chart.js plots (spectra, Kr(f), singularity metric).
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import numpy as np

from .analysis import group_velocity, solve_dispersion, solve_dispersion_array
from .io import TestMeta
from .pipeline import IrregularResult, clip_bounds

# Re-use the stylesheet + tank SVG + geometry cards + chart bootstrap from rw_report.
from .rw_report import (  # noqa: F401
    CHARTJS_CDN,
    _CHART_DEFAULTS,
    _CSS,
    _geometry_cards,
    _tank_svg,
)


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


def _chart_wrap(canvas_id: str, height_px: int, inner_script: str) -> str:
    return (
        f'<div style="position:relative;width:100%;height:{height_px}px;'
        f'background:var(--color-background-secondary);'
        f'border-radius:var(--border-radius-md);padding:0.75rem;">'
        f'<canvas id="{canvas_id}"></canvas></div>'
        f'<script>{inner_script}</script>'
    )


def _to_list(arr) -> list:
    """NaN → None for JSON."""
    out = []
    for v in np.asarray(arr, dtype=float).tolist():
        out.append(None if (v != v) else v)  # NaN check
    return out


def _spectra_canvas(result: IrregularResult, win: dict) -> str:
    f = _to_list(result.f_smooth)
    sI = _to_list(result.S_I_smooth)
    sR = _to_list(result.S_R_smooth)
    pad = 0.1 * (win["f_hi"] - win["f_lo"])
    x_min = max(0.0, win["f_lo"] - pad)
    x_max = win["f_hi"] + pad
    script = f"""(function(){{
  const f = {json.dumps(f)}, sI = {json.dumps(sI)}, sR = {json.dumps(sR)};
  const ptsI = f.map((x,i)=>({{x, y:sI[i]}})).filter(p=>p.y!=null);
  const ptsR = f.map((x,i)=>({{x, y:sR[i]}})).filter(p=>p.y!=null);
  new Chart(document.getElementById('spectraChart'), {{
    type:'line',
    data:{{datasets:[
      {{label:'Incident S_I', data:ptsI, borderColor:'#3266ad',
        backgroundColor:'#3266ad', pointRadius:0, tension:0.1, borderWidth:1.5}},
      {{label:'Reflected S_R', data:ptsR, borderColor:'#D85A30',
        backgroundColor:'#D85A30', pointRadius:0, tension:0.1, borderWidth:1.5}}
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:false,
      interaction:{{mode:'nearest', axis:'x', intersect:false}},
      plugins:{{
        legend:{{labels:{{boxWidth:12}}}},
        tooltip:{{callbacks:{{
          title:(items)=>'f = '+items[0].parsed.x.toFixed(3)+' Hz',
          label:(ctx)=>ctx.dataset.label+': '+ctx.parsed.y.toExponential(3)+' m²/Hz'
        }}}}
      }},
      scales:{{
        x:{{type:'linear', min:{x_min}, max:{x_max}, title:{{display:true,text:'f (Hz)'}}}},
        y:{{title:{{display:true,text:'S(f) [m²/Hz]'}}, beginAtZero:true}}
      }}
    }},
    plugins:[bandPlugin({win['f_lo']}, {win['f_hi']}, {win['f_peak']})]
  }});
}})();"""
    return _chart_wrap("spectraChart", 380, script)


def _kr_f_canvas(result: IrregularResult, win: dict) -> str:
    f = _to_list(result.f_smooth)
    kr = _to_list(result.Kr_f)
    pad = 0.1 * (win["f_hi"] - win["f_lo"])
    x_min = max(0.0, win["f_lo"] - pad)
    x_max = win["f_hi"] + pad
    script = f"""(function(){{
  const f = {json.dumps(f)}, kr = {json.dumps(kr)};
  const pts = f.map((x,i)=>({{x, y:kr[i]}})).filter(p=>p.y!=null);
  const overall = {result.Kr_overall if np.isfinite(result.Kr_overall) else 'null'};
  new Chart(document.getElementById('krfChart'), {{
    type:'line',
    data:{{datasets:[
      {{label:'K_r(f)', data:pts, borderColor:'#0F6E56',
        backgroundColor:'#0F6E56', pointRadius:2, pointHoverRadius:5,
        tension:0.15, borderWidth:1.5}}
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{boxWidth:12}}}},
        tooltip:{{callbacks:{{
          title:(items)=>'f = '+items[0].parsed.x.toFixed(3)+' Hz',
          label:(ctx)=>'K_r = '+ctx.parsed.y.toFixed(3)
        }}}}
      }},
      scales:{{
        x:{{type:'linear', min:{x_min}, max:{x_max}, title:{{display:true,text:'f (Hz)'}}}},
        y:{{title:{{display:true,text:'K_r(f)'}}, beginAtZero:true}}
      }}
    }},
    plugins:[bandPlugin({win['f_lo']}, {win['f_hi']}, {win['f_peak']}),
            hLinePlugin(overall, '#A32D2D', 'K_r overall = '+(overall==null?'—':overall.toFixed(3)))]
  }});
}})();"""
    return _chart_wrap("krfChart", 380, script)


def _singularity_canvas(result: IrregularResult, meta: TestMeta,
                        method: str, win: dict) -> str:
    f = np.asarray(result.f)
    k = solve_dispersion_array(f, meta.water_depth_m, g=meta.gravity_m_s2)
    if method == "goda":
        metric = np.sin(k * meta.X13_m) ** 2
        threshold = 0.05
        y_label = "sin²(k·X13)  (Goda)"
    else:
        sb = np.sin(k * meta.X12_m)
        sg = np.sin(k * meta.X13_m)
        sgb = np.sin(k * meta.X13_m - k * meta.X12_m)
        metric = 2.0 * (sb * sb + sg * sg + sgb * sgb)
        threshold = 0.1
        y_label = "D  (Mansard–Funke)"
    pad = 0.1 * (win["f_hi"] - win["f_lo"])
    x_min = max(0.0, win["f_lo"] - pad)
    x_max = win["f_hi"] + pad
    f_list = _to_list(f)
    m_list = _to_list(metric)
    script = f"""(function(){{
  const f = {json.dumps(f_list)}, m = {json.dumps(m_list)};
  const pts = f.map((x,i)=>({{x, y:m[i]}})).filter(p=>p.y!=null);
  new Chart(document.getElementById('singChart'), {{
    type:'line',
    data:{{datasets:[
      {{label:{json.dumps(y_label)}, data:pts, borderColor:'#5f5e5a',
        backgroundColor:'#5f5e5a', pointRadius:0, borderWidth:1.2, tension:0.1}}
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{boxWidth:12}}}},
        tooltip:{{callbacks:{{
          title:(items)=>'f = '+items[0].parsed.x.toFixed(3)+' Hz',
          label:(ctx)=>ctx.dataset.label+' = '+ctx.parsed.y.toFixed(4)
        }}}}
      }},
      scales:{{
        x:{{type:'linear', min:{x_min}, max:{x_max}, title:{{display:true,text:'f (Hz)'}}}},
        y:{{title:{{display:true,text:{json.dumps(y_label)}}}, beginAtZero:true}}
      }}
    }},
    plugins:[bandPlugin({win['f_lo']}, {win['f_hi']}, {win['f_peak']}),
            hLinePlugin({threshold}, '#A32D2D', 'threshold = {threshold}')]
  }});
}})();"""
    return _chart_wrap("singChart", 320, script)


_CHART_PLUGINS = r"""
function bandPlugin(fLo, fHi, fPeak){
  return {
    id:'band'+fLo+'_'+fHi,
    beforeDatasetsDraw(chart){
      const ctx = chart.ctx, xS = chart.scales.x, area = chart.chartArea;
      const xL = xS.getPixelForValue(fLo), xR = xS.getPixelForValue(fHi);
      ctx.save();
      ctx.fillStyle = 'rgba(50,102,173,0.07)';
      ctx.fillRect(xL, area.top, xR-xL, area.bottom-area.top);
      ctx.restore();
    },
    afterDraw(chart){
      const ctx = chart.ctx, xS = chart.scales.x, area = chart.chartArea;
      const xP = xS.getPixelForValue(fPeak);
      if(xP < area.left || xP > area.right) return;
      ctx.save();
      ctx.strokeStyle = '#534AB7'; ctx.lineWidth = 1; ctx.setLineDash([5,3]);
      ctx.beginPath(); ctx.moveTo(xP, area.top); ctx.lineTo(xP, area.bottom); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = '#534AB7';
      ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
      ctx.fillText('f_peak = '+fPeak.toFixed(2)+' Hz', xP+4, area.top+10);
    }
  };
}
function hLinePlugin(y, color, label){
  return {
    id:'hline'+y+color,
    afterDraw(chart){
      if(y==null) return;
      const ctx = chart.ctx, yS = chart.scales.y, area = chart.chartArea;
      const yp = yS.getPixelForValue(y);
      if(yp < area.top || yp > area.bottom) return;
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.setLineDash([2,3]);
      ctx.beginPath(); ctx.moveTo(area.left, yp); ctx.lineTo(area.right, yp); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = color;
      ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
      ctx.fillText(label, area.left+6, yp-4);
    }
  };
}
"""


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

    csv_path = out_dir / f"{stem}_spectrum.csv"
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
    geo_rows = [{"t_gen": meta.t_gen_s}]

    body = "\n".join([
        header,
        layout_block,
        _geometry_cards(meta, geo_rows),
        "<h2>Summary</h2>",
        _summary_cards(result, win),
        "<h2>Incident & reflected spectra</h2>",
        _spectra_canvas(result, win),
        "<h2>K<sub>r</sub>(f)</h2>",
        _kr_f_canvas(result, win),
        "<h2>Singularity metric</h2>",
        _singularity_canvas(result, meta, method, win),
    ])

    html_doc = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0">'
        f'<title>{html.escape(result.test_id)} reflection report</title>'
        f'<style>{_CSS}</style>'
        f'<script src="{CHARTJS_CDN}"></script>'
        f'</head><body>'
        f'<script>{_CHART_DEFAULTS}</script>'
        f'<script>{_CHART_PLUGINS}</script>'
        f'{body}</body></html>'
    )
    out_path = out_dir / f"{stem}_report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path
