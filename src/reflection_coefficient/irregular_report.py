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

from .analysis import solve_dispersion, solve_dispersion_array
from .io import TestMeta
from .pipeline import IrregularResult

# Re-use the stylesheet + tank SVG + geometry cards + chart bootstrap from rw_report.
from .rw_report import (  # noqa: F401
    CHARTJS_CDN,
    _CHART_DEFAULTS,
    _CSS,
    _geometry_cards,
    _tank_svg,
)


def _window_info(result: IrregularResult, meta: TestMeta) -> dict:
    """Read the time-window and band metadata straight from the pipeline's
    diagnostics so the report cannot drift from what the analysis used."""
    d = result.diagnostics
    depth = meta.water_depth_m
    g = meta.gravity_m_s2
    f_peak = float(d["f_peak_used_Hz"])
    f_lo = meta.f_min_Hz if meta.f_min_Hz else 0.5 * f_peak
    f_hi = meta.f_max_Hz if meta.f_max_Hz else 2.5 * f_peak
    _, L_peak = solve_dispersion(f_peak, depth, g=g)
    return {
        "f_peak": f_peak, "f_lo": float(f_lo), "f_hi": float(f_hi),
        "t_start": float(d["t_start_s"]),
        "t_end": float(d["t_end_s"]),
        "t_ana_start": float(d.get("t_analysis_start_s", d["t_start_s"])),
        "t_ana_end": float(d.get("t_analysis_end_s", d["t_end_s"])),
        "head_drop": float(d.get("head_drop_s", 0.0)),
        "tail_drop": float(d.get("tail_drop_s", 0.0)),
        "t_gen": meta.t_gen_s,
        "record_tail": float(d["record_tail_s"]),
        "runtime_bound": float(d["runtime_bound_s"]),
        "runtime_capped": bool(d["runtime_capped"]),
        "cg_fastest": float(d["cg_fastest_m_s"]),
        "cg_slowest": float(d["cg_slowest_m_s"]),
        "L_peak": float(L_peak),
    }


def _window_timeline_canvas(win: dict) -> str:
    """Single-row Gantt timeline for this test's clip window.

    Shows [0 → t_start] (wait for first reflection) and
    [t_start → t_end] (usable segment), plus dashed vertical markers at
    ``t_gen_s`` (paddle stop) and the record tail.
    """
    t_ana = max(win["t_ana_end"] - win["t_ana_start"], 0.0)
    T_p = 1.0 / win["f_peak"] if win["f_peak"] > 0 else float("nan")
    n_per = t_ana / T_p if T_p > 0 else 0.0
    status = "CLAMPED" if win["runtime_capped"] else "OK"
    ana_color = "rgba(186,117,23,0.6)" if win["runtime_capped"] else "rgba(50,102,173,0.6)"
    ana_edge = "#BA7517" if win["runtime_capped"] else "#3266ad"

    payload = json.dumps({
        "t_start": win["t_start"], "t_end": win["t_end"],
        "t_ana_start": win["t_ana_start"], "t_ana_end": win["t_ana_end"],
        "head_drop": win["head_drop"], "tail_drop": win["tail_drop"],
        "t_gen": win["t_gen"], "tail": win["record_tail"],
        "t_ana": t_ana, "n_per": n_per, "status": status,
        "ana_color": ana_color, "ana_edge": ana_edge,
    })
    script = f"""(function(){{
  const d = {payload};
  const markers = [];
  if (d.t_gen != null) markers.push({{x:d.t_gen, color:'#E24B4A', label:'t_gen = '+d.t_gen+' s'}});
  markers.push({{x:d.tail, color:'#534AB7', label:'record tail = '+d.tail.toFixed(1)+' s'}});
  const maxX = Math.max(d.t_end, d.tail, d.t_gen||0) * 1.05;
  new Chart(document.getElementById('winChart'), {{
    type:'bar',
    data:{{labels:[''], datasets:[
      {{label:'Wait', data:[[0, d.t_start]],
        backgroundColor:'rgba(136,135,128,0.2)', borderColor:'rgba(136,135,128,0.4)',
        borderWidth:1, borderSkipped:false}},
      {{label:'Head drop',
        data:[d.head_drop > 0 ? [d.t_start, d.t_ana_start] : null],
        backgroundColor:'rgba(186,117,23,0.25)', borderColor:'rgba(186,117,23,0.6)',
        borderWidth:1, borderSkipped:false}},
      {{label:'Analysis', data:[[d.t_ana_start, d.t_ana_end]],
        backgroundColor:d.ana_color, borderColor:d.ana_edge,
        borderWidth:1, borderSkipped:false}},
      {{label:'Tail drop',
        data:[d.tail_drop > 0 ? [d.t_ana_end, d.t_end] : null],
        backgroundColor:'rgba(186,117,23,0.25)', borderColor:'rgba(186,117,23,0.6)',
        borderWidth:1, borderSkipped:false}}
    ]}},
    options:{{
      responsive:true, maintainAspectRatio:false, indexAxis:'y',
      plugins:{{
        legend:{{display:false}},
        tooltip:{{callbacks:{{
          label:(ctx)=>{{
            if(!ctx.raw) return ctx.dataset.label+': —';
            const [a,b] = ctx.raw;
            const name = ctx.dataset.label;
            if(name==='Analysis')
              return ['Analysis: '+a.toFixed(1)+' → '+b.toFixed(1)+' s',
                      'T_use = '+d.t_ana.toFixed(1)+' s  ('+d.n_per.toFixed(1)+' T_p)',
                      'Status: '+d.status];
            if(name==='Head drop')
              return 'Head drop: '+a.toFixed(1)+' → '+b.toFixed(1)+' s  ('+d.head_drop.toFixed(1)+' s)';
            if(name==='Tail drop')
              return 'Tail drop: '+a.toFixed(1)+' → '+b.toFixed(1)+' s  ('+d.tail_drop.toFixed(1)+' s)';
            return 'Wait: 0 → '+b.toFixed(1)+' s';
          }}
        }}}}
      }},
      scales:{{
        x:{{title:{{display:true,text:'Time (s)'}}, min:0, max:maxX}},
        y:{{display:false}}
      }}
    }},
    plugins:[{{
      id:'tMarkers',
      afterDraw(chart){{
        const ctx = chart.ctx, xS = chart.scales.x, area = chart.chartArea;
        markers.forEach(m => {{
          const x = xS.getPixelForValue(m.x);
          ctx.save();
          ctx.strokeStyle = m.color; ctx.lineWidth = 1.5; ctx.setLineDash([5,3]);
          ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.bottom); ctx.stroke();
          ctx.restore();
          ctx.fillStyle = m.color;
          ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
          ctx.fillText(m.label, x+4, area.top+10);
        }});
      }}
    }}]
  }});
}})();"""
    legend = (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;margin:8px 0 0;'
        'font-size:12px;color:var(--color-text-secondary);">'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(136,135,128,0.3);margin-right:4px;"></span>Wait for reflections</span>'
        '<span><span style="display:inline-block;width:10px;height:10px;'
        'background:rgba(186,117,23,0.25);margin-right:4px;"></span>Head / tail drop</span>'
        f'<span><span style="display:inline-block;width:10px;height:10px;'
        f'background:{ana_color};margin-right:4px;"></span>Analysis ({status.lower()})</span>'
        '<span><span style="display:inline-block;width:2px;height:10px;'
        'background:#E24B4A;margin-right:4px;"></span>t_gen</span>'
        '<span><span style="display:inline-block;width:2px;height:10px;'
        'background:#534AB7;margin-right:4px;"></span>record tail</span>'
        '</div>'
    )
    return (
        '<div style="position:relative;width:100%;height:140px;'
        'background:var(--color-background-secondary);'
        'border-radius:var(--border-radius-md);padding:0.75rem;">'
        '<canvas id="winChart"></canvas></div>'
        f'<script>{script}</script>{legend}'
    )


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
        ("Head / tail drop",
         f"{win['head_drop']:g} / {win['tail_drop']:g} s"),
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
    window_mode: str = "canonical",
) -> Path:
    """Write the single-test irregular-wave report and supporting files.

    Returns the path to the generated HTML file.
    """
    win = _window_info(result, meta)
    suffix = "" if window_mode == "canonical" else f"_{window_mode}"
    stem = f"{result.test_id}_{method}{suffix}"

    csv_path = out_dir / f"{stem}_spectrum.csv"
    _write_spectrum_csv(result, csv_path)

    mode_tag = "" if window_mode == "canonical" else f", {window_mode} window"
    header = (
        f"<h1>Irregular-wave reflection report — {html.escape(result.test_id)} "
        f"({html.escape(meta.campaign)}, method: {html.escape(method)}"
        f"{html.escape(mode_tag)})</h1>"
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
        "<h2>Time-window breakdown</h2>",
        _window_timeline_canvas(win),
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
