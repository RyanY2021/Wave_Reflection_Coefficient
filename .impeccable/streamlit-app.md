# Design Brief — Reflection Coefficient Streamlit App

Produced by `/impeccable:shape`. Design planning only — no code written here.
Hand off to `/impeccable craft` or any implementation skill when ready.

See `.impeccable.md` for the project-wide Design Context this brief inherits.

## 1. Feature Summary

A local browser-based analysis companion on top of `src/reflection_coefficient/`.
Researchers point it at tank config + probe data, pick a test (or compare a
regular-wave sweep against an irregular-wave run), run the analysis, and see a
compact validity headline before deciding whether to preview the full HTML
report and download it. It reads as a continuation of the report itself — an
editorial lab notebook with interactive controls — not a SaaS dashboard.

## 2. Primary User Action

**Validate, then share.** The decisive moment is judging whether an
experiment's Kr is trustworthy — singularity clean, clip window sensible,
incident/reflected split reasonable. The headline card exists for
"should I even bother?"; the preview + download exists for
"this is worth keeping."

## 3. Design Direction

Inherits `.impeccable.md` exactly: warm cream / deep charcoal, serif display
with tabular-lining numerics, desaturated status pills, `prefers-color-scheme`
auto. Tonal reference: claude.ai's ivory restraint. A single warm
terracotta-ish accent, used only on the primary Run action and the headline
Kr value.

Three words: **measured, warm, exact.**

## 4. Layout Strategy

Three zones, stacked vertically with generous rhythm. Not a wizard, not a
dashboard.

- **Setup zone** (top) — single unified surface. Asymmetric two-column:
  paths on the left (rarely change), analysis knobs on the right (change
  often). Collapses to a one-line summary strip after a successful run so
  the preview can dominate.
- **Result headline zone** (middle) — compact card on run completion: large
  serif Kr value, singularity pill, miniature clip-window bar, download
  actions. The decide-or-discard moment.
- **Preview zone** (bottom) — full HTML report embedded, scrolling inside
  its own pane, theme matched to app chrome.

**Left sidebar** holds the **run history** (pulled from `<project>/log/`,
bounded to the newest 10), chronologically, with parameter fingerprint +
headline Kr per entry. Click a history entry to re-open its scalar values
in the headline zone without a re-run.

**Comparison mode**: a segmented toggle in the setup zone ("Single | Compare").
In Compare, the setup surfaces two slots side by side:

- **Reference sweep** — scheme fixed to `rw`, test fixed to `--test all`,
  pulling every regular-wave discrete frequency in the current data dir.
- **Irregular run** — scheme `wn` (or `js`), single test picker.

Output is **one unified Kr vs. frequency_Hz chart** in the headline zone —
rw as discrete markers (tinted by singularity flag), the irregular test as a
continuous band-averaged curve over the same axes. Below the chart, a compact
two-row summary table: rw (test count, Kr range, singularity rate) and
wn/js (Hm0_I, Hm0_R, Kr_overall, Tp). No embedded HTML report preview in
Compare — the overlay is the deliverable.

Break the grid intentionally: the headline Kr number in Single mode is set
large, left-aligned, with unusual leading space — the visual anchor of a
valid run.

## 5. Key States

- **Empty / first open** — setup full height; inline captions explain
  non-obvious controls; headline hint: "Pick a test and run to see the
  headline."
- **Paths resolved, no run** — controls ready; headline empty.
- **Running** — controls dim slightly; headline shows a skeleton card
  (Kr: —, singularity: —, clip: —), not a generic spinner; log strip
  accumulates live lines.
- **Success** — headline renders; preview loads; setup collapses to a
  one-line summary strip
  (`RW001 · least_squares · hann 0.04 Hz · head/tail 3 s — edit`).
- **Success but flagged** — singularity pill set to warning; one-line
  plain-language caption explains the flag.
- **Failure** — inline error under the relevant control (never a modal).
  Human explanation first; raw exception behind a "details" disclosure.
- **Comparison** — two slim parameter slots in setup (rw all-sweep +
  wn/js single test); headline zone becomes the overlay Kr(f) chart
  (rw markers + irregular curve) with the two-row summary table beneath.
  No preview pane.
- **History entry selected** — headline shows historical scalars; quiet
  label "from log — preview not cached"; secondary "Re-run to preview"
  action.

## 6. Interaction Model

- Single warm-accent **Run** button; no other element competes for that
  color.
- Parameter changes update live — no submit step; Run re-enables.
- Download HTML / CSV save via browser; disk remains untouched until
  clicked.
- Log strip: collapsed band above setup on entry, expandable for live
  output; auto-collapses to one line ("Logged to …") after run.
- History click → scalars in headline; secondary "Re-run" rebuilds the
  report.
- Compare toggle reshapes setup into two fixed-purpose slots: **rw
  all-sweep** (reference) + **irregular single test**. Output is one
  overlaid Kr(f) chart — markers for rw, line for the irregular — not
  separate columns.
- Critical interactions are click/focus, not hover — respects projector /
  shared-screen sessions.

## 7. Content Requirements

Copy is measured, not marketing. Short, declarative, quiet.

- **Headings** (serif, sentence case, never all-caps): "Setup", "Result",
  "Report", "History".
- **Control labels** (one or two words, sentence case): "Tank config",
  "Metadata", "Data", "Scheme", "Test", "Method", "Window", "Bandwidth",
  "Head drop", "Tail drop".
- **Inline captions** under non-obvious controls:
  - Bandwidth — "Resolution of the irregular-wave band-average. Smaller =
    noisier Kr(f), more detail."
  - Head / tail drop — "Seconds trimmed from the travel-time window to skip
    ramp transients."
  - Method — "Goda uses two probes; Mansard–Funke uses three and is more
    robust near singularities."
- **Empty-state** (no run yet): "Pick a test and run to see the headline."
- **Flagged result**: "Singularity near this frequency — the probe-spacing
  trigonometry is noisy here, treat Kr as approximate."
- **Error copy** — explains *what*, then *where to look*:
  "Couldn't load *RW005* — file at `…/RW005.txt` is missing. Check
  **Data dir**."
- **History entry**: timestamp (relative for today, absolute otherwise),
  test id, Kr or Kr_overall to 3 decimals, one-character singularity glyph.
- **Log strip collapsed**: "Logged to `log/20260419_143022.log` · 7 runs kept."

Dynamic content ranges to design for:
- History: up to ~10 entries (matches `_LOG_KEEP` in `scripts/streamlit_app.py`).
- Test selection: up to 30 tests for rw (chips, not a scroll); single-select
  for irregular.
- Headline Kr: typical 0.05–1.2; present with 3 decimals.
- Preview HTML: 300–900 KB self-contained with embedded Chart.js.

## 8. Recommended References

During implementation, consult:

- `reference/typography.md` — carry the serif-display + mono-numerics
  pairing from the report into Streamlit.
- `reference/color-and-contrast.md` — port the palette to OKLCH; tint
  neutrals toward the accent hue.
- `reference/spatial-design.md` — the three-zone rhythm and the
  setup-collapses-after-run move.
- `reference/interaction-design.md` — inline-error pattern and segmented
  Single/Compare control.
- `reference/motion-design.md` — setup-zone collapse and skeleton→headline
  transitions; ease-out-quart, 150–250 ms, nothing bouncy.

## 9. Open Questions

Decisions to make during implementation:

1. **Streamlit vs. a custom Flask/FastAPI front-end.** Streamlit's style
   primitives are constrained; heavy `st.markdown` + CSS injection may be
   needed. If the editorial target isn't reachable in Streamlit, a
   lightweight Flask/FastAPI + HTMX app is a real alternative.
2. **Structured log sidecar.** Current logs are printed lines; a per-run
   JSON sidecar would make the history browser much more robust.
3. **"Up to date" run-button state.** Skippable for MVP; adds chrome, saves
   little.
4. **Exact accent OKLCH.** A warm terracotta passing AA against both cream
   and charcoal surfaces; tint neutrals toward its hue by chroma 0.005–0.01.

*(Resolved during shaping: cross-scheme comparison semantics — overlay rw
markers + irregular curve on one Kr-vs-f chart.)*
