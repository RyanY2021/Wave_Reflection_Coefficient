"""Streamlit companion for reflection-coefficient analysis.

    pip install streamlit
    streamlit run scripts/streamlit_app.py

Visual design follows `.impeccable/streamlit-app.md` (feature brief) and
`.impeccable.md` (project design context): editorial lab notebook aesthetic,
warm cream / deep charcoal palette, serif display + mono numerics, auto
light/dark via prefers-color-scheme.

Reports live only in a temp dir that is removed before the Streamlit
callback returns; nothing persists to disk unless the user clicks Download.
A bounded log file per run is written to <project>/log/ (newest 10 kept).
"""

from __future__ import annotations

import csv
import html as _html
import shutil
import tempfile
import time
import tkinter as tk
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

import streamlit as st
import streamlit.components.v1 as components

from reflection_coefficient.io import (
    list_tests,
    load_probe_data,
    resolve_data_dir,
    resolve_metadata_dir,
    resolve_tank_config,
)
from reflection_coefficient.irregular_report import write_irregular_report
from reflection_coefficient.pipeline import IrregularResult, RegularResult, analyse
from reflection_coefficient.rw_report import write_rw_report

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOG_DIR = _PROJECT_ROOT / "log"
_LOG_KEEP = 10

_SCHEME_LABELS = {
    "rw": "Regular wave",
    "wn": "White-noise irregular",
    "js": "JONSWAP irregular",
}


# --- styling -----------------------------------------------------------------

_LIGHT_TOKENS = """
  --bg:         oklch(0.985 0.004 78);
  --surface:    oklch(0.965 0.006 80);
  --surface-2:  oklch(0.935 0.007 80);
  --ink:        oklch(0.26 0.008 70);
  --ink-muted:  oklch(0.50 0.008 70);
  --ink-faint:  oklch(0.66 0.006 72);
  --rule:       oklch(0.88 0.006 75);
  --rule-strong:oklch(0.78 0.008 75);
  --accent:     oklch(0.58 0.135 35);
  --accent-ink: oklch(0.98 0.008 80);
  --accent-soft:oklch(0.92 0.05 40);
  --ok-bg:   oklch(0.93 0.045 115);
  --ok-ink:  oklch(0.38 0.06 125);
  --warn-bg: oklch(0.93 0.07 70);
  --warn-ink:oklch(0.42 0.09 55);
  --err-bg:  oklch(0.92 0.06 25);
  --err-ink: oklch(0.42 0.10 25);
"""

_DARK_TOKENS = """
  --bg:        oklch(0.165 0.007 75);
  --surface:   oklch(0.215 0.008 75);
  --surface-2: oklch(0.255 0.009 75);
  --ink:       oklch(0.93 0.008 80);
  --ink-muted: oklch(0.72 0.008 80);
  --ink-faint: oklch(0.54 0.007 78);
  --rule:      oklch(0.32 0.008 75);
  --rule-strong: oklch(0.44 0.010 75);
  --accent:    oklch(0.74 0.125 36);
  --accent-ink:oklch(0.18 0.010 70);
  --accent-soft: oklch(0.32 0.055 40);
  --ok-bg:   oklch(0.28 0.05 125);
  --ok-ink:  oklch(0.86 0.06 125);
  --warn-bg: oklch(0.30 0.075 60);
  --warn-ink:oklch(0.88 0.08 65);
  --err-bg:  oklch(0.28 0.06 25);
  --err-ink: oklch(0.88 0.075 25);
"""

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Commit+Mono:wght@400;500;700&display=swap');

:root {
  --space-1: 4px;  --space-2: 8px;  --space-3: 12px; --space-4: 16px;
  --space-5: 24px; --space-6: 32px; --space-7: 48px; --space-8: 64px;

  --font-serif: 'Spectral', 'Galaxie Copernicus', 'Copernicus', Georgia, 'Times New Roman', serif;
  --font-mono:  'Commit Mono', ui-monospace, 'SF Mono', Consolas, 'Courier New', monospace;

  --text-2xs: 0.75rem;  --lh-2xs: 1.35;
  --text-xs:  0.8125rem; --lh-xs: 1.45;
  --text-sm:  0.9375rem; --lh-sm: 1.55;
  --text-md:  1.0625rem; --lh-md: 1.6;
  --text-lg:  1.375rem;  --lh-lg: 1.35;
  --text-xl:  2rem;       --lh-xl: 1.2;
  --text-display: 3.25rem; --lh-display: 1.05;

  /* LIGHT defaults */
  __LIGHT__
}

@media (prefers-color-scheme: dark) {
  :root { __DARK__ }
}

/* reset the Streamlit frame */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--ink);
  font-family: var(--font-serif);
  font-feature-settings: 'liga', 'kern', 'onum';
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { right: var(--space-4); }
#MainMenu, footer { visibility: hidden; }
.block-container {
  max-width: 1120px !important;
  padding-top: var(--space-6) !important;
  padding-bottom: var(--space-8) !important;
  padding-left: var(--space-6) !important;
  padding-right: var(--space-6) !important;
}

/* typography */
h1, h2, h3, h4 {
  font-family: var(--font-serif);
  color: var(--ink);
  letter-spacing: -0.01em;
  font-weight: 500;
}
p, li, label, span, div { color: var(--ink); }

/* the masthead */
.rc-masthead {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-5);
  padding: var(--space-2) 0 var(--space-4);
  margin-bottom: var(--space-6);
  border-bottom: 1px solid var(--rule);
}
.rc-masthead__title {
  font-family: var(--font-serif);
  font-weight: 500;
  font-size: var(--text-lg);
  line-height: var(--lh-lg);
  letter-spacing: -0.015em;
  margin: 0;
}
.rc-masthead__title em {
  font-style: italic;
  color: var(--ink-muted);
  font-weight: 400;
}
.rc-masthead__meta {
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.02em;
  color: var(--ink-faint);
  text-transform: uppercase;
}

/* sections */
.rc-section-label {
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
  padding-bottom: var(--space-2);
  margin: var(--space-5) 0 var(--space-3);
  border-bottom: 1px solid var(--rule);
}

/* caption under controls */
.rc-caption {
  font-family: var(--font-serif);
  font-style: italic;
  font-size: var(--text-xs);
  line-height: var(--lh-xs);
  color: var(--ink-muted);
  margin: calc(var(--space-1) * -1) 0 var(--space-3) 0;
  max-width: 62ch;
}

/* widget shells — buttons */
.stButton > button {
  font-family: var(--font-serif) !important;
  font-size: var(--text-sm) !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  padding: 10px 18px !important;
  border-radius: 2px !important;
  border: 1px solid var(--rule-strong) !important;
  background: transparent !important;
  color: var(--ink) !important;
  box-shadow: none !important;
  transition: background-color 180ms cubic-bezier(0.22,1,0.36,1),
              border-color 180ms cubic-bezier(0.22,1,0.36,1);
}
.stButton > button:hover {
  border-color: var(--ink) !important;
  background: var(--surface) !important;
}
.stButton > button:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: var(--accent-ink) !important;
  border-color: var(--accent) !important;
  font-weight: 500 !important;
  letter-spacing: 0.005em !important;
}
.stButton > button[kind="primary"]:hover {
  background: color-mix(in oklch, var(--accent), black 6%) !important;
  border-color: color-mix(in oklch, var(--accent), black 6%) !important;
}
.stButton > button:disabled {
  opacity: 0.45 !important;
  cursor: not-allowed !important;
}
.stDownloadButton > button {
  font-family: var(--font-serif) !important;
  font-size: var(--text-sm) !important;
  border-radius: 2px !important;
  border: 1px solid var(--rule-strong) !important;
  background: transparent !important;
  color: var(--ink) !important;
  padding: 8px 16px !important;
}
.stDownloadButton > button:hover {
  border-color: var(--ink) !important;
  background: var(--surface) !important;
}

/* inputs */
.stTextInput > div > div > input,
.stNumberInput input {
  font-family: var(--font-mono) !important;
  font-size: var(--text-sm) !important;
  background: var(--surface) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
  padding: 9px 12px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput input:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px color-mix(in oklch, var(--accent) 18%, transparent) !important;
}

/* select / multiselect */
[data-baseweb="select"] > div {
  background: var(--surface) !important;
  border-color: var(--rule) !important;
  border-radius: 2px !important;
  font-family: var(--font-serif) !important;
  font-size: var(--text-sm) !important;
}
[data-baseweb="select"] input { color: var(--ink) !important; }
[data-baseweb="tag"] {
  background: var(--surface-2) !important;
  color: var(--ink) !important;
  border-radius: 2px !important;
  font-family: var(--font-mono) !important;
  font-size: var(--text-2xs) !important;
}

/* dropdown popover (the panel that opens when a selectbox is clicked) —
   BaseWeb renders it in a portal outside the widget tree, so it has to be
   targeted globally. Without these rules it picks up Streamlit's default
   palette and reads white-on-white in light mode. */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] ul,
div[data-baseweb="popover"] > div > div {
  background: var(--surface) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
  box-shadow: 0 6px 18px -8px color-mix(in oklch, var(--ink) 22%, transparent) !important;
  font-family: var(--font-serif) !important;
  font-size: var(--text-sm) !important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li {
  background: var(--surface) !important;
  color: var(--ink) !important;
  font-family: var(--font-serif) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="menu"] li[aria-selected="true"] {
  background: var(--surface-2) !important;
  color: var(--ink) !important;
}

/* radio group — horizontal segmented */
[role="radiogroup"] {
  gap: var(--space-1) !important;
}
[role="radiogroup"] > label {
  padding: 6px 14px !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
  background: var(--surface) !important;
  cursor: pointer !important;
  transition: border-color 150ms, background-color 150ms;
}
[role="radiogroup"] > label:hover { border-color: var(--ink-muted) !important; }
[role="radiogroup"] > label > div:first-child { display: none !important; } /* hide the radio dot */
[role="radiogroup"] > label p {
  font-family: var(--font-mono) !important;
  font-size: var(--text-2xs) !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  margin: 0 !important;
  color: var(--ink-muted) !important;
}
[role="radiogroup"] > label:has(input:checked) {
  background: var(--ink) !important;
  border-color: var(--ink) !important;
}
[role="radiogroup"] > label:has(input:checked) p { color: var(--bg) !important; }

/* label & caption text for widgets */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
  font-family: var(--font-mono) !important;
  font-size: var(--text-2xs) !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--ink-faint) !important;
  margin-bottom: var(--space-1) !important;
}

/* setup panel wrapper */
.rc-setup {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  padding: var(--space-5) var(--space-5) var(--space-4);
  margin-bottom: var(--space-5);
}

/* summary strip after a run */
.rc-summary-strip {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--space-4);
  padding: var(--space-3) var(--space-4);
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--ink-muted);
  margin-bottom: var(--space-5);
}
.rc-summary-strip code {
  font-family: var(--font-mono);
  color: var(--ink);
  background: transparent;
  padding: 0;
}

/* headline card */
.rc-headline {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(220px, 320px);
  gap: var(--space-6);
  align-items: end;
  padding: var(--space-6) var(--space-6) var(--space-6);
  margin: var(--space-3) 0 var(--space-5);
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
}
.rc-headline__label {
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin: 0 0 var(--space-2);
}
.rc-headline__value {
  font-family: var(--font-serif);
  font-feature-settings: 'tnum', 'lnum', 'kern';
  font-size: var(--text-display);
  line-height: var(--lh-display);
  font-weight: 400;
  letter-spacing: -0.025em;
  color: var(--ink);
  margin: 0;
}
.rc-headline__detail {
  font-family: var(--font-serif);
  font-size: var(--text-sm);
  color: var(--ink-muted);
  margin: var(--space-3) 0 0;
  max-width: 52ch;
  line-height: 1.55;
}
.rc-headline__side {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}
.rc-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border-radius: 2px;
  width: fit-content;
}
.rc-pill--ok   { background: var(--ok-bg);   color: var(--ok-ink); }
.rc-pill--warn { background: var(--warn-bg); color: var(--warn-ink); }
.rc-pill--err  { background: var(--err-bg);  color: var(--err-ink); }
.rc-pill__dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: currentColor;
}
.rc-fact {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  color: var(--ink-muted);
  padding: 6px 0;
  border-bottom: 1px solid var(--rule);
}
.rc-fact:last-child { border-bottom: none; }
.rc-fact__k { color: var(--ink-faint); letter-spacing: 0.05em; text-transform: uppercase; font-size: var(--text-2xs); }
.rc-fact__v { color: var(--ink); font-feature-settings: 'tnum', 'lnum'; }

/* clip-window strip */
.rc-clip {
  margin-top: var(--space-3);
}
.rc-clip__label {
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--ink-faint);
  margin-bottom: var(--space-2);
}

/* skeleton for running state */
.rc-skel {
  display: block;
  height: var(--text-display);
  width: 38%;
  background: linear-gradient(90deg,
    var(--surface-2) 0%, var(--surface) 50%, var(--surface-2) 100%);
  background-size: 200% 100%;
  animation: rc-shimmer 1600ms ease-in-out infinite;
  border-radius: 2px;
}
@keyframes rc-shimmer {
  0%   { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}

/* preview chrome */
.rc-preview-chrome {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: var(--space-6);
  margin-bottom: var(--space-3);
  padding-bottom: var(--space-2);
  border-bottom: 1px solid var(--rule);
}
.rc-preview-title {
  font-family: var(--font-serif);
  font-weight: 500;
  font-size: var(--text-md);
  letter-spacing: -0.01em;
  margin: 0;
}
.rc-preview-kicker {
  font-family: var(--font-mono);
  font-size: var(--text-2xs);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-faint);
}

/* expander (log strip) — Streamlit paints the <summary> header dark once
   the details element is open, and recent versions wrap the header in a
   <button>/<details> whose background flips on hover. Hold every layer on
   the surface tokens so neither open-state nor hover repaints it black. */
[data-testid="stExpander"],
[data-testid="stExpander"] > details,
[data-testid="stExpander"] > details[open] {
  border: none !important;
  border-top: 1px solid var(--rule) !important;
  border-radius: 0 !important;
  background: transparent !important;
  background-color: transparent !important;
  margin-top: var(--space-6) !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > details > summary,
[data-testid="stExpander"] > details[open] > summary,
[data-testid="stExpander"] button,
[data-testid="stExpander"] [data-testid="stExpanderToggle"],
[data-testid="stExpander"] [data-testid="stExpanderHeader"] {
  background: transparent !important;
  background-color: transparent !important;
  color: var(--ink-faint) !important;
  font-family: var(--font-mono) !important;
  font-size: var(--text-2xs) !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  padding: var(--space-3) var(--space-1) !important;
  box-shadow: none !important;
  border: none !important;
}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] > details > summary:hover,
[data-testid="stExpander"] > details[open] > summary:hover,
[data-testid="stExpander"] button:hover,
[data-testid="stExpander"] [data-testid="stExpanderToggle"]:hover,
[data-testid="stExpander"] [data-testid="stExpanderHeader"]:hover {
  background: var(--surface) !important;
  background-color: var(--surface) !important;
  color: var(--ink) !important;
}
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] button svg {
  fill: currentColor !important;
  stroke: currentColor !important;
}
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  background: transparent !important;
  background-color: transparent !important;
}

/* st.code block (log pane inside the expander) — Streamlit's default theme
   ships a dark Prism background that fights the cream light palette. Force
   it onto our surface tokens so the log reads cleanly in both modes, and
   hold that background on :hover too (Streamlit's base CSS flips it dark
   when the pointer enters the block). */
[data-testid="stCode"],
[data-testid="stCodeBlock"],
[data-testid="stCode"]:hover,
[data-testid="stCodeBlock"]:hover {
  background: var(--surface) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 2px !important;
}
[data-testid="stCode"] pre,
[data-testid="stCodeBlock"] pre,
[data-testid="stCode"] code,
[data-testid="stCodeBlock"] code,
[data-testid="stCode"] pre:hover,
[data-testid="stCodeBlock"] pre:hover,
[data-testid="stCode"] code:hover,
[data-testid="stCodeBlock"] code:hover {
  background: transparent !important;
  color: var(--ink) !important;
  font-family: var(--font-mono) !important;
  font-size: var(--text-xs) !important;
  line-height: var(--lh-xs) !important;
}
[data-testid="stCode"] span,
[data-testid="stCodeBlock"] span {
  color: inherit !important;
  background: transparent !important;
}
/* Streamlit's copy button sits in an absolutely-positioned toolbar that
   fades in on hover — its default dark chrome is what reads as "turns
   black" when the pointer enters the log pane. Re-skin every layer of the
   toolbar (and its pseudo-backgrounds) onto the surface tokens. */
[data-testid="stCode"] [data-testid="stCodeCopyButton"],
[data-testid="stCodeBlock"] [data-testid="stCodeCopyButton"],
[data-testid="stCode"] button,
[data-testid="stCodeBlock"] button,
[data-testid="stCode"] [class*="copy"],
[data-testid="stCodeBlock"] [class*="copy"],
[data-testid="stCode"] [class*="Copy"],
[data-testid="stCodeBlock"] [class*="Copy"] {
  background: transparent !important;
  background-color: transparent !important;
  color: var(--ink-muted) !important;
  border: 1px solid var(--rule) !important;
  box-shadow: none !important;
}
[data-testid="stCode"] button:hover,
[data-testid="stCodeBlock"] button:hover,
[data-testid="stCode"] [class*="copy"]:hover,
[data-testid="stCodeBlock"] [class*="copy"]:hover,
[data-testid="stCode"] [class*="Copy"]:hover,
[data-testid="stCodeBlock"] [class*="Copy"]:hover {
  background: var(--surface-2) !important;
  background-color: var(--surface-2) !important;
  color: var(--ink) !important;
}
[data-testid="stCode"] button svg,
[data-testid="stCodeBlock"] button svg,
[data-testid="stCode"] [class*="copy"] svg,
[data-testid="stCodeBlock"] [class*="copy"] svg {
  fill: currentColor !important;
  stroke: currentColor !important;
}

/* alerts (st.error / st.warning) */
[data-testid="stAlert"] {
  border-radius: 3px !important;
  border: 1px solid var(--rule) !important;
  font-family: var(--font-serif) !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
  background: var(--surface-2) !important;
}

/* spinner label */
[data-testid="stSpinner"] p {
  font-family: var(--font-mono) !important;
  font-size: var(--text-xs) !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--ink-muted) !important;
}

/* subtle entrance for the setup panel / headline / preview */
.rc-setup, .rc-headline, .rc-summary-strip, .rc-preview-chrome {
  animation: rc-rise 520ms cubic-bezier(0.22, 1, 0.36, 1) both;
}
@keyframes rc-rise {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* tables emitted via st.dataframe */
[data-testid="stDataFrame"] {
  font-family: var(--font-mono) !important;
  font-size: var(--text-xs) !important;
}
</style>
""".replace("__LIGHT__", _LIGHT_TOKENS).replace("__DARK__", _DARK_TOKENS)

# Manual theme overrides — emitted *after* _CSS so source-order wins over the
# prefers-color-scheme media query. "system" emits nothing and lets the media
# query decide.
_FORCE_LIGHT_CSS = f"<style>:root {{ {_LIGHT_TOKENS} }}</style>"
_FORCE_DARK_CSS = f"<style>:root {{ {_DARK_TOKENS} }}</style>"


# Report palette overrides — must mirror the variable names defined in
# `reflection_coefficient.rw_report._CSS` (shared with irregular_report).
# When the user forces a theme in the app chrome, we inject one of these
# into the generated report HTML before embedding it in an iframe, so the
# preview tracks the app theme instead of the iframe's own
# prefers-color-scheme.
_REPORT_LIGHT = """
:root {
  --color-background-primary:#ffffff;--color-background-secondary:#f5f5f2;
  --color-background-success:#EAF3DE;--color-background-warning:#FAEEDA;
  --color-background-danger:#FCEBEB;--color-text-primary:#1a1a1a;
  --color-text-secondary:#5f5e5a;--color-border-tertiary:rgba(0,0,0,0.15);
  --color-border-secondary:rgba(0,0,0,0.3);
}
"""

_REPORT_DARK = """
:root {
  --color-background-primary:#1a1a1a;--color-background-secondary:#2c2c2a;
  --color-background-success:#173404;--color-background-warning:#412402;
  --color-background-danger:#501313;--color-text-primary:#e8e8e8;
  --color-text-secondary:#b4b2a9;--color-border-tertiary:rgba(255,255,255,0.15);
  --color-border-secondary:rgba(255,255,255,0.3);
}
/* Suppress the report's own prefers-color-scheme rule when we force light.
   This override sits in a second @media block so the user's chosen dark
   palette survives even in OS light mode (and vice versa). */
@media (prefers-color-scheme: light) {
  :root {
    --color-background-primary:#1a1a1a;--color-background-secondary:#2c2c2a;
    --color-background-success:#173404;--color-background-warning:#412402;
    --color-background-danger:#501313;--color-text-primary:#e8e8e8;
    --color-text-secondary:#b4b2a9;--color-border-tertiary:rgba(255,255,255,0.15);
    --color-border-secondary:rgba(255,255,255,0.3);
  }
}
"""

_REPORT_LIGHT_FORCE = """
@media (prefers-color-scheme: dark) {
  :root {
    --color-background-primary:#ffffff;--color-background-secondary:#f5f5f2;
    --color-background-success:#EAF3DE;--color-background-warning:#FAEEDA;
    --color-background-danger:#FCEBEB;--color-text-primary:#1a1a1a;
    --color-text-secondary:#5f5e5a;--color-border-tertiary:rgba(0,0,0,0.15);
    --color-border-secondary:rgba(0,0,0,0.3);
  }
}
"""


def _themed_report_html(html_str: str, mode: str) -> str:
    """Inject a theme override into a generated report before iframe embed.

    ``mode`` is the user's chrome-theme choice ("system" | "light" | "dark").
    "system" leaves the HTML untouched — the iframe already follows the OS
    preference via the report's own media query. For "light" / "dark" we
    splice a <style> block just before </head> that pins both the default
    ``:root`` and the relevant media-query branch onto the chosen palette,
    so the report does not revert when the OS preference disagrees.
    """
    if mode == "light":
        override = f"<style>{_REPORT_LIGHT}{_REPORT_LIGHT_FORCE}</style>"
    elif mode == "dark":
        override = f"<style>{_REPORT_DARK}</style>"
    else:
        return html_str
    # Place right before </head> so source order beats the original :root.
    # Fallback to prepending if the document lacks a head (shouldn't happen
    # with the current reports, but be defensive).
    if "</head>" in html_str:
        return html_str.replace("</head>", f"{override}</head>", 1)
    return override + html_str


# --- plumbing (unchanged from prior revision) --------------------------------

def _pick(kind: str) -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        if kind == "file":
            return filedialog.askopenfilename() or None
        return filedialog.askdirectory() or None
    finally:
        root.destroy()


def _path_picker(label: str, key: str, kind: str, default: Path) -> Path:
    tkey = f"{key}_text"
    pkey = f"{key}_pending"
    # Apply any path staged by a prior Browse click *before* the widget exists.
    if pkey in st.session_state:
        st.session_state[tkey] = st.session_state.pop(pkey)
    if tkey not in st.session_state:
        st.session_state[tkey] = str(default)
    c1, c2 = st.columns([6, 1])
    c1.text_input(label, key=tkey, label_visibility="visible")
    with c2:
        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
        if st.button("Browse", key=f"{key}_btn", use_container_width=True):
            picked = _pick(kind)
            if picked:
                st.session_state[pkey] = picked
                st.rerun()
    return Path(st.session_state[tkey])


@contextmanager
def _scratch_dir():
    path = Path(tempfile.mkdtemp(prefix="rc_preview_"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_kr_vs_freq(results, out_dir: Path, method: str,
                      window_mode: str = "canonical") -> Path:
    rows = sorted(results, key=lambda r: r.f_Hz)
    suffix = "" if window_mode == "canonical" else f"_{window_mode}"
    csv_path = out_dir / f"rw_kr_vs_freq_{method}{suffix}.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_id", "f_Hz", "k_rad_m", "L_m",
                    "H_I_m", "H_R_m", "Kr", "singularity_ok"])
        for r in rows:
            w.writerow([r.test_id, f"{r.f_Hz:.6f}", f"{r.k:.6f}",
                        f"{r.wavelength_m:.6f}", f"{r.H_I:.6f}",
                        f"{r.H_R:.6f}", f"{r.Kr:.6f}",
                        int(r.singularity_ok)])
    return csv_path


def _prune_logs(log_dir: Path, keep: int) -> None:
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    for old in logs[:-keep] if keep > 0 else logs:
        try:
            old.unlink()
        except OSError:
            pass


@contextmanager
def _open_log():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / (datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    fh = log_path.open("w", encoding="utf-8")

    def log(msg: str = "") -> None:
        fh.write(msg + "\n")
        fh.flush()

    try:
        yield log_path, log
    finally:
        fh.close()
        _prune_logs(_LOG_DIR, _LOG_KEEP)


# --- HTML composition --------------------------------------------------------

def _pill(label: str, tone: str) -> str:
    cls = {"ok": "rc-pill--ok", "warn": "rc-pill--warn", "err": "rc-pill--err"}[tone]
    return (
        f'<span class="rc-pill {cls}"><span class="rc-pill__dot"></span>'
        f'{_html.escape(label)}</span>'
    )


def _fact(key: str, value: str) -> str:
    return (
        f'<div class="rc-fact">'
        f'<span class="rc-fact__k">{_html.escape(key)}</span>'
        f'<span class="rc-fact__v">{_html.escape(value)}</span>'
        f'</div>'
    )


def _clip_bar(t_start: float, t_end: float,
              t_ana_start: float, t_ana_end: float,
              t_full: float) -> str:
    """Miniature SVG strip showing the travel-time clip window and the
    analysis window within the full record."""
    t_full = max(t_full, t_end + 0.5)
    scale = 100.0 / t_full

    def x(t: float) -> float:
        return max(0.0, min(100.0, t * scale))

    clip_x, clip_w = x(t_start), x(t_end) - x(t_start)
    ana_x, ana_w = x(t_ana_start), x(t_ana_end) - x(t_ana_start)
    return (
        '<div class="rc-clip">'
        '<div class="rc-clip__label">Clip window</div>'
        '<svg viewBox="0 0 100 10" preserveAspectRatio="none" '
        'style="width:100%;height:20px;display:block;">'
        '<rect x="0" y="4" width="100" height="2" fill="var(--rule)"/>'
        f'<rect x="{clip_x:.2f}" y="3" width="{clip_w:.2f}" height="4" '
        'fill="var(--rule-strong)"/>'
        f'<rect x="{ana_x:.2f}" y="2.2" width="{ana_w:.2f}" height="5.6" '
        'fill="var(--accent)"/>'
        '</svg>'
        f'<div style="display:flex;justify-content:space-between;'
        'font-family:var(--font-mono);font-size:var(--text-2xs);'
        'color:var(--ink-faint);margin-top:4px;">'
        f'<span>0 s</span><span>{t_full:.1f} s</span></div>'
        '</div>'
    )


def _headline_regular(rows: list[dict]) -> str:
    if not rows:
        return ""
    kr_values = [r["Kr"] for r in rows]
    kr_min, kr_max = min(kr_values), max(kr_values)
    n_sing = sum(1 for r in rows if not r["singularity_ok"])
    n_total = len(rows)

    if n_total == 1:
        r = rows[0]
        kr_text = f"{r['Kr']:.3f}"
        pill = (_pill("Singular", "warn") if not r["singularity_ok"]
                else _pill("Clean", "ok"))
        detail = (
            f'<p class="rc-headline__detail">Reflection coefficient at '
            f'<span style="font-family:var(--font-mono);">{r["f_Hz"]:.3f} Hz</span> '
            f'from test <span style="font-family:var(--font-mono);">{_html.escape(r["test_id"])}</span>. '
            f'H<sub>I</sub> = {r["H_I"]:.4f} m, H<sub>R</sub> = {r["H_R"]:.4f} m.</p>'
        )
        facts = _fact("Test", r["test_id"]) + _fact("f", f"{r['f_Hz']:.3f} Hz")
    else:
        kr_text = f"{kr_min:.2f}–{kr_max:.2f}"
        pill = (_pill(f"{n_sing} flagged", "warn") if n_sing
                else _pill("All clean", "ok"))
        detail = (
            f'<p class="rc-headline__detail">{n_total} regular-wave tests '
            f'aggregated into a K<sub>r</sub>(f) sweep. '
            f'{n_sing} of {n_total} hit the singularity mask.</p>'
        )
        facts = (
            _fact("Tests", str(n_total))
            + _fact("Kr range", f"{kr_min:.3f} – {kr_max:.3f}")
            + _fact("Flagged", f"{n_sing} / {n_total}")
        )

    return (
        '<div class="rc-headline">'
        '<div>'
        '<p class="rc-headline__label">Kr</p>'
        f'<p class="rc-headline__value">{kr_text}</p>'
        f'{detail}'
        '</div>'
        f'<div class="rc-headline__side">{pill}{facts}</div>'
        '</div>'
    )


def _headline_irregular(result: IrregularResult, meta) -> str:
    d = result.diagnostics
    ok = d.get("D_or_sin2_min", 1.0) > (
        0.1 if result.method == "least_squares" else 0.05
    )
    pill = _pill("Clean", "ok") if ok else _pill("Near singularity", "warn")
    detail = (
        f'<p class="rc-headline__detail">Energy-based K<sub>r,overall</sub> '
        f'from {result.method.replace("_", "–")} separation. '
        f'H<sub>m0,I</sub> = {result.Hm0_I:.4f} m, '
        f'H<sub>m0,R</sub> = {result.Hm0_R:.4f} m, '
        f'T<sub>p</sub> = {result.Tp_I:.3f} s.</p>'
    )
    facts = (
        _fact("Test", result.test_id)
        + _fact("Bins", f"{d['n_bins_valid']} valid")
        + _fact("min D/sin²", f"{d['D_or_sin2_min']:.3f}")
    )
    clip = _clip_bar(
        float(d.get("t_start_s", 0.0)),
        float(d.get("t_end_s", 0.0)),
        float(d.get("t_analysis_start_s", d.get("t_start_s", 0.0))),
        float(d.get("t_analysis_end_s", d.get("t_end_s", 0.0))),
        float(d.get("runtime_bound_s", d.get("t_end_s", 1.0) * 1.2)),
    )
    return (
        '<div class="rc-headline">'
        '<div>'
        '<p class="rc-headline__label">Kr overall</p>'
        f'<p class="rc-headline__value">{result.Kr_overall:.3f}</p>'
        f'{detail}'
        f'{clip}'
        '</div>'
        f'<div class="rc-headline__side">{pill}{facts}</div>'
        '</div>'
    )


def _headline_skeleton() -> str:
    return (
        '<div class="rc-headline">'
        '<div>'
        '<p class="rc-headline__label">Kr</p>'
        '<span class="rc-skel"></span>'
        '<p class="rc-headline__detail" style="color:var(--ink-faint);">'
        'Running the separation — the headline will settle in a moment.</p>'
        '</div>'
        '<div class="rc-headline__side">'
        f'{_pill("Computing", "warn")}'
        '</div>'
        '</div>'
    )


def _summary_strip(scheme: str, method: str, window: str, bandwidth: float,
                   head: float, tail: float, n_tests: int,
                   window_mode: str = "canonical",
                   freq_source: str = "bin",
                   goda_pair: str = "13") -> str:
    bw = f"{bandwidth:g} Hz" if window != "none" else "—"
    tests = f"{n_tests} test" + ("s" if n_tests != 1 else "")
    mode_chip = (
        f' · mode <code>{_html.escape(window_mode)}</code>'
        if window_mode != "canonical" else ""
    )
    freq_chip = (
        f' · freq <code>{_html.escape(freq_source)}</code>'
        if scheme == "rw" and freq_source != "bin" else ""
    )
    pair_chip = (
        f' · pair <code>{_html.escape(goda_pair)}</code>'
        if method == "goda" else ""
    )
    return (
        f'<div class="rc-summary-strip">'
        f'<span><code>{_html.escape(_SCHEME_LABELS[scheme])}</code> · '
        f'<code>{tests}</code> · <code>{method}</code>{pair_chip} · '
        f'window <code>{window}</code> ({bw}) · '
        f'head <code>{head:g}s</code> tail <code>{tail:g}s</code>'
        f'{mode_chip}{freq_chip}</span>'
        f'<span>edit below</span>'
        f'</div>'
    )


# --- page --------------------------------------------------------------------

st.set_page_config(
    page_title="Reflection coefficient",
    page_icon="◦",
    layout="centered",
)
st.markdown(_CSS, unsafe_allow_html=True)

# theme override — system honours OS / report preview, light/dark force the chrome
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "system"
_mode = st.session_state["theme_mode"]
if _mode == "light":
    st.markdown(_FORCE_LIGHT_CSS, unsafe_allow_html=True)
elif _mode == "dark":
    st.markdown(_FORCE_DARK_CSS, unsafe_allow_html=True)

# masthead — title left, theme selector right, rule beneath
m_left, m_right = st.columns([5, 2])
m_left.markdown(
    '<h1 class="rc-masthead__title" style="margin:0;padding-top:6px;">'
    'Reflection coefficient '
    '<em>— a wave-tank companion</em></h1>',
    unsafe_allow_html=True,
)
with m_right:
    st.radio(
        "Theme", ["system", "light", "dark"],
        horizontal=True, label_visibility="collapsed",
        key="theme_mode",
    )
st.markdown(
    '<div style="border-bottom:1px solid var(--rule);'
    'margin:var(--space-3) 0 var(--space-6);"></div>',
    unsafe_allow_html=True,
)

# setup zone
st.markdown('<div class="rc-setup">', unsafe_allow_html=True)

st.markdown(
    '<div class="rc-section-label">Inputs</div>',
    unsafe_allow_html=True,
)
tank_cfg = _path_picker("Tank config", "tank_cfg", "file",
                        resolve_tank_config(None))
meta_dir = _path_picker("Metadata", "meta_dir", "dir",
                        resolve_metadata_dir(None))
data_dir = _path_picker("Data", "data_dir", "dir",
                        resolve_data_dir(None))

st.markdown(
    '<div class="rc-section-label">Scheme &amp; tests</div>',
    unsafe_allow_html=True,
)
scheme = st.radio("Scheme", ["rw", "wn", "js"], horizontal=True,
                  label_visibility="collapsed")
try:
    tests = list_tests(scheme, data_dir=data_dir, metadata_dir=meta_dir)
except Exception as exc:
    st.error(f"Could not enumerate tests: {exc}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
if not tests:
    st.warning("No tests found for this scheme at the given paths.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

if scheme == "rw":
    selected = st.multiselect(
        "Tests",
        options=tests, default=tests,
        help="Regular wave: pick ≥2 tests to aggregate a Kr-vs-f sweep.",
    )
else:
    selected = [st.selectbox("Test", options=tests)]

st.markdown(
    '<div class="rc-section-label">Analysis</div>',
    unsafe_allow_html=True,
)
c1, c2, c3 = st.columns(3)
method = c1.selectbox(
    "Method", ["least_squares", "goda"],
    help="Goda: 2 probes. Mansard–Funke (least_squares): 3 probes, more robust.",
)
window = c2.selectbox("Window", ["hann", "none"])
bandwidth = c3.number_input("Bandwidth (Hz)", value=0.04, step=0.01,
                            format="%.3f")
st.markdown(
    '<p class="rc-caption">Bandwidth controls the irregular-wave band-average. '
    'Smaller values give finer Kr(f) resolution but noisier curves.</p>',
    unsafe_allow_html=True,
)

c4, c5 = st.columns(2)
head_drop = c4.number_input("Head drop (s)", value=3.0, step=0.5)
tail_drop = c5.number_input("Tail drop (s)", value=3.0, step=0.5)
st.markdown(
    '<p class="rc-caption">Head / tail drop trim the travel-time window to skip '
    'ramp transients at the start and end of the clean record.</p>',
    unsafe_allow_html=True,
)

c6, c7, c8 = st.columns(3)
window_mode = c6.selectbox(
    "Window mode", ["canonical", "noref"],
    help=(
        "canonical: standard post-reflection clip. "
        "noref: pre-reflection window (incident at every probe, first "
        "reflection has not yet returned to wp3) — a baseline sanity "
        "check that should give Kr ≈ 0."
    ),
)
freq_source = c7.selectbox(
    "Frequency source", ["bin", "target"],
    help=(
        "Regular-wave only. bin: use the nearest FFT bin to meta.f_Hz. "
        "target: single-point DFT evaluated at exactly meta.f_Hz, "
        "bypassing bin quantisation (reduces leakage on short clips)."
    ),
    disabled=(scheme != "rw"),
)
goda_pair = c8.selectbox(
    "Goda pair", ["13", "12", "23"],
    help=(
        "Goda only. Which probe pair feeds the two-probe separation: "
        "'13' (wp1 & wp3, widest spacing, default), '12' (wp1 & wp2), "
        "'23' (wp2 & wp3, spacing = X13 − X12). Changing Δ moves the "
        "kΔ = nπ singularities — useful when the default pair sits on "
        "a near-singular frequency."
    ),
    disabled=(method != "goda"),
)
st.markdown(
    '<p class="rc-caption">Window mode switches between the reflection '
    'clip and the pre-reflection incident-only window. Frequency source '
    'applies to regular waves only; Goda pair applies only when the '
    'method is Goda.</p>',
    unsafe_allow_html=True,
)

run_col, _ = st.columns([1, 3])
with run_col:
    run_clicked = st.button(
        "Run analysis",
        type="primary",
        disabled=not selected,
        use_container_width=True,
    )

st.markdown('</div>', unsafe_allow_html=True)  # close .rc-setup

# --- run ---------------------------------------------------------------------

if run_clicked:
    for k in ("report_html", "report_csv", "report_name", "result_rows",
              "result_irregular", "result_meta", "log_path",
              "summary_args"):
        st.session_state.pop(k, None)

    # show a skeleton headline while the spinner runs
    skel_slot = st.empty()
    skel_slot.markdown(_headline_skeleton(), unsafe_allow_html=True)

    with st.spinner("Running separation…"), _open_log() as (log_path, log):
        t0 = time.perf_counter()
        log(f"[streamlit_app] log file: {log_path}")
        bw_txt = f"{bandwidth:g} Hz" if window != "none" else "—"
        pair_txt = f" | goda_pair={goda_pair}" if method == "goda" else ""
        banner = (
            f" {_SCHEME_LABELS[scheme]} | method={method}{pair_txt} "
            f"| window={window} (bw {bw_txt}) "
            f"| drops head {head_drop:g}s tail {tail_drop:g}s "
            f"| mode={window_mode} | freq={freq_source} "
        )
        log("=" * len(banner))
        log(banner)
        log("=" * len(banner))
        log(f"[streamlit_app] tank_config  = {tank_cfg}")
        log(f"[streamlit_app] metadata_dir = {meta_dir}")
        log(f"[streamlit_app] data_dir     = {data_dir}")
        log(f"[streamlit_app] selected {len(selected)} test(s): "
            f"{', '.join(selected)}")

        regular_rows: list[dict] = []
        regular_results, regular_metas = [], []
        irregular_result = None
        irregular_meta = None
        html_bytes, csv_bundle, report_name = None, None, None

        with _scratch_dir() as tmp_path:
            for tid in selected:
                t, e1, e2, e3, meta = load_probe_data(
                    tid, campaign=scheme,
                    tank_config=tank_cfg, metadata_dir=meta_dir,
                    data_dir=data_dir,
                )
                log(f"[streamlit_app] {tid}: N={len(t)}, "
                    f"fs≈{1/(t[1]-t[0]):.1f} Hz")
                try:
                    result = analyse(
                        t, e1, e2, e3, meta, method=method,
                        window=window, bandwidth_Hz=bandwidth,
                        head_drop_s=head_drop, tail_drop_s=tail_drop,
                        window_mode=window_mode,
                        freq_source=freq_source,
                        goda_pair=goda_pair,
                    )
                except Exception as exc:
                    st.error(f"Couldn't analyse {tid}: {exc}")
                    log(f"  !! {tid}: {exc}")
                    continue

                if isinstance(result, RegularResult):
                    regular_results.append(result)
                    regular_metas.append(meta)
                    regular_rows.append({
                        "test_id": result.test_id, "f_Hz": result.f_Hz,
                        "H_I": result.H_I, "H_R": result.H_R,
                        "Kr": result.Kr,
                        "singularity_ok": result.singularity_ok,
                    })
                    log(
                        f"  {result.test_id} [{result.method}] "
                        f"f={result.f_Hz:.3f} Hz  H_I={result.H_I:.4f} m  "
                        f"H_R={result.H_R:.4f} m  Kr={result.Kr:.3f}"
                        + ("" if result.singularity_ok
                           else "  [SINGULARITY]")
                    )
                else:
                    irregular_result = result
                    irregular_meta = meta
                    hp = write_irregular_report(
                        result, meta, tmp_path, method, timestamp="preview",
                        window_mode=window_mode,
                    )
                    html_bytes = hp.read_bytes()
                    mode_suffix = (
                        "" if window_mode == "canonical" else f"_{window_mode}"
                    )
                    cp = tmp_path / (
                        f"{result.test_id}_{method}{mode_suffix}_spectrum.csv"
                    )
                    csv_bundle = (cp.name, cp.read_bytes())
                    report_name = hp.name
                    d = result.diagnostics
                    log(
                        f"  {result.test_id} [{result.method}] "
                        f"Hm0_I={result.Hm0_I:.4f} m  "
                        f"Hm0_R={result.Hm0_R:.4f} m  "
                        f"Tp_I={result.Tp_I:.3f} s  "
                        f"Kr={result.Kr_overall:.3f}  "
                        f"(bins={d['n_bins_valid']}, "
                        f"min D/sin²={d['D_or_sin2_min']:.3f})"
                    )
                    log(f"[streamlit_app] built {report_name} "
                        f"(pending user download)")

            if scheme == "rw" and len(regular_results) >= 2:
                csv_path = _write_kr_vs_freq(
                    regular_results, tmp_path, method,
                    window_mode=window_mode,
                )
                hp = write_rw_report(
                    list(zip(regular_results, regular_metas)),
                    tmp_path, method, csv_path=csv_path, timestamp="preview",
                    window_mode=window_mode,
                )
                html_bytes = hp.read_bytes()
                csv_bundle = (csv_path.name, csv_path.read_bytes())
                report_name = hp.name
                log(f"[streamlit_app] built {report_name} "
                    f"(pending user download)")

        log(f"[streamlit_app] done in {time.perf_counter() - t0:.2f} s; "
            f"report produced: {'yes' if html_bytes is not None else 'no'}")

    skel_slot.empty()

    st.session_state["report_html"] = html_bytes
    st.session_state["report_csv"] = csv_bundle
    st.session_state["report_name"] = report_name
    st.session_state["result_rows"] = regular_rows
    st.session_state["result_irregular"] = irregular_result
    st.session_state["result_meta"] = irregular_meta
    st.session_state["log_path"] = str(log_path)
    st.session_state["summary_args"] = {
        "scheme": scheme, "method": method, "window": window,
        "bandwidth": bandwidth, "head": head_drop, "tail": tail_drop,
        "n_tests": len(selected),
        "window_mode": window_mode, "freq_source": freq_source,
        "goda_pair": goda_pair,
    }

# --- result ------------------------------------------------------------------

html_bytes = st.session_state.get("report_html")
rows = st.session_state.get("result_rows") or []
irregular = st.session_state.get("result_irregular")
irregular_meta = st.session_state.get("result_meta")
summary_args = st.session_state.get("summary_args")

if summary_args is not None:
    st.markdown(_summary_strip(**summary_args), unsafe_allow_html=True)

if irregular is not None:
    st.markdown(
        _headline_irregular(irregular, irregular_meta),
        unsafe_allow_html=True,
    )
elif rows:
    st.markdown(_headline_regular(rows), unsafe_allow_html=True)

# --- preview + download ------------------------------------------------------

if html_bytes is not None:
    kicker = (
        "Regular-wave aggregate" if rows
        else "Irregular-wave report"
    )
    st.markdown(
        f'<div class="rc-preview-chrome">'
        f'<h2 class="rc-preview-title">Report preview</h2>'
        f'<span class="rc-preview-kicker">{kicker}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    components.html(
        _themed_report_html(
            html_bytes.decode("utf-8"),
            st.session_state.get("theme_mode", "system"),
        ),
        height=820,
        scrolling=True,
    )

    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button(
        "Download report",
        data=html_bytes,
        file_name=st.session_state["report_name"],
        mime="text/html",
    )
    if st.session_state.get("report_csv"):
        fname, data = st.session_state["report_csv"]
        d2.download_button(
            "Download CSV",
            data=data,
            file_name=fname,
            mime="text/csv",
        )

# fallback: regular single test (no HTML)
if rows and html_bytes is None:
    st.markdown(
        '<div class="rc-section-label">Single-test scalars</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(rows, use_container_width=True)

# --- log strip ---------------------------------------------------------------

log_path = st.session_state.get("log_path")
if log_path:
    with st.expander(f"Run log · {Path(log_path).name} · newest {_LOG_KEEP} kept"):
        try:
            st.code(Path(log_path).read_text(encoding="utf-8"), language="text")
        except OSError:
            st.write("Log file no longer accessible.")
