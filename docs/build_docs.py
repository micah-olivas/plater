#!/usr/bin/env python3
"""Generate a self-contained HTML reference for plater.

Introspects the live package (signatures + docstrings) and writes a single
offline-friendly `docs/index.html` in a clean, minimal style. Guide example
plots are rendered from synthetic data (via plater's own plotting functions,
see `_examples.py`) and embedded as base64 PNGs so the page stays a single
self-contained file. If figure rendering fails, the page still builds without
them.

Usage:
    python docs/build_docs.py
"""

from __future__ import annotations

import html
import inspect
import re
import sys
import textwrap
from pathlib import Path

import plater as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# Section grouping — maps each function's __module__ to a display section,
# in the order they should appear (mirrors the README / natural workflow).
# --------------------------------------------------------------------------- #
SECTIONS = [
    ("plater.io", "io", "Loading & I/O"),
    ("plater.rates", "rates", "Initial rates"),
    ("plater.kinetics", "kinetics", "Michaelis–Menten"),
    ("plater.standards", "standards", "Standard curves"),
    ("plater.assays", "assays", "Colorimetric assays (BCA)"),
    ("plater.plotting", "plotting", "Plotting"),
]
MODULE_TO_SLUG = {mod: slug for mod, slug, _ in SECTIONS}
MODULE_TO_TITLE = {mod: title for mod, _, title in SECTIONS}


# --------------------------------------------------------------------------- #
# Curated guide — worked workflows lifted from the README. Each entry is
# (anchor id, title, intro prose, code block).
# --------------------------------------------------------------------------- #
GUIDE = [
    (
        "install",
        "Install",
        "Install from source (add the <code>[plot]</code> extra for on-line "
        "rate labels in progress curves):",
        """git clone https://github.com/micah-olivas/plater.git
cd plater
pip install -e .
pip install -e ".[plot]"   # optional: annotate_rates='lines'""",
    ),
    (
        "loading",
        "Loading data",
        "<code>load()</code> parses a Tecan Spark Excel export and auto-detects "
        "the layout. Pass a <code>conditions</code> dict mapping each well to its "
        "metadata <code>[Replicate, Substrate, S (µM), E (nM)]</code>, or omit it "
        "for a first-pass look at an unfamiliar file. Subset a path-corrected / "
        "partial plate with <code>wells=</code>.",
        """import plater as pl

conditions = {
    'A1': [1, 'pNPA', 1250, 100],
    'A2': [1, 'pNPA',  625, 100],
    'B1': [1, 'pNPA', 1250,   0],   # no-enzyme control
}

df = pl.load('myfile.xlsx', conditions=conditions)

# Subset a partial plate:
df = pl.load('plate.xlsx', wells='auto')    # Tecan 'Plate area'
df = pl.load('plate.xlsx', wells='B2-D5')   # rows B-D × cols 2-5""",
    ),
    (
        "folders",
        "Loading a folder",
        "When one experiment is split across several exports (same plate layout "
        "per run), <code>load_folder()</code> loads them all and stacks them, "
        "tagging each row with a <code>Notebook</code> source column.",
        """df = pl.load_folder('runs/', conditions=conditions)   # every *.xlsx
df['Notebook'].unique()                               # ['run1', 'run2', ...]

# one panel per notebook, or overlay them:
pl.plot_progress_curves(df, split_by='Notebook')""",
    ),
    (
        "kinetics-workflow",
        "Kinetics workflow",
        "Compute initial rates per well/condition, fit Michaelis–Menten per "
        "substrate, and plot.",
        """# Linear fit over [0, t_end] per (well x condition)
rates = pl.compute_initial_rates(df, t_end=75)

# Michaelis-Menten fit per substrate
mm = pl.fit_michaelis_menten(
    rates,
    exclude=[{'Substrate': 'pNPA', 'S (µM)': 1250}],
)

# Plots
pl.plot_progress_curves(df, rates_df=rates, t_end_fit=75)
pl.plot_initial_rates(rates, mm_params_df=mm)""",
    ),
    (
        "scans",
        "Kinetic scans",
        "For full-spectrum-vs-time data, load the scan, pick a probe wavelength "
        "with <code>plot_spectra()</code>, then collapse to a single &lambda; "
        "with <code>extract_wavelength()</code> — the result drops straight into "
        "the kinetics workflow above.",
        """scan = pl.load('scan.xlsx', conditions=conditions)
pl.plot_spectra(scan, n_timepoints=8)        # pick a probe wavelength
df = pl.extract_wavelength(scan, 405)        # collapse to single lambda""",
    ),
    (
        "standards",
        "Standard curves",
        "Build a standard curve from no-enzyme wells, fit it, and convert signal "
        "to product concentration.",
        """std = pl.compute_standard_curve(stds_df, conc_col='S (µM)')
fit = pl.fit_standard_curve(std)             # slope = extinction coefficient

# convert a rates frame to d[P]/dt, or long-form signal to [P]:
rates = pl.apply_standard_curve(rates, fit, product_name='P', conc_unit='µM')""",
    ),
]


# --------------------------------------------------------------------------- #
# Example media — maps a guide anchor to the artifact(s) shown beside its code
# block. Each entry is (kind, key, caption): kind is 'img' (key → base64 PNG)
# or 'html' (key → a ready-made HTML fragment, e.g. a DataFrame table). Keys
# index the dict from `render_guide_figures()`. Captions are HTML.
# --------------------------------------------------------------------------- #
GUIDE_FIGURES = {
    "loading": [
        ("html", "loading-df",
         "The tidy DataFrame <code>load()</code> returns: one row per well × "
         "timepoint, with your <code>conditions</code> merged onto each well "
         "(<code>df.head()</code>)."),
    ],
    "folders": [
        ("img", "folders",
         "<code>split_by='Notebook'</code> facets the stacked frame into one "
         "panel per run, sharing a y-axis for easy comparison."),
    ],
    "kinetics-workflow": [
        ("img", "kinetics-workflow",
         "Progress curves with the linear fit window (here [0, 75 s]) marked — "
         "the slopes over that window become the initial rates."),
        ("img", "kinetics-workflow-mm",
         "<code>plot_initial_rates()</code>: initial rates vs [S] with the "
         "Michaelis–Menten fit overlaid and K<sub>M</sub> / V<sub>max</sub> "
         "annotated."),
    ],
    "scans": [
        ("img", "scans",
         "<code>plot_spectra()</code> — absorbance vs wavelength colored by "
         "time, one panel per well. The product peak near 405&nbsp;nm grows as "
         "the reaction runs; pick it as the probe wavelength."),
    ],
    "standards": [
        ("img", "standards",
         "A product standard curve fit with <code>plot_standard_curves()</code>; "
         "the slope is the extinction coefficient used to convert signal to [P]."),
    ],
}

# Columns rendered right-aligned + monospace (numeric) in DataFrame previews.
_DF_NUMERIC_COLS = {"Replicate", "S (µM)", "E (nM)", "Time [s]", "Absorbance"}


def df_preview_html(df) -> str:
    """Render a DataFrame as a compact, modern HTML table (no pandas styling).

    Text columns left-align; numeric columns right-align in tabular-figure
    monospace. Values are formatted per column so the preview reads cleanly.
    """
    def fmt(col, v):
        if col == "Absorbance":
            return f"{v:.3f}"
        if col == "Time [s]":
            return f"{v:g}"
        if col in _DF_NUMERIC_COLS:
            return f"{int(v)}"
        return str(v)

    cols = list(df.columns)
    head = "".join(
        f'<th class="{"num" if c in _DF_NUMERIC_COLS else "txt"}">{esc(str(c))}</th>'
        for c in cols
    )
    body = []
    for _, row in df.iterrows():
        cells = "".join(
            f'<td class="{"num" if c in _DF_NUMERIC_COLS else "txt"}">'
            f"{esc(fmt(c, row[c]))}</td>"
            for c in cols
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="df-preview"><table class="df">'
        f"<thead><tr>{head}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def render_guide_figures() -> dict[str, str]:
    """Return {key: artifact}, or {} if rendering is unavailable.

    Values are base64 PNGs for plot keys and ready-made HTML for table keys
    (e.g. the loading DataFrame preview). Guide media are nice-to-have; a
    failure here (missing matplotlib, a plotting bug) should never block the
    reference build, so we swallow it and emit a guide with code blocks only.
    """
    try:
        from _examples import example_load_df, render_example_figures

        figs = render_example_figures()
        figs["loading-df"] = df_preview_html(example_load_df().head(5))
        return figs
    except Exception as exc:  # noqa: BLE001 — figures are optional
        print(f"  (skipping example figures: {exc})")
        return {}


# --------------------------------------------------------------------------- #
# Docstring rendering
# --------------------------------------------------------------------------- #
def esc(s: str) -> str:
    return html.escape(s, quote=True)


_PY_KEYWORDS = {
    "import", "from", "as", "for", "in", "if", "else", "elif", "return",
    "def", "None", "True", "False", "and", "or", "not", "with", "while",
    "lambda", "class",
}

_HL_RE = re.compile(
    r"(?P<comment>\#[^\n]*)"
    r"|(?P<string>'[^'\n]*'|\"[^\"\n]*\")"
    r"|(?P<number>\b\d+\.?\d*\b)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)


def highlight_code(code: str) -> str:
    """Build-time syntax highlighting → HTML with <span class="tok-*"> tokens.

    Covers the guide's Python (and shell-ish install) snippets: comments,
    strings, numbers, keywords, the `pl` namespace, and call names. All text is
    HTML-escaped; anything unmatched passes through escaped, so the output is
    always safe and round-trips the original characters.
    """
    out: list[str] = []
    pos = 0
    for m in _HL_RE.finditer(code):
        out.append(esc(code[pos:m.start()]))
        kind = m.lastgroup
        text = m.group()
        cls = None
        if kind == "comment":
            cls = "c"
        elif kind == "string":
            cls = "s"
        elif kind == "number":
            cls = "n"
        elif kind == "name":
            if text in _PY_KEYWORDS:
                cls = "k"
            elif text == "pl":
                cls = "b"
            elif code[m.end():m.end() + 1] == "(":
                cls = "f"
        out.append(f'<span class="tok-{cls}">{esc(text)}</span>' if cls else esc(text))
        pos = m.end()
    out.append(esc(code[pos:]))
    return "".join(out)


_INLINE_DOUBLE = re.compile(r"``(.+?)``")
_INLINE_SINGLE = re.compile(r"`([^`]+?)`")
_BULLET = re.compile(r"^\s*[-*]\s")


def inline_code(escaped: str) -> str:
    """Turn ``code`` / `code` (already HTML-escaped) into <code> spans."""
    s = _INLINE_DOUBLE.sub(r"<code>\1</code>", escaped)
    s = _INLINE_SINGLE.sub(r"<code>\1</code>", s)
    return s


def flow(text: str) -> str:
    """Render a description block as readable HTML.

    Plain prose paragraphs are reflowed (source hard-wraps joined into a single
    flowing line). Paragraphs containing bullets or aligned enums are preserved
    verbatim in a wrapping <pre> so their structure survives.
    """
    text = textwrap.dedent(text).strip("\n")
    if not text.strip():
        return ""
    out = []
    for para in re.split(r"\n\s*\n", text):
        lines = [ln for ln in para.split("\n")]
        if not any(ln.strip() for ln in lines):
            continue
        if any(_BULLET.match(ln) for ln in lines):
            out.append(f'<pre class="pdesc">{esc(para.rstrip())}</pre>')
        else:
            joined = " ".join(ln.strip() for ln in lines if ln.strip())
            out.append(f'<p class="pdesc-flow">{inline_code(esc(joined))}</p>')
    return "\n".join(out)


def render_params(body: list[str]) -> str:
    """Render a Parameters/Other Parameters body as a definition list."""
    entries: list[dict] = []
    cur: dict | None = None
    for line in body:
        is_header = bool(line.strip()) and not line[:1].isspace()
        if is_header:
            if cur is not None:
                entries.append(cur)
            names, _, typ = line.partition(" : ")
            cur = {"names": names.strip(), "type": typ.strip(), "desc": []}
        elif cur is not None:
            cur["desc"].append(line)
    if cur is not None:
        entries.append(cur)

    if not entries:
        return flow("\n".join(body))

    out = ['<dl class="params">']
    for e in entries:
        dt = f'<code class="pname">{esc(e["names"])}</code>'
        if e["type"]:
            dt += f' <span class="ptype">{inline_code(esc(e["type"]))}</span>'
        desc = flow("\n".join(e["desc"])) if any(x.strip() for x in e["desc"]) else ""
        out.append(f"<dt>{dt}</dt><dd>{desc}</dd>")
    out.append("</dl>")
    return "\n".join(out)


def split_sections(lines: list[str]):
    """Split docstring body into (intro_lines, [(section_name, body_lines)])."""
    sections: list[tuple[str, list[str]]] = []
    name: str | None = None
    body: list[str] = []
    intro: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line.strip() and nxt and set(nxt) == {"-"}:
            if name is None:
                intro = body
            else:
                sections.append((name, body))
            name = line.strip()
            body = []
            i += 2
            continue
        body.append(line)
        i += 1
    if name is None:
        intro = body
    else:
        sections.append((name, body))
    return intro, sections


def render_docstring(doc: str | None) -> str:
    """Render a NumPy-style docstring into clean, scannable HTML."""
    if not doc or not doc.strip():
        return '<p class="muted">No description available.</p>'

    lines = doc.split("\n")

    # First paragraph -> summary.
    i = 0
    summary_lines: list[str] = []
    while i < len(lines) and lines[i].strip():
        summary_lines.append(lines[i].strip())
        i += 1
    summary = " ".join(summary_lines)
    while i < len(lines) and not lines[i].strip():
        i += 1
    rest = lines[i:]

    parts = [f'<p class="fn-summary">{inline_code(esc(summary))}</p>']

    intro, sections = split_sections(rest)
    if any(ln.strip() for ln in intro):
        parts.append(flow("\n".join(intro)))

    for name, body in sections:
        parts.append(f'<h4 class="doc-section">{esc(name)}</h4>')
        low = name.lower()
        if low in ("parameters", "other parameters"):
            parts.append(render_params(body))
        elif low in ("examples", "example"):
            code = textwrap.dedent("\n".join(body)).strip("\n")
            parts.append(f'<pre class="doc-text">{esc(code)}</pre>')
        else:
            parts.append(flow("\n".join(body)))

    return "\n".join(parts)


def summary_of(doc: str | None) -> str:
    if not doc or not doc.strip():
        return ""
    out = []
    for ln in doc.split("\n"):
        if not ln.strip():
            break
        out.append(ln.strip())
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Introspection
# --------------------------------------------------------------------------- #
def collect():
    """Walk plater.__all__ -> (grouped functions, config constants)."""
    grouped: dict[str, list[dict]] = {slug: [] for _, slug, _ in SECTIONS}
    constants: list[tuple[str, str]] = []

    for name in pl.__all__:
        obj = getattr(pl, name)
        if inspect.ismodule(obj):
            continue  # the `style` re-export
        if inspect.isfunction(obj):
            mod = obj.__module__
            slug = MODULE_TO_SLUG.get(mod)
            if slug is None:
                continue
            try:
                sig = str(inspect.signature(obj))
            except (ValueError, TypeError):
                sig = "(...)"
            doc = inspect.getdoc(obj)
            grouped[slug].append(
                {
                    "name": name,
                    "sig": sig,
                    "doc": doc,
                    "summary": summary_of(doc),
                }
            )
        else:
            constants.append((name, repr(obj)))

    return grouped, constants


# --------------------------------------------------------------------------- #
# HTML assembly
# --------------------------------------------------------------------------- #
CSS = """
:root {
  --bg: #F8FAFC; --surface: #FFFFFF; --text: #1E293B; --muted: #64748B;
  --border: #E2E8F0; --code-bg: #EAEFF3; --accent: #2563EB; --accent-soft: #EFF4FE;
  --sidebar-w: 248px; --measure: 780px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 16px; line-height: 1.6; -webkit-font-smoothing: antialiased;
}
code, pre, .mono {
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
}

/* layout */
.sidebar {
  position: fixed; top: 0; left: 0; width: var(--sidebar-w); height: 100vh;
  overflow-y: auto; border-right: 1px solid var(--border); background: var(--surface);
  padding: 20px 16px 48px;
}
main {
  margin-left: var(--sidebar-w); padding: 48px 40px 120px;
  max-width: calc(var(--measure) + 80px);
}
.wrap { max-width: var(--measure); }

/* sidebar */
.brand { font-weight: 700; font-size: 18px; letter-spacing: -0.01em; margin: 0 0 16px; }
.search input {
  width: 100%; padding: 8px 10px; font-size: 14px; color: var(--text);
  border: 1px solid var(--border); border-radius: 8px; background: var(--bg);
  margin-bottom: 18px; outline: none;
}
.search input:focus-visible { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.nav-h {
  display: block; font-size: 12px; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); margin: 18px 0 6px;
}
a.nav-h { text-decoration: none; }
.nav-fn, .nav-sub {
  display: block; padding: 3px 8px; margin: 1px 0; border-radius: 6px;
  font-size: 14px; color: var(--text); text-decoration: none;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  transition: background-color 180ms, color 180ms;
}
.nav-fn { font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace; }
.nav-fn:hover, .nav-sub:hover { background: var(--code-bg); }
.nav-fn.active, .nav-sub.active { background: var(--accent-soft); color: var(--accent); }

/* headings */
h1 { font-size: 34px; font-weight: 700; letter-spacing: -0.02em; margin: 0 0 6px; }
.lede { color: var(--muted); font-size: 17px; margin: 0 0 8px; }
.version { color: var(--muted); font-size: 14px; }
h2 {
  font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 56px 0 20px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
}
h3 { font-size: 22px; font-weight: 650; letter-spacing: -0.01em; margin: 40px 0 16px; }

/* links */
a { color: var(--accent); }

/* guide */
.guide-block { margin: 0 0 32px; }
.guide-block h3 { margin-top: 28px; }
.guide-block p { margin: 0 0 12px; }

/* Code + plot side by side. The row "breaks out" wider than the prose measure
   (which stays narrow for readability) so both the code and the plot get real
   room; it shares the left edge with the surrounding text and only extends to
   the right into the empty margin. Stacks vertically on narrow screens. */
.guide-cols {
  display: flex; gap: 24px; align-items: flex-start;
  width: min(1080px, calc(100vw - 300px));
}
.guide-cols .guide-code { flex: 1.3 1 0; min-width: 0; }
.guide-cols .guide-figs { flex: 1 1 0; min-width: 0; }
.guide-cols .guide-code pre { margin: 0; font-size: 12.5px; line-height: 1.5; }
/* Table media: size every part to its content so nothing clips and the row
   stays only as wide as it needs. The code column hugs its longest line; the
   table column hugs its natural per-column widths (header + cells). The
   caption is pinned to the table's width so it wraps instead of widening the
   box (width:0 keeps it out of the box's intrinsic-width calc). */
.guide-cols--wide-media { width: fit-content; max-width: calc(100vw - 260px); }
.guide-cols--wide-media .guide-code { flex: 0 0 auto; min-width: 0; }
.guide-cols--wide-media .guide-figs { flex: 0 0 auto; }
.guide-cols--wide-media .guide-fig { width: fit-content; max-width: 100%; }
.guide-cols--wide-media .guide-fig figcaption { width: 0; min-width: 100%; }
@media (max-width: 900px) {
  .guide-cols, .guide-cols--wide-media { width: auto; flex-direction: column; }
}

.guide-fig {
  margin: 16px 0 0; padding: 16px; background: var(--surface);
  border: 1px solid var(--border); border-radius: 10px;
}
.guide-figs .guide-fig { margin: 0 0 16px; }
.guide-figs .guide-fig:last-child { margin-bottom: 0; }
.guide-fig img { display: block; max-width: 100%; height: auto; margin: 0 auto; }
.guide-fig figcaption {
  margin: 12px 2px 0; color: var(--muted); font-size: 13.5px; line-height: 1.5;
}

/* modern DataFrame preview table */
.df-preview { overflow-x: auto; }
table.df { width: auto; border-collapse: collapse; font-size: 12.5px; }
table.df thead th {
  text-align: left; font-weight: 600; color: var(--muted);
  font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.04em;
  padding: 0 14px 9px; border-bottom: 1px solid var(--border); white-space: nowrap;
}
table.df tbody td {
  padding: 7px 14px; border-bottom: 1px solid var(--border);
  white-space: nowrap; color: var(--text); vertical-align: middle;
}
table.df tbody tr:last-child td { border-bottom: none; }
table.df tbody tr { transition: background-color 120ms; }
table.df tbody tr:hover td { background: var(--accent-soft); }
table.df th.num, table.df td.num { text-align: right; }
table.df td.num {
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; color: #334155;
}
table.df td.txt:first-child { font-weight: 600; }

/* code blocks */
pre {
  background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 14px 16px; overflow-x: auto; font-size: 13.5px; line-height: 1.55; margin: 0 0 8px;
}
/* guide code blocks: no fill (just a thin frame), so there's no shading box —
   color comes from the syntax highlighting, not a background. */
pre.code { background: transparent; }
/* inline code in prose / captions only — NOT inside <pre> (that background +
   pad was the faint per-token shading, and read as a one-space indent) */
p code, li code, figcaption code {
  background: var(--code-bg); padding: 1px 5px; border-radius: 4px; font-size: 0.9em;
}
pre code {
  background: none; padding: 0; border-radius: 0; font-size: inherit;
}

/* syntax highlighting (rendered at build time) — tuned for the light page */
.tok-c { color: #6B7280; font-style: italic; }   /* comment   */
.tok-s { color: #047857; }                        /* string    */
.tok-n { color: #B45309; }                        /* number    */
.tok-k { color: #7C3AED; }                        /* keyword   */
.tok-f { color: #2563EB; }                        /* call name */
.tok-b { color: #DB2777; font-weight: 600; }      /* pl module */

/* function cards */
.fn-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 20px 22px; margin: 0 0 16px; scroll-margin-top: 24px;
}
.fn-sig { display: block; font-size: 14px; overflow-x: auto; margin: 0 0 12px; padding-bottom: 2px; }
.fn-sig .fn-name { font-weight: 700; color: var(--text); }
.fn-sig .fn-args { color: var(--muted); }
.fn-summary { margin: 0 0 4px; font-size: 16px; }
.doc-section {
  font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em;
  color: var(--muted); margin: 22px 0 10px;
}
pre.doc-text {
  background: #F1F5F9; border: 1px solid var(--border); border-radius: 8px;
  padding: 12px 14px; font-size: 13px; white-space: pre-wrap; word-wrap: break-word; color: var(--text);
}

/* parameter definition list */
dl.params { margin: 4px 0 0; }
dl.params dt { margin: 0; }
dl.params dd {
  margin: 2px 0 16px; padding: 0 0 0 16px; border-left: 2px solid var(--border);
}
dl.params dd:last-child { margin-bottom: 4px; }
.pname {
  background: var(--code-bg); padding: 1.5px 7px; border-radius: 5px;
  font-weight: 700; font-size: 13.5px; color: var(--text);
}
.ptype {
  font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  color: var(--muted); font-size: 13px; margin-left: 4px;
}
.pdesc-flow { margin: 6px 0; font-size: 15px; line-height: 1.55; }
pre.pdesc {
  background: transparent; border: none; padding: 0; margin: 6px 0;
  font-family: inherit; font-size: 14.5px; line-height: 1.5;
  white-space: pre-wrap; word-wrap: break-word; color: var(--text);
}
.muted { color: var(--muted); }

/* config table */
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
td.k { font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace; font-weight: 600; white-space: nowrap; }
td.v { font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace; color: var(--muted); word-break: break-word; }

.hidden { display: none !important; }
.empty-note { color: var(--muted); font-style: italic; }

/* responsive */
@media (max-width: 820px) {
  .sidebar {
    position: static; width: auto; height: auto; border-right: none;
    border-bottom: 1px solid var(--border);
  }
  main { margin-left: 0; padding: 32px 20px 80px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  * { transition: none !important; }
}
"""

JS = """
const q = document.getElementById('q');
const cards = Array.from(document.querySelectorAll('.fn-card'));
const navFns = Array.from(document.querySelectorAll('.nav-fn'));
const refGroups = Array.from(document.querySelectorAll('.ref-group'));
const navGroups = Array.from(document.querySelectorAll('.nav-group'));
const guideEls = Array.from(document.querySelectorAll('[data-guide]'));

function applyFilter() {
  const term = q.value.trim().toLowerCase();
  const searching = term.length > 0;
  guideEls.forEach(el => el.classList.toggle('hidden', searching));
  cards.forEach(c => {
    const hay = (c.dataset.name + ' ' + c.dataset.summary).toLowerCase();
    c.classList.toggle('hidden', searching && !hay.includes(term));
  });
  navFns.forEach(a => {
    const hay = a.dataset.name.toLowerCase();
    a.classList.toggle('hidden', searching && !hay.includes(term));
  });
  refGroups.forEach(g => {
    const any = g.querySelectorAll('.fn-card:not(.hidden)').length > 0;
    g.classList.toggle('hidden', !any);
  });
  navGroups.forEach(g => {
    const any = g.querySelectorAll('.nav-fn:not(.hidden)').length > 0;
    g.classList.toggle('hidden', !any);
  });
}
q.addEventListener('input', applyFilter);

// scroll-spy: highlight the sidebar link for the card in view
const linkByName = {};
navFns.forEach(a => { linkByName[a.dataset.name] = a; });
const obs = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    const link = linkByName[e.target.dataset.name];
    if (!link) return;
    if (e.isIntersecting) {
      navFns.forEach(a => a.classList.remove('active'));
      link.classList.add('active');
    }
  });
}, { rootMargin: '-10% 0px -80% 0px', threshold: 0 });
cards.forEach(c => obs.observe(c));
"""


def fmt_signature(name: str, sig: str) -> str:
    return (
        f'<code class="fn-sig"><span class="fn-name">{esc(name)}</span>'
        f'<span class="fn-args">{esc(sig)}</span></code>'
    )


def build_html(grouped, constants, figures=None) -> str:
    try:
        from importlib.metadata import version

        ver = version("plater")
    except Exception:
        ver = ""
    lede = (pl.__doc__ or "").strip().split("\n")[0]

    # ---- sidebar ----
    side = ['<aside class="sidebar">']
    side.append('<div class="brand">plater</div>')
    side.append('<div class="search"><input id="q" type="search" '
                'placeholder="Search functions…" aria-label="Search functions"></div>')
    side.append('<a class="nav-h" href="#guide" data-guide>Guide</a>')
    for gid, title, _, _ in GUIDE:
        side.append(f'<a class="nav-sub" href="#{gid}" data-guide>{esc(title)}</a>')
    side.append('<a class="nav-h" href="#reference">Reference</a>')
    for mod, slug, title in SECTIONS:
        fns = grouped.get(slug, [])
        if not fns:
            continue
        side.append(f'<div class="nav-group" data-section="{slug}">')
        side.append(f'<div class="nav-h">{esc(title)}</div>')
        for fn in fns:
            side.append(
                f'<a class="nav-fn" data-name="{esc(fn["name"])}" '
                f'href="#{esc(fn["name"])}">{esc(fn["name"])}</a>'
            )
        side.append("</div>")
    side.append('<a class="nav-h" href="#configuration">Configuration</a>')
    side.append("</aside>")

    # ---- main ----
    m = ['<main><div class="wrap">']
    m.append("<header>")
    m.append('<h1>plater</h1>')
    if lede:
        m.append(f'<p class="lede">{esc(lede)}</p>')
    if ver:
        m.append(f'<p class="version">v{esc(ver)}</p>')
    m.append("</header>")

    # guide
    m.append('<h2 id="guide" data-guide>Guide</h2>')
    figures = figures or {}
    for gid, title, intro, code in GUIDE:
        m.append(f'<section class="guide-block" id="{gid}" data-guide>')
        m.append(f"<h3>{esc(title)}</h3>")
        m.append(f"<p>{intro}</p>")
        code_html = f'<pre class="code"><code>{highlight_code(code)}</code></pre>'

        figs_html = []
        for kind, key, caption in GUIDE_FIGURES.get(gid, []):
            content = figures.get(key)
            if not content:
                continue
            if kind == "img":
                body = (f'<img src="data:image/png;base64,{content}" '
                        f'alt="{esc(title)} example plot">')
            else:  # 'html' — a ready-made fragment (e.g. a DataFrame table)
                body = content
            figs_html.append(
                f'<figure class="guide-fig">{body}'
                f"<figcaption>{caption}</figcaption></figure>"
            )

        if figs_html:
            # code on the left, media on the right (stacks on narrow screens).
            # Table media needs more horizontal room than a plot, so widen +
            # rebalance the row when this block shows an HTML table.
            cols_cls = "guide-cols"
            if any(k == "html" for k, _, _ in GUIDE_FIGURES.get(gid, [])):
                cols_cls += " guide-cols--wide-media"
            m.append(f'<div class="{cols_cls}">')
            m.append(f'<div class="guide-code">{code_html}</div>')
            m.append(f'<div class="guide-figs">{"".join(figs_html)}</div>')
            m.append("</div>")
        else:
            m.append(code_html)
        m.append("</section>")

    # reference
    m.append('<h2 id="reference">Reference</h2>')
    for mod, slug, title in SECTIONS:
        fns = grouped.get(slug, [])
        if not fns:
            continue
        m.append(f'<div class="ref-group" data-section="{slug}">')
        m.append(f"<h3>{esc(title)}</h3>")
        for fn in fns:
            body = render_docstring(fn["doc"])
            m.append(
                f'<section class="fn-card" id="{esc(fn["name"])}" '
                f'data-name="{esc(fn["name"])}" data-summary="{esc(fn["summary"])}">'
            )
            m.append(fmt_signature(fn["name"], fn["sig"]))
            m.append(body)
            m.append("</section>")
        m.append("</div>")

    # configuration
    m.append('<h2 id="configuration">Configuration</h2>')
    m.append('<p class="muted">Tunable module-level defaults &mdash; override on '
             'the package, e.g. <code>pl.DEFAULT_DPI = 200</code>. Values shown are '
             'the current defaults.</p>')
    m.append("<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>")
    for name, val in constants:
        m.append(f'<tr><td class="k">{esc(name)}</td><td class="v">{esc(val)}</td></tr>')
    m.append("</tbody></table>")

    m.append("</div></main>")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>plater — reference</title>
<style>{CSS}</style>
</head>
<body>
{''.join(side)}
{''.join(m)}
<script>{JS}</script>
</body>
</html>
"""


def main():
    grouped, constants = collect()
    figures = render_guide_figures()
    out = Path(__file__).resolve().parent / "index.html"
    out.write_text(build_html(grouped, constants, figures), encoding="utf-8")
    n = sum(len(v) for v in grouped.values())
    print(f"Wrote {out}  ({n} functions, {len(constants)} config constants, "
          f"{len(figures)} figures)")


if __name__ == "__main__":
    main()
