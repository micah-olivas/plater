"""Synthetic example figures for the docs guide.

Builds small, deterministic fake datasets in the same long-form shape `load()`
emits, runs the real plater plotting/analysis functions on them, and returns
each figure as a base64-encoded PNG (so `index.html` stays a single offline
file). Imported by build_docs.py; not part of the plater package.

Keep the data tiny and the seed fixed — these are illustrations, not tests.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # headless: no display needed at build time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plater as pl


# Shared MM ground truth for the titration-based figures.
_VMAX_ABS = 4.0e-3          # ΔAbs/s at saturation
_KM = 300.0                 # µM
_BEND_K = 1.8e-3            # gentle progress-curve curvature (1/s)
_A0 = 0.04                  # baseline absorbance
_S_LEVELS = [0, 78, 156, 312, 625, 1250]   # µM (0 = no-substrate baseline)
_REPLICATES = [1, 2, 3]
_TIMES = np.arange(0, 301, 15.0)           # s


def _png(fig) -> str:
    """Render a Matplotlib figure to a base64-encoded PNG data string.

    Rendered at a high DPI so the plots stay crisp on retina displays — the
    CSS scales them to the column width, so this only buys sharpness.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _progress_df(rng, notebook=None, rate_scale=1.0):
    """One substrate titration's worth of long-form progress-curve data.

    A(t) = A0 + (v/k)(1 − e^(−k t)) with v = Vmax·S/(Km+S). The S = 0 level has
    no substrate, so v = 0 and its trace stays flat at baseline — plotting by
    [S] renders it as the grey "0 (baseline)" entry, no separate control needed.
    `rate_scale` lets a second notebook differ slightly so the faceted folder
    example reads as two runs.
    """
    rows = []
    well_row = "ABCDEFGH"
    for r in _REPLICATES:
        for i, s in enumerate(_S_LEVELS):
            v = rate_scale * _VMAX_ABS * s / (_KM + s)
            well = f"{well_row[i]}{r}"
            rise = (v / _BEND_K) * (1.0 - np.exp(-_BEND_K * _TIMES))
            noise = rng.normal(0, 0.004, size=_TIMES.shape)
            abss = _A0 + rise + noise
            for t, a in zip(_TIMES, abss):
                row = {
                    "Well": well,
                    "Replicate": r,
                    "Substrate": "pNPA",
                    "S (µM)": s,
                    "E (nM)": 100,
                    "Time [s]": float(t),
                    "Absorbance": float(a),
                }
                if notebook is not None:
                    row["Notebook"] = notebook
                rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["signal_kind"] = "absorbance"
    return df


def example_load_df():
    """A representative long-form frame as `load()` would return it.

    Mirrors the `conditions` dict shown in the guide's loading snippet (wells
    A1/A2/B1, substrate pNPA) across a few timepoints, so the rendered table
    reads as the literal output of that code. The B1 no-enzyme control stays
    flat near baseline; A1/A2 climb per their [S].
    """
    rng = np.random.default_rng(0)
    wells = {
        "A1": (1, "pNPA", 1250, 100),
        "A2": (1, "pNPA", 625, 100),
        "B1": (1, "pNPA", 1250, 0),
    }
    rows = []
    for t in (0.0, 30.0, 60.0):
        for well, (rep, sub, s, e) in wells.items():
            v = _VMAX_ABS * s / (_KM + s) if e else 0.0
            a = _A0 + (v / _BEND_K) * (1.0 - np.exp(-_BEND_K * t)) + rng.normal(0, 0.003)
            rows.append({
                "Well": well,
                "Replicate": rep,
                "Substrate": sub,
                "S (µM)": s,
                "E (nM)": e,
                "Time [s]": t,
                "Absorbance": round(float(a), 3),
            })
    df = pd.DataFrame(rows)
    df.attrs["signal_kind"] = "absorbance"
    return df


def _standard_df(rng):
    """No-enzyme product standard: A vs [P], roughly linear (Beer–Lambert)."""
    eps, intercept = 2.5e-3, 0.03   # ΔAbs per µM
    rows = []
    for conc in [0, 25, 50, 100, 200, 400]:
        for r in _REPLICATES:
            a = intercept + eps * conc + rng.normal(0, 0.01)
            rows.append({"Replicate": r, "S (µM)": conc, "Absorbance": float(a)})
    df = pd.DataFrame(rows)
    df.attrs["signal_kind"] = "absorbance"
    return df


def _scan_df(rng):
    """Full spectrum vs time for a couple of wells: a product peak near 405 nm
    growing over time on a falling substrate shoulder."""
    waves = np.arange(320, 561, 10.0)
    times = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    rows = []
    for well, scale in (("A1", 1.0), ("A2", 0.55)):
        for t in times:
            grow = scale * (1.0 - np.exp(-_BEND_K * t))
            product = 0.9 * grow * np.exp(-0.5 * ((waves - 405) / 22.0) ** 2)
            substrate = 0.35 * np.exp(-0.5 * ((waves - 345) / 18.0) ** 2)
            a = 0.02 + substrate + product + rng.normal(0, 0.003, size=waves.shape)
            for w, av in zip(waves, a):
                rows.append({
                    "Well": well,
                    "Wavelength [nm]": float(w),
                    "Time [s]": t,
                    "Absorbance": float(av),
                })
    df = pd.DataFrame(rows)
    df.attrs["signal_kind"] = "absorbance"
    return df


def render_example_figures() -> dict[str, str]:
    """Return {guide_id: base64_png} for the guide blocks that get a figure."""
    figs: dict[str, str] = {}

    # (The loading block shows the returned DataFrame as a table, not a plot —
    # see example_load_df() + df_preview_html() in build_docs.py.)

    # --- folders: same layout across two notebooks, faceted ----------------
    rng = np.random.default_rng(11)
    folder_df = pd.concat(
        [_progress_df(rng, notebook="run1", rate_scale=1.0),
         _progress_df(rng, notebook="run2", rate_scale=0.78)],
        ignore_index=True,
    )
    folder_df.attrs["signal_kind"] = "absorbance"
    fig, *_ = pl.plot_progress_curves(
        folder_df, split_by="Notebook", color_by="S (µM)",
        collapse_replicates=False,
    )
    figs["folders"] = _png(fig)

    # --- kinetics-workflow: fits overlaid + the MM curve -------------------
    rng = np.random.default_rng(7)
    df = _progress_df(rng)
    rates = pl.compute_initial_rates(df, t_end=75)
    mm = pl.fit_michaelis_menten(rates)
    # Draw the raw point readings (not replicate-averaged mean lines) so the
    # figure matches the input data and the linear fits visibly pass through
    # the points over the [0, 75 s] window.
    fig, *_ = pl.plot_progress_curves(
        df, rates_df=rates, t_end_fit=75, color_by="S (µM)",
        collapse_replicates=False,
    )
    figs["kinetics-workflow"] = _png(fig)
    fig, *_ = pl.plot_initial_rates(rates, mm_params_df=mm)
    figs["kinetics-workflow-mm"] = _png(fig)

    # --- scans: spectra vs time (one well keeps it from getting too wide) ---
    rng = np.random.default_rng(3)
    fig, *_ = pl.plot_spectra(_scan_df(rng), wells="A1", n_timepoints=6)
    figs["scans"] = _png(fig)

    # --- standards: product standard curve + linear fit --------------------
    rng = np.random.default_rng(5)
    std = pl.compute_standard_curve(_standard_df(rng), conc_col="S (µM)")
    fig, *_ = pl.plot_standard_curves(std, conc_col="S (µM)")
    figs["standards"] = _png(fig)

    return figs


if __name__ == "__main__":
    out = render_example_figures()
    for k, v in out.items():
        print(f"{k}: {len(v)} base64 chars")
