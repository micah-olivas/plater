"""plater.plotting: all plotting routines (progress curves, rates, standards, spectra)."""

import re
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit

from . import _style
from ._common import DATA_COLUMNS, SIGNAL_RATE_UNIT_BY_KIND, _fourPL, _guess_signal_kind, _rate_col_label, _rate_col_name, _single_wavelength, _value_col
from .kinetics import fit_michaelis_menten, michaelis_menten, _build_exclusion_mask
from .rates import compute_initial_rates
from .standards import fit_standard_curve


def _color_dict_get(color_dict, level, default):
    """Look up `level` in a user `color_dict`, matching by value or string form.

    Keys are matched against `level` directly first, then by `str()` so a dict
    keyed by 'Hs1' still hits a level stored as a numpy string or number.
    Returns `default` when `color_dict` is None/empty or has no matching key.
    """
    if not color_dict:
        return default
    if level in color_dict:
        return color_dict[level]
    s = str(level)
    for k, v in color_dict.items():
        if str(k) == s:
            return v
    return default


def _match_category(target, categories):
    """Find `target` among `categories`, matching by value then string form.

    Mirrors the lookup convention in `_color_dict_get` so a reference passed as
    'Hs1' still hits a category stored as a numpy string or number. Returns the
    matching category (in its original form) or None.
    """
    if target in categories:
        return target
    s = str(target)
    for c in categories:
        if str(c) == s:
            return c
    return None


def _apply_color_dict(color_map, color_dict):
    """Override entries of an auto-built {level: color} map with `color_dict`.

    Levels absent from `color_dict` keep their auto-assigned (cmap) color.
    Mutates and returns `color_map`. No-op when `color_dict` is None/empty.
    """
    if not color_dict:
        return color_map
    for level in list(color_map):
        color_map[level] = _color_dict_get(color_dict, level, color_map[level])
    return color_map


def _fmt_level(col, val):
    """Format one categorical-level value, appending a unit when the column
    name carries one (e.g. col='E (nM)', val=20 -> '20 nM'; 'Hs1' -> 'Hs1')."""
    if pd.isna(val):
        return ''
    is_num = (isinstance(val, (int, float, np.integer, np.floating))
              and not isinstance(val, bool))
    if is_num:
        m = re.search(r'\(([^)]+)\)\s*$', str(col))
        s = f'{val:g}'
        return f'{s} {m.group(1)}' if m else s
    return str(val)


def _compound_label(cols, values, sep=', '):
    """Join several (col, value) levels into one readable label.

    e.g. cols=('Construct', 'E (nM)'), values=('Hs1', 20) -> 'Hs1, 20 nM'.
    """
    return sep.join(_fmt_level(c, v) for c, v in zip(cols, values))


def _add_compound_column(df, cols, sep=', '):
    """Return (df_copy, new_col_name) with a compound categorical column added.

    Used so `color_by` / `x_col` can accept a list of columns: each row's label
    is the per-column values formatted and joined (see `_compound_label`). The
    column is named by joining the source names with ' / ' so it reads as a
    sensible legend/axis title. `df.attrs` is preserved.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"compound level columns {missing} not in df: {list(df.columns)}"
        )
    name = ' / '.join(cols)
    out = df.copy()
    out[name] = [
        _compound_label(cols, vals, sep=sep)
        for vals in zip(*(df[c] for c in cols))
    ]
    return out, name


def _saturating_exp(x, a, d, k):
    """Rise-to-plateau exponential: a at x=0, asymptotes to d as x → ∞."""
    return a + (d - a) * (1.0 - np.exp(-k * np.asarray(x, dtype=float)))


def _fit_saturating_exp(x, y):
    """Fit y = a + (d − a)·(1 − exp(−k·x)). Returns dict with params, R², n."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a0 = float(y[np.argmin(x)]) if len(x) else 0.0
    d0 = float(np.nanmax(y))
    span = max(float(np.nanmax(x)), 1e-9)
    k0 = 3.0 / span  # ≈ reaches plateau over the observed x-range
    p0 = [a0, d0, k0]
    bounds = (
        [-np.inf, -np.inf, 1e-12],
        [np.inf, np.inf, np.inf],
    )
    popt, _ = curve_fit(_saturating_exp, x, y, p0=p0, bounds=bounds, maxfev=20000)
    yhat = _saturating_exp(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    a, d, k = popt
    return {'a': float(a), 'd': float(d), 'k': float(k), 'r2': r2, 'n': int(len(x))}


def _fit_4pl_curve(x, y):
    """Fit a 4-parameter logistic y = d + (a − d) / (1 + (x/c)^b).

    Returns dict with a (lower asymptote), b (Hill slope), c (EC50),
    d (upper asymptote), r², n.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nonzero = x[x > 0]
    if len(np.unique(nonzero)) < 3:
        raise ValueError(
            f'4PL fit needs ≥3 distinct non-zero x values; '
            f'got {len(np.unique(nonzero))}'
        )
    a0 = float(y[np.argmin(x)])
    d0 = float(np.nanmax(y))
    c0 = float(np.median(nonzero))
    b0 = 1.0
    p0 = [a0, b0, c0, d0]
    bounds = (
        [-np.inf, 0.1, 1e-6, -np.inf],
        [np.inf, 5.0, float(nonzero.max()) * 100, np.inf],
    )
    popt, _ = curve_fit(_fourPL, x, y, p0=p0, bounds=bounds, maxfev=20000)
    yhat = _fourPL(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    a, b, c, d = popt
    return {'a': float(a), 'b': float(b), 'c': float(c), 'd': float(d),
            'r2': r2, 'n': int(len(x))}


def plot_standard_curves(curves, conc_col='S (µM)', value_col='Absorbance',
                         label_col='Dataset', show_fit=True, max_conc=None,
                         fit='linear', color_dict=None,
                         wavelength=None, title=None, xscale=None, yscale=None,
                         xlim=None, ylim=None,
                         figsize=_style.DEFAULT_FIGSIZE_WIDE, dpi=_style.DEFAULT_DPI,
                         ax=None, transparent=False):
    """Overlay one or more standard curves, with optional per-dataset fit.

    `curves` may be:
      - dict {label: DataFrame} — each frame plotted as one curve
      - DataFrame — split by `label_col` if present, else plotted as one curve

    Frames may be raw (per-row) or pre-aggregated (with a `<value_col>_sem`
    column from `compute_standard_curve`). Returns (fig, ax, fits_df).

    Parameters
    ----------
    fit : 'linear' | 'exponential' | '4pl'
        Functional form for the per-dataset fit.
          - 'linear' (default): y = slope·x + intercept (uses
            `fit_standard_curve`). `max_conc` clips the points used.
          - 'exponential': rise-to-plateau y = a + (d − a)·(1 − exp(−k·x)).
            Returns (a, d, k, r²).
          - '4pl': 4-parameter logistic y = d + (a − d) / (1 + (x/c)^b),
            with a = lower asymptote, b = Hill slope, c = EC50,
            d = upper asymptote. Returns (a, b, c, d, r²).
    color_dict : dict | None
        Map of `label_col` value → color, overriding the auto-assigned `tab10`
        color per dataset (e.g. {'Hs1G': 'green', 'Hs1': 'grey'}). Labels
        absent from the dict keep their default palette color. Keys match by
        value or string form.
    wavelength : float | None
        Probe wavelength (nm) shown in the y-axis label. If None, auto-
        detected from a single-valued wavelength column in the input —
        either 'Wavelength (nm)' (the per-measurement tag added by `load()`
        for single-wavelength reads) or the scan-axis 'Wavelength [nm]'.
    xscale, yscale : str | None
        Axis scale passed to `ax.set_xscale` / `ax.set_yscale` (e.g. 'log',
        'symlog'). Default None leaves the matplotlib default ('linear'). With
        xscale='log', fit curves are drawn over a log-spaced x range and
        non-positive x points are dropped by matplotlib.
    xlim, ylim : tuple[float, float] | None
        Axis limits (min, max). Default None keeps the auto limits.
    transparent : bool
        If True, make the figure + axes background transparent (also on save).
        Default False (white). Ignored when `ax` is passed.
    """
    if fit not in ('linear', 'exponential', '4pl'):
        raise ValueError(
            f"fit={fit!r}; expected 'linear', 'exponential', or '4pl'"
        )
    auto_labeled = False
    if isinstance(curves, dict):
        frames = []
        for k, v in curves.items():
            f = v.copy()
            f[label_col] = k
            frames.append(f)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = curves.copy()
        if label_col not in df.columns:
            df[label_col] = 'data'
            auto_labeled = True

    if wavelength is None:
        wavelength = _single_wavelength(df)

    sem_col = f'{value_col}_sem'
    if sem_col not in df.columns:
        df = (
            df.dropna(subset=[conc_col, value_col, label_col])
              .groupby([label_col, conc_col])[value_col]
              .agg(['mean', 'sem', 'count'])
              .rename(columns={'mean': value_col, 'sem': sem_col, 'count': 'n'})
              .reset_index()
        )
        df[sem_col] = df[sem_col].fillna(0)

    labels = list(df[label_col].dropna().unique())
    cmap = plt.get_cmap('tab10')
    colors = {lbl: cmap(i % cmap.N) for i, lbl in enumerate(labels)}
    _apply_color_dict(colors, color_dict)

    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    unit = (conc_col.split('(')[-1].rstrip(')').strip()
            if '(' in conc_col else 'unit')

    fits = []
    fit_handles = []
    data_handles = []
    fit_top_y = None
    for lbl in labels:
        g = df[df[label_col] == lbl].sort_values(conc_col)
        c = colors[lbl]
        # Auto-named single series ('data') reads as noise in the fit label.
        lbl_prefix = '' if auto_labeled else f'{lbl} '
        face = _style._lighten(c)
        ax.errorbar(g[conc_col], g[value_col], yerr=g[sem_col],
                    fmt='o', markersize=6, color=c,
                    markerfacecolor=face,
                    markeredgecolor=_style.POINT_EDGE_COLOR,
                    markeredgewidth=_style.POINT_EDGE_WIDTH,
                    ecolor=c, elinewidth=0.9, capsize=2.5, capthick=0.9,
                    linestyle='none', zorder=3)
        data_handles.append(
            mpl.lines.Line2D([0], [0], marker='o', color=c, linestyle='',
                             markerfacecolor=face,
                             markeredgecolor=_style.POINT_EDGE_COLOR,
                             markeredgewidth=_style.POINT_EDGE_WIDTH,
                             markersize=6, label=str(lbl))
        )

        if not show_fit:
            continue

        x_max = max_conc if max_conc is not None else g[conc_col].max()
        if xscale == 'log':
            pos = g[conc_col][g[conc_col] > 0]
            x_lo = float(pos.min()) if len(pos) else x_max / 1e3
            x_fit = np.geomspace(x_lo, x_max, 200)
        else:
            x_fit = np.linspace(0, x_max, 200)

        if fit == 'linear':
            fit_df = fit_standard_curve(
                g, conc_col=conc_col, value_col=value_col, max_conc=max_conc,
            )
            if fit_df.empty:
                continue
            row = fit_df.iloc[0]
            y_fit = row['slope'] * x_fit + row['intercept']
            sign = '+' if row['intercept'] >= 0 else '−'
            fit_label = (
                f"{lbl_prefix}linear\n"
                f"  y = {row['slope']:.3e}·x  {sign} {abs(row['intercept']):.2f}\n"
                f"  R²={row['r2']:.3f}"
            )
            fits.append({label_col: lbl, 'fit': 'linear', **row.to_dict()})
        elif fit == 'exponential':
            sub = g[[conc_col, value_col]].dropna()
            if max_conc is not None:
                sub = sub[sub[conc_col] <= max_conc]
            if len(sub) < 3:
                continue
            try:
                row = _fit_saturating_exp(
                    sub[conc_col].to_numpy(float),
                    sub[value_col].to_numpy(float),
                )
            except Exception:
                continue
            y_fit = _saturating_exp(x_fit, row['a'], row['d'], row['k'])
            fit_label = (
                f"{lbl_prefix}exponential\n"
                f"  a={row['a']:.0f}, d={row['d']:.0f}\n"
                f"  k={row['k']:.2e} /{unit}\n"
                f"  R²={row['r2']:.3f}"
            )
            fits.append({label_col: lbl, 'fit': 'exponential', **row})
        else:  # '4pl'
            sub = g[[conc_col, value_col]].dropna()
            if max_conc is not None:
                sub = sub[sub[conc_col] <= max_conc]
            if len(sub) < 4:
                continue
            try:
                row = _fit_4pl_curve(
                    sub[conc_col].to_numpy(float),
                    sub[value_col].to_numpy(float),
                )
            except Exception:
                continue
            y_fit = _fourPL(x_fit, row['a'], row['b'], row['c'], row['d'])
            fit_label = (
                f"{lbl_prefix}4PL\n"
                f"  a={row['a']:.0f}, d={row['d']:.0f}\n"
                f"  b={row['b']:.2f}, c={row['c']:.1f} {unit}\n"
                f"  R²={row['r2']:.3f}"
            )
            fits.append({label_col: lbl, 'fit': '4pl', **row})

        ax.plot(x_fit, y_fit, color='black', lw=1.4, ls='-', zorder=2)
        fit_handles.append(
            mpl.lines.Line2D([0], [0], color='black', linestyle='-', lw=1.4,
                             label=fit_label)
        )

        fit_top_y = float(y_fit[-1]) if fit_top_y is None else max(fit_top_y, float(y_fit[-1]))

    if max_conc is not None and fit_top_y is not None:
        y_bot = ax.get_ylim()[0]
        ax.fill_between([0, max_conc], y_bot, fit_top_y,
                        color='0.92', zorder=0)
        ax.plot([max_conc, max_conc], [y_bot, fit_top_y],
                color='0.75', lw=0.8, zorder=1)
        ax.text(max_conc, (y_bot + fit_top_y) / 2,
                f'  fit range ≤ {max_conc:g} {unit}',
                color='0.45', fontsize=8, va='center', ha='left', zorder=1)

    if value_col == 'Absorbance' and wavelength is not None:
        ylabel = f'Absorbance ({wavelength:g} nm)'
    else:
        ylabel = value_col

    if title:
        ax.set_title(title)
    ax.set_xlabel(conc_col, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.margins(x=0.03, y=0.05)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    single_series = len(data_handles) <= 1
    if not single_series:
        data_leg = ax.legend(handles=data_handles, loc='upper left',
                             bbox_to_anchor=(1.02, 1.0), frameon=False,
                             fontsize=9)
        if fit_handles:
            ax.add_artist(data_leg)
            fit_anchor_y = max(0.05, 1.0 - 0.07 * (len(data_handles) + 0.5))
            ax.legend(handles=fit_handles, loc='upper left',
                      bbox_to_anchor=(1.02, fit_anchor_y), frameon=False,
                      fontsize=8, handlelength=2.4, labelspacing=0.9,
                      borderaxespad=0.0)
    elif fit_handles:
        # one dataset → skip the redundant data legend and drop fit params
        # into the emptiest corner of the axes
        all_xs = [x for h in data_handles for x in [0]]  # actual data already plotted
        xs = df[conc_col].to_numpy(float)
        ys = df[value_col].to_numpy(float)
        scores = _score_corners(ax, xs, ys, frac_x=0.40, frac_y=0.35)
        corner = min(scores, key=scores.get)
        text_lines = [(h.get_label(), h.get_color()) for h in fit_handles]
        _annotate_fit_params(ax, text_lines, corner,
                             line_h=0.22, fontsize=8.5)

    fig.tight_layout()
    if owns_fig:
        _style._apply_background(fig, ax, transparent)
    return fig, ax, pd.DataFrame(fits)


def _row_matches(row, filter_dict):
    """True if every (col, val) in filter_dict matches the Series row."""
    if not filter_dict:
        return False
    for k, v in filter_dict.items():
        if k not in row.index or row[k] != v:
            return False
    return True


def _resolve_annotate_modes(annotate_rates):
    """Normalize annotate_rates input → set of {'legend', 'lines'}."""
    if annotate_rates in (False, None):
        return set()
    if annotate_rates is True or annotate_rates == 'legend':
        return {'legend'}
    if annotate_rates == 'lines':
        return {'lines'}
    if annotate_rates == 'both':
        return {'legend', 'lines'}
    raise ValueError(
        f"annotate_rates={annotate_rates!r}; expected False, True, "
        "'legend', 'lines', or 'both'"
    )


def _collapse_replicates(df, condition_keys=None, value_col=None):
    """Mean ± SEM absorbance per (condition × Time [s]), pooling replicates.

    Condition columns default to every column in `df` except DATA_COLUMNS,
    'Well', and 'Replicate' — i.e. the per-experiment metadata.
    """
    if condition_keys is None:
        condition_keys = [
            c for c in df.columns
            if c not in DATA_COLUMNS and c not in ('Well', 'Replicate')
        ]
    if not condition_keys:
        raise ValueError(
            "no condition columns to group by; pass condition_keys= explicitly"
        )
    grp_cols = [*condition_keys, 'Time [s]']
    if value_col is None:
        value_col = _value_col(df)
    sem_col = f'{value_col}_sem'
    agg = (
        df.dropna(subset=['Time [s]', value_col])
          .groupby(grp_cols, as_index=False, dropna=False)[value_col]
          .agg(['mean', 'sem'])
          .rename(columns={'mean': value_col, 'sem': sem_col})
    )
    agg[sem_col] = agg[sem_col].fillna(0)
    if 'signal_kind' in getattr(df, 'attrs', {}):
        agg.attrs['signal_kind'] = df.attrs['signal_kind']
    return agg, condition_keys


_TIME_UNIT_ALIASES = {
    's': 's', 'sec': 's', 'secs': 's', 'second': 's', 'seconds': 's',
    'min': 'min', 'mins': 'min', 'm': 'min', 'minute': 'min', 'minutes': 'min',
    'h': 'h', 'hr': 'h', 'hrs': 'h', 'hour': 'h', 'hours': 'h',
}


_TIME_UNIT_SECONDS = {'s': 1.0, 'min': 60.0, 'h': 3600.0}


def _normalize_time_unit(time_unit):
    key = str(time_unit).strip().lower()
    if key not in _TIME_UNIT_ALIASES:
        raise ValueError(
            f"time_unit={time_unit!r}; expected one of "
            f"{sorted(_TIME_UNIT_ALIASES)}"
        )
    return _TIME_UNIT_ALIASES[key]


def plot_progress_curves(
    df,
    rates_df=None,
    show_rates=False,
    annotate_rates=False,
    color_by=None,
    color_dict=None,
    label_by=None,
    colorbar=False,
    split_by=None,
    hollow_where=None,
    t_start_fit=0,
    t_end_fit=100,
    window_by=None,
    wavelength=None,
    cmap_name=None,
    cmap_range=(0.40, 1.0),
    zero_baseline_color='0.4',
    zero_baseline_label='baseline',
    figsize=None,
    dpi=_style.DEFAULT_DPI,
    show_inset=False,
    collapse_replicates='auto',
    clip_y_to_non_hollow=False,
    ylim=None,
    xlim=None,
    xscale=None,
    yscale=None,
    sharex=None,
    sharey=None,
    time_unit='s',
    value_col=None,
    point_alpha=None,
    point_size=None,
    transparent=False,
):
    """A vs t per well (or per condition, if replicates are pooled).

    An optional inset (when `show_inset=True`) zooms the linear fit window
    [t_start_fit, t_end_fit]. Traces matching
    `hollow_where` are drawn with hollow markers / dashed lines and skipped
    from fit overlays.

    `t_start_fit` and `t_end_fit` may be scalars or dicts keyed by values
    of `window_by` (e.g. ``t_end_fit={'EnzA': 75, 'EnzB': 200}``,
    ``window_by='Enzyme'``) to use a different linear-fit window per
    reaction identity.

    When `split_by` is set, the function facets into one subplot per unique
    value (shared y-axis) and returns (fig, axes_list); the inset is not
    drawn in faceted mode.

    Parameters
    ----------
    time_unit : str
        X-axis units for the plot. Accepts 's' / 'sec' / 'second' /
        'seconds' for seconds, 'min' / 'm' / 'minute' / 'minutes' for
        minutes, and 'h' / 'hr' / 'hour' / 'hours' for hours
        (case-insensitive). `t_start_fit` / `t_end_fit` are always
        interpreted in seconds (to match `compute_initial_rates`); only the
        rendered axis changes.
    rates_df : DataFrame | None
        Pre-computed rates (from compute_initial_rates) to overlay as linear
        fits. In collapse mode, fits are recomputed from the collapsed mean
        trace so they match the displayed line.
    split_by : str | None
        Column name to facet by. One subplot per unique value, shared y.
    show_rates : bool
        If True and rates_df is None, compute_initial_rates is run internally
        with t_end=t_end_fit and the fits are overlaid.
    annotate_rates : False | True | 'legend' | 'lines' | 'both'
        Where to display the per-trace initial-rate value:
          - False : no annotation
          - True / 'legend' : append the rate (ΔAbs/s) to each legend entry,
            but only for `color_by` levels that map to a single curve; levels
            with multiple curves are left unannotated (one number would hide
            the spread)
          - 'lines' : place labels directly on each fit line, with collision-
            resistant positioning via `adjustText`
          - 'both' : both
        'lines'/'both' require the optional `adjustText` package.
    collapse_replicates : 'auto' | bool
        If True, average traces across Replicate per (condition × Time) and
        draw one mean line + shaded SEM band per condition. 'auto' enables
        this whenever a 'Replicate' column with >1 unique value is present.
    show_inset : bool | 'auto'
        If True, draw a side inset of the linear fit range. Defaults to
        False (off) since the inset often overlaps the main axes. Pass
        'auto' to draw it only when t_end_fit is meaningfully shorter than
        the full trace.
    clip_y_to_non_hollow : bool
        If True (and hollow_where is set), set y-axis limits from the
        non-hollow traces only — useful when controls are flat near A0 and
        would otherwise compress the active dynamic range.
    xlim : tuple[float, float] | None
        If set, the x-axis (time) limits (xmin, xmax) in the displayed
        `time_unit`, applied to the main axis (every facet when `split_by` is
        used). Default None keeps the auto limits.
    xscale, yscale : str | None
        Axis scale passed to `ax.set_xscale` / `ax.set_yscale` (e.g. 'log',
        'symlog'). Default None leaves the matplotlib default ('linear'). On a
        log scale non-positive values (e.g. the t=0 point) are dropped by
        matplotlib.
    sharex, sharey : bool | None
        Share the x/y axis across facet panels (only applies when `split_by`
        is set). Default None: sharex is off and sharey is auto-enabled when
        the panels' value ranges are within 3× of each other. Pass True/False
        to force the behavior.
    value_col : str | None
        Column to plot on the y-axis (and to fit when computing rates). If
        None, auto-detected from `df.attrs['signal_kind']` (typically RFU /
        Absorbance / Luminescence). Pass a name like '[NADH] (µM)' to plot
        a converted-concentration column from `apply_standard_curve`. The
        column header should end in '(<unit>)' so the y-axis label and rate
        units render correctly.
    wavelength : float | None
        Probe wavelength (nm) shown in the y-axis label. If None, auto-
        detected from a single-valued wavelength column in `df` — either
        'Wavelength (nm)' (the per-measurement tag added by `load()` for
        single-wavelength reads) or the scan-axis 'Wavelength [nm]'.
    color_by : str | None
        Column to color traces by. Numeric → sequential colormap (default
        'Blues'); categorical → qualitative ('tab10'). Default: 'S (µM)' if
        present, else 'Well' (per-well mode) / first condition key (collapsed).
    color_dict : dict | None
        Map of `color_by` level → color, overriding the auto-assigned color
        for matching levels (e.g. {'Hs1G': 'green', 'Hs1': 'grey'}). Most
        useful with a categorical `color_by`; levels absent from the dict keep
        their default palette color. Keys match by value or string form.
    hollow_where : dict | None
        Wells/conditions where every {col: val} pair matches are drawn with
        hollow markers (per-well mode) or a dashed line (collapsed mode), and
        skipped from fit overlays. Useful for highlighting controls
        (e.g. {'E (nM)': 0}).
    cmap_name : str | None
        Override the auto-selected colormap.
    zero_baseline_color : str | None
        For numeric `color_by` (e.g. 'S (µM)'), the value-0 level is drawn
        in this color and excluded from the colormap gradient — so the
        no-substrate baseline is visually distinct from the dose-response
        series. Pass None to keep 0 as part of the gradient.
    zero_baseline_label : str | None
        Suffix appended to the value-0 legend entry (e.g. "0 (baseline)").
        Pass None or '' to suppress.
    label_by : str | None
        Column whose value is drawn as a text label at the right end of each
        curve. One label per unique value (anchored at the curve with the
        largest end-x in that group). Labels use the curve color when
        `label_by == color_by`, else black; rendered with a white bbox for
        legibility and de-overlapped with `adjustText` when available.
    colorbar : bool | str
        If truthy (and `color_by` is numeric), replace the right-side
        categorical legend with a colorbar. True draws a discrete tile-per-
        level bar; 'continuous' draws a smooth gradient with a thinned set of
        value ticks — better when there are many levels. The value-0 baseline,
        when present, is shown as a separated grey tile below the gradient.
    point_alpha : float | None
        Override the scatter-point alpha in per-well mode. Defaults to 0.15
        for filled points and 0.22 for hollow points. Ignored in collapsed
        (replicate-averaged) mode, which draws lines + SEM bands.
    point_size : float | None
        Override the scatter-point area (matplotlib `s`) in per-well mode.
        Defaults to 6 (inset points scale to 2/3 of this). Ignored in
        collapsed (replicate-averaged) mode, which draws lines + SEM bands.
    transparent : bool
        If True, make the figure + axes background transparent (also on save).
        Default False (white). The fit-range inset stays opaque.

    `color_by` may be a list of columns (e.g. ['Construct', 'E (nM)']) to color
    by a compound level — each combination gets its own legend entry and color,
    so a category that spans several values of the second column (e.g. 'Hs1' at
    two enzyme concentrations) is split out (legend entries 'Hs1, 20 nM' /
    'Hs1, 1 nM'). `color_dict` then keys on those compound labels.
    """
    if isinstance(color_by, (list, tuple)):
        df, color_by = _add_compound_column(df, color_by)
    if split_by is not None:
        if split_by not in df.columns:
            raise KeyError(
                f"split_by={split_by!r} not in df: {list(df.columns)}"
            )
        levels = list(df[split_by].dropna().unique())
        if not levels:
            raise ValueError(f"split_by={split_by!r} has no non-null values")
        n = len(levels)

        if value_col is None:
            facet_value_col = _value_col(df)
        else:
            facet_value_col = value_col
        ranges = []
        for lvl in levels:
            vals = df.loc[df[split_by] == lvl, facet_value_col].dropna()
            if len(vals):
                ranges.append(float(vals.max() - vals.min()))
        if sharey is None:
            facet_sharey = bool(
                ranges
                and min(ranges) > 0
                and max(ranges) / min(ranges) <= 3.0
            )
        else:
            facet_sharey = sharey
        facet_sharex = bool(sharex) if sharex is not None else False

        if figsize is None:
            facet_figsize = (n * 3.0, 3.0)
        else:
            facet_figsize = figsize

        fig, axes = plt.subplots(
            1, n,
            figsize=facet_figsize,
            dpi=dpi, sharex=facet_sharex, sharey=facet_sharey,
            squeeze=False,
        )
        axes = list(axes[0])
        if color_by is not None and color_by in df.columns:
            global_levels_raw = df[color_by].dropna().unique()
            if pd.api.types.is_numeric_dtype(df[color_by]):
                global_color_levels = sorted(global_levels_raw)
            else:
                global_color_levels = list(global_levels_raw)
        else:
            global_color_levels = None
        for i_ax, (ax_i, lvl) in enumerate(zip(axes, levels)):
            sub = df[df[split_by] == lvl].copy()
            sub.attrs.update(df.attrs)
            if rates_df is not None and split_by in rates_df.columns:
                sub_rates = rates_df[rates_df[split_by] == lvl]
            else:
                sub_rates = rates_df
            is_last = i_ax == len(axes) - 1
            _plot_progress_curves_on_ax(
                ax_i, sub,
                rates_df=sub_rates,
                show_rates=show_rates,
                annotate_rates=annotate_rates,
                color_by=color_by,
                color_dict=color_dict,
                label_by=label_by,
                colorbar=(colorbar if is_last else False),
                hollow_where=hollow_where,
                t_start_fit=t_start_fit,
                t_end_fit=t_end_fit,
                window_by=window_by,
                wavelength=wavelength,
                cmap_name=cmap_name,
                cmap_range=cmap_range,
                zero_baseline_color=zero_baseline_color,
                zero_baseline_label=zero_baseline_label,
                show_inset=False,
                collapse_replicates=collapse_replicates,
                clip_y_to_non_hollow=clip_y_to_non_hollow,
                legend=is_last,
                ylim=ylim,
                xlim=xlim,
                xscale=xscale,
                yscale=yscale,
                time_unit=time_unit,
                value_col=value_col,
                point_alpha=point_alpha,
                point_size=point_size,
                color_levels=global_color_levels,
            )
            per_facet_in = facet_figsize[0] / n
            wrap_w = max(10, int(per_facet_in * 6))
            title = textwrap.fill(
                f'{split_by} = {lvl}', width=wrap_w,
                break_long_words=False, break_on_hyphens=False,
            )
            ax_i.set_title(title, fontsize=10)
        for ax_i in axes[1:]:
            ax_i.set_ylabel('')
        fig.tight_layout()
        _style._apply_background(fig, axes, transparent)
        return fig, axes

    fig, ax = plt.subplots(figsize=figsize or _style.DEFAULT_FIGSIZE_WIDE, dpi=dpi)
    axins = _plot_progress_curves_on_ax(
        ax, df,
        rates_df=rates_df,
        show_rates=show_rates,
        annotate_rates=annotate_rates,
        color_by=color_by,
        color_dict=color_dict,
        label_by=label_by,
        colorbar=colorbar,
        hollow_where=hollow_where,
        t_start_fit=t_start_fit,
        t_end_fit=t_end_fit,
        window_by=window_by,
        wavelength=wavelength,
        cmap_name=cmap_name,
        cmap_range=cmap_range,
        zero_baseline_color=zero_baseline_color,
        zero_baseline_label=zero_baseline_label,
        show_inset=show_inset,
        collapse_replicates=collapse_replicates,
        clip_y_to_non_hollow=clip_y_to_non_hollow,
        ylim=ylim,
        xlim=xlim,
        xscale=xscale,
        yscale=yscale,
        time_unit=time_unit,
        value_col=value_col,
        point_alpha=point_alpha,
        point_size=point_size,
    )
    fig.subplots_adjust(left=0.10, right=0.60, top=0.92, bottom=0.16)
    # Leave the inset axes opaque so its fit-range zoom reads clearly over the
    # main traces, even when the surrounding figure is transparent.
    _style._apply_background(fig, ax, transparent)
    return fig, ax, axins


_CORNER_ANCHORS = {
    'tl': dict(xy=(0.02, 0.98), ha='left',  va='top'),
    'tr': dict(xy=(0.98, 0.98), ha='right', va='top'),
    'bl': dict(xy=(0.02, 0.02), ha='left',  va='bottom'),
    'br': dict(xy=(0.98, 0.02), ha='right', va='bottom'),
}


def _score_corners(ax, xs, ys, extra_xy=None, frac=0.22,
                   frac_x=None, frac_y=None):
    """Return {corner_name: score} where lower is emptier.

    Counts plotted (`xs`, `ys`) plus any `extra_xy` highlight markers inside
    each corner box (size `frac_x` × `frac_y` of the axes; both default to
    `frac`). `extra_xy` is weighted 3× because the corresponding text label
    needs to avoid visible markers, not just the background scatter cloud.
    """
    if frac_x is None:
        frac_x = frac
    if frac_y is None:
        frac_y = frac
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xw = xlim[1] - xlim[0]
    yh = ylim[1] - ylim[0]
    bounds_by_corner = {
        'tl': (xlim[0], xlim[0] + frac_x * xw,
               ylim[1] - frac_y * yh, ylim[1]),
        'tr': (xlim[1] - frac_x * xw, xlim[1],
               ylim[1] - frac_y * yh, ylim[1]),
        'bl': (xlim[0], xlim[0] + frac_x * xw,
               ylim[0], ylim[0] + frac_y * yh),
        'br': (xlim[1] - frac_x * xw, xlim[1],
               ylim[0], ylim[0] + frac_y * yh),
    }
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    ep_x = np.array([p[0] for p in extra_xy], dtype=float) if extra_xy else np.array([])
    ep_y = np.array([p[1] for p in extra_xy], dtype=float) if extra_xy else np.array([])
    scores = {}
    for name, (x0, x1, y0, y1) in bounds_by_corner.items():
        scatter_hits = int(((xs >= x0) & (xs <= x1)
                            & (ys >= y0) & (ys <= y1)).sum())
        fit_hits = (
            int(((ep_x >= x0) & (ep_x <= x1)
                 & (ep_y >= y0) & (ep_y <= y1)).sum())
            if ep_x.size else 0
        )
        scores[name] = scatter_hits + 3 * fit_hits
    return scores


def _annotate_fit_params(ax, text_lines, corner_name, line_h=0.09, fontsize=7.5):
    """Annotate fit-parameter text in a chosen corner of `ax`.

    `text_lines` is a list of `(text, color)` tuples (empty `text` is a spacer).
    The first non-empty line always reads at the top of the block, regardless
    of which corner is anchored.
    """
    anchor = _CORNER_ANCHORS[corner_name]
    x0, y0 = anchor['xy']
    if anchor['va'] == 'top':
        offsets = [-i * line_h for i in range(len(text_lines))]
    else:
        n_visible = sum(1 for t, _ in text_lines if t)
        offsets = [(n_visible - 1 - i) * line_h for i in range(len(text_lines))]
    for (txt, color), dy in zip(text_lines, offsets):
        if not txt:
            continue
        ax.annotate(
            txt,
            xy=(x0, y0 + dy),
            xycoords='axes fraction',
            ha=anchor['ha'], va=anchor['va'],
            fontsize=fontsize, color=color,
        )


def _nice_tick(v):
    """Round up to one significant digit for clean tick labels."""
    if v <= 0 or not np.isfinite(v):
        return 0.0
    exp = int(np.floor(np.log10(v)))
    base = 10.0 ** exp
    return float(np.ceil(v / base) * base)


def _fmt_sig(x, sig=3):
    """Format to `sig` significant figures, preferring fixed-point notation.

    Keeps fit-parameter labels readable across the usual magnitudes (e.g. 1140,
    5.84, 0.0208) instead of ``"%.3g"``, which flips to scientific notation at
    ~1e3. Values outside [1e-3, 1e6) fall back to scientific so extreme numbers
    stay compact rather than dragging a long run of zeros.
    """
    if not np.isfinite(x):
        return str(x)
    if x == 0:
        return '0'
    exp = int(np.floor(np.log10(abs(x))))
    if exp < -3 or exp >= 6:
        return f'{x:.{sig - 1}e}'
    return f'{x:.{max(sig - 1 - exp, 0)}f}'


class _TopLineHandler(mpl.legend_handler.HandlerLine2D):
    """Legend handler that lifts the marker to the first (top) text line.

    matplotlib centers each legend handle against the full height of its label,
    so for a multi-line label (group name + indented fit params) the marker
    lands beside the middle line. This shifts it up to sit on the first line.
    """

    def __init__(self, n_lines=1, linespacing=1.39, **kw):
        super().__init__(**kw)
        self._n_lines = n_lines
        self._linespacing = linespacing

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                        width, height, fontsize, trans):
        artists = super().create_artists(
            legend, orig_handle, xdescent, ydescent,
            width, height, fontsize, trans,
        )
        if self._n_lines > 1:
            line_px = fontsize * self._linespacing * legend.figure.dpi / 72.0
            dy = (self._n_lines - 1) / 2.0 * line_px
            shift = mpl.transforms.Affine2D().translate(0.0, dy)
            for a in artists:
                a.set_transform(a.get_transform() + shift)
        return artists


_TIME_STEP_CHOICES = {
    'min': (0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 240),
    'h':   (0.25, 0.5, 1, 2, 3, 6, 12, 24, 48),
}


def _apply_time_ticks(ax, unit, max_ticks=8):
    """Tick the (seconds-valued) x-axis at unit-friendly multiples and
    label the tick text in `unit` (one of 'min' | 'h')."""
    seconds_per_unit = _TIME_UNIT_SECONDS[unit]
    choices = _TIME_STEP_CHOICES[unit]
    x0, x1 = ax.get_xlim()
    span = max(x1 - x0, 1.0) / seconds_per_unit
    step = choices[-1]
    for s in choices:
        if span / s <= max_ticks:
            step = s
            break
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(step * seconds_per_unit))
    ax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _pos: f'{v / seconds_per_unit:g}')
    )


def _annotate_curve_labels(
    ax, df_plot, group_keys, value_col, *,
    label_by, color_by, color_map, is_numeric_label,
):
    """Place one text label per unique `label_by` value at the right end of
    its curve(s). Labels are colored to match the curve when label_by ==
    color_by, drawn over a white bbox for legibility, and de-overlapped with
    `adjustText` when available."""
    fmt = (lambda v: f'{v:g}') if is_numeric_label else (lambda v: str(v))

    texts = []
    for lbl_val, sub in df_plot.groupby(label_by, dropna=True):
        ends_x, ends_y, ends_color_val = [], [], []
        for _, group in sub.groupby(group_keys):
            g = group.dropna(subset=['Time [s]', value_col]) \
                     .sort_values('Time [s]')
            if g.empty:
                continue
            ends_x.append(float(g.iloc[-1]['Time [s]']))
            ends_y.append(float(g.iloc[-1][value_col]))
            ends_color_val.append(group[color_by].iloc[0]
                                  if color_by in group.columns else None)
        if not ends_x:
            continue
        i_anchor = int(np.argmax(ends_x))
        x = ends_x[i_anchor]
        y = ends_y[i_anchor]
        if color_by == label_by:
            c = color_map.get(lbl_val, 'black')
        else:
            c_val = ends_color_val[i_anchor]
            c = color_map.get(c_val, 'black') if c_val is not None else 'black'
        texts.append(ax.text(
            x, y, fmt(lbl_val),
            color=c, fontsize=8, fontweight='semibold',
            ha='left', va='center',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.85, pad=1.4, boxstyle='round,pad=0.18'),
            zorder=15, clip_on=False,
        ))

    if not texts:
        return

    try:
        from adjustText import adjust_text
    except ImportError:
        return  # leave labels at default positions

    try:
        adjust_text(
            texts, ax=ax,
            only_move={'text': 'y', 'static': 'y', 'explode': 'y'},
            expand=(1.05, 1.3),
            arrowprops=dict(arrowstyle='-', color='0.6', lw=0.4),
        )
    except TypeError:
        adjust_text(
            texts, ax=ax,
            expand=(1.05, 1.3),
            arrowprops=dict(arrowstyle='-', color='0.6', lw=0.4),
        )


def _draw_discrete_colorbar(
    ax, levels, color_map, *,
    title=None,
    has_zero_baseline=False,
    zero_baseline_label='baseline',
    fmt_label=str,
    bbox=(1.05, 0.05, 0.05, 0.50),
    alpha=None,
):
    """Draw a discrete colorbar in an inset axis to the right of `ax`.

    Numeric levels are stacked low→high from bottom to top. When
    `has_zero_baseline=True`, the 0 level is rendered as a separated grey
    tile beneath a small gap so the no-substrate baseline reads as distinct
    from the dose-response gradient. `alpha` fades the tile fills to match
    the plotted points' opacity (None leaves them opaque).
    """
    cax = ax.inset_axes(bbox)
    cax.set_xticks([])
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)

    grad_levels = sorted(
        v for v in levels if not (has_zero_baseline and v == 0)
    )
    has_zero = has_zero_baseline and any(v == 0 for v in levels)

    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)

    n_grad = len(grad_levels)
    gap = 0.06 if has_zero else 0.0
    n_tiles = n_grad + (1 if has_zero else 0)
    tile_h = (1.0 - gap) / n_tiles if n_tiles else 0.0
    zero_h = tile_h if has_zero else 0.0

    top_y = 0.0
    for i, v in enumerate(grad_levels):
        y0 = zero_h + gap + i * tile_h
        cax.add_patch(mpl.patches.Rectangle(
            (0, y0), 1, tile_h,
            facecolor=mpl.colors.to_rgba(color_map[v], alpha),
            edgecolor='white', lw=0.4,
        ))
        cax.text(1.35, y0 + tile_h / 2, fmt_label(v),
                 ha='left', va='center', fontsize=8,
                 clip_on=False)
        top_y = y0 + tile_h

    if has_zero:
        cax.add_patch(mpl.patches.Rectangle(
            (0, 0), 1, zero_h,
            facecolor=mpl.colors.to_rgba(color_map.get(0, '0.4'), alpha),
            edgecolor='white', lw=0.4,
        ))
        lbl = fmt_label(0)
        if zero_baseline_label:
            lbl = f'{lbl} ({zero_baseline_label})'
        cax.text(1.35, zero_h / 2, lbl,
                 ha='left', va='center', fontsize=8,
                 clip_on=False)
        top_y = max(top_y, zero_h)

    if title:
        cax.text(0, top_y + 0.04, title,
                 ha='left', va='bottom',
                 fontsize=10, transform=cax.transAxes,
                 clip_on=False)


def _draw_continuous_colorbar(
    ax, levels, color_map, *,
    cmap, cmap_range=(0.40, 1.0),
    title=None,
    has_zero_baseline=False,
    zero_baseline_label='baseline',
    fmt_label=str,
    bbox=(1.05, 0.05, 0.05, 0.50),
    max_ticks=8,
    alpha=None,
):
    """Draw a continuous gradient colorbar in an inset axis right of `ax`.

    The gradient is the same `cmap` slice (`cmap_range`) used to color the
    traces, so colors match exactly; numeric levels are positioned by rank
    (matching how trace colors are assigned) and only a thinned subset of
    `max_ticks` labels is shown to avoid crowding. A 0 level (when
    `has_zero_baseline`) is drawn as a separated grey tile below the gradient.
    `alpha` fades the gradient and zero tile to match the plotted points'
    opacity (None leaves them opaque).
    """
    cax = ax.inset_axes(bbox)
    cax.set_xticks([])
    cax.set_yticks([])
    for s in cax.spines.values():
        s.set_visible(False)

    grad_levels = sorted(
        v for v in levels if not (has_zero_baseline and v == 0)
    )
    has_zero = has_zero_baseline and any(v == 0 for v in levels)
    lo, hi = cmap_range
    n = len(grad_levels)

    cax.set_xlim(0, 1)
    cax.set_ylim(0, 1)

    gap = 0.06 if has_zero else 0.0
    # Reserve one rank-slot worth of height for the zero tile so it visually
    # matches the gradient's per-level density.
    zero_h = (1.0 - gap) / (n + 1) if (has_zero and n) else (0.5 if has_zero else 0.0)
    y_base = zero_h + gap

    grad = cmap(np.linspace(lo, hi, 256)).reshape(256, 1, 4)
    cax.imshow(
        grad, origin='lower', aspect='auto',
        extent=(0, 1, y_base, 1.0), zorder=1, alpha=alpha,
    )
    cax.add_patch(mpl.patches.Rectangle(
        (0, y_base), 1, 1.0 - y_base,
        fill=False, edgecolor='white', lw=0.4, zorder=2,
    ))

    # Label "nice" round values (10, 30, 100 … or 0.2, 0.4 …) rather than the
    # raw dilution-series numbers, placing each by interpolating its rank
    # position. Colors are assigned by rank, which for a ~geometric series is
    # near-linear in log(value), so we interpolate in log space when the range
    # spans more than ~1 decade. Nice ticks land inside the range, so the bar's
    # extreme ends are not labeled right at the edge.
    vmin, vmax = float(grad_levels[0]), float(grad_levels[-1])
    positions = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    log_scale = vmin > 0 and (vmax / vmin) > 20

    if n <= 1:
        ticks = [(grad_levels[0], 0.5)]
    else:
        if log_scale:
            cand = mpl.ticker.LogLocator(
                base=10, subs=(1.0, 2.0, 5.0), numticks=max_ticks + 2,
            ).tick_values(vmin, vmax)
        else:
            cand = mpl.ticker.MaxNLocator(
                nbins=max_ticks - 1, steps=[1, 2, 2.5, 5, 10],
            ).tick_values(vmin, vmax)
        cand = [float(v) for v in cand if vmin <= v <= vmax]
        if not cand:
            cand = [vmin, vmax]
        if log_scale:
            fracs = np.interp(np.log(cand), np.log(grad_levels), positions)
        else:
            fracs = np.interp(cand, grad_levels, positions)
        ticks = sorted(zip(cand, fracs), key=lambda t: t[1])

    shown = []
    for v, frac in ticks:
        if shown and abs(frac - shown[-1][1]) < 0.05:  # avoid crowding
            continue
        shown.append((v, frac))
    for v, frac in shown:
        y = y_base + (1.0 - y_base) * frac
        cax.plot([1.0, 1.18], [y, y], color='0.3', lw=0.6, clip_on=False)
        cax.text(1.35, y, fmt_label(v),
                 ha='left', va='center', fontsize=8, clip_on=False)

    if has_zero:
        cax.add_patch(mpl.patches.Rectangle(
            (0, 0), 1, zero_h,
            facecolor=mpl.colors.to_rgba(color_map.get(0, '0.4'), alpha),
            edgecolor='white', lw=0.4,
        ))
        lbl = fmt_label(0)
        if zero_baseline_label:
            lbl = f'{lbl} ({zero_baseline_label})'
        cax.text(1.35, zero_h / 2, lbl,
                 ha='left', va='center', fontsize=8, clip_on=False)

    if title:
        cax.text(0, 1.04, title,
                 ha='left', va='bottom',
                 fontsize=10, transform=cax.transAxes, clip_on=False)


def _window_min(t_start_fit):
    if isinstance(t_start_fit, dict):
        return float(min(t_start_fit.values())) if t_start_fit else 0.0
    return float(t_start_fit)


def _window_max(t_end_fit):
    if isinstance(t_end_fit, dict):
        return float(max(t_end_fit.values())) if t_end_fit else 0.0
    return float(t_end_fit)


def _window_for_row(t_start_fit, t_end_fit, window_by, row):
    """Resolve scalar (t_start, t_end) for one plotted group given a row
    (with the `window_by` column set). Unknown dict keys fall through to
    the dict's max/min so the trace still plots — compute_initial_rates is
    the place that errors on missing keys."""
    if isinstance(t_start_fit, dict):
        if window_by is None or window_by not in row.index:
            ts = _window_min(t_start_fit)
        else:
            ts = float(t_start_fit.get(row[window_by], _window_min(t_start_fit)))
    else:
        ts = float(t_start_fit)
    if isinstance(t_end_fit, dict):
        if window_by is None or window_by not in row.index:
            te = _window_max(t_end_fit)
        else:
            te = float(t_end_fit.get(row[window_by], _window_max(t_end_fit)))
    else:
        te = float(t_end_fit)
    return ts, te


def _plot_progress_curves_on_ax(
    ax, df, *,
    rates_df=None,
    show_rates=False,
    annotate_rates=False,
    color_by=None,
    color_dict=None,
    label_by=None,
    colorbar=False,
    hollow_where=None,
    t_start_fit=0,
    t_end_fit=100,
    window_by=None,
    wavelength=None,
    cmap_name=None,
    cmap_range=(0.40, 1.0),
    zero_baseline_color='0.4',
    zero_baseline_label='baseline',
    show_inset=False,
    collapse_replicates='auto',
    clip_y_to_non_hollow=False,
    legend=True,
    ylim=None,
    xlim=None,
    xscale=None,
    yscale=None,
    time_unit='s',
    value_col=None,
    point_alpha=None,
    point_size=None,
    color_levels=None,
):
    """Render the progress-curves view onto an existing matplotlib axis.

    Internal helper for `plot_progress_curves`. Returns the inset axis (or
    None when `show_inset` is False) so the public wrapper can adjust it.
    """
    time_unit = _normalize_time_unit(time_unit)
    seconds_per_unit = _TIME_UNIT_SECONDS[time_unit]
    x_label = f'Time ({time_unit})'

    if (isinstance(t_start_fit, dict) or isinstance(t_end_fit, dict)) \
            and window_by is None:
        raise ValueError(
            "dict t_start_fit/t_end_fit requires window_by= to name the "
            "column whose values match the dict keys"
        )

    if collapse_replicates == 'auto':
        collapse_replicates = (
            'Replicate' in df.columns
            and df['Replicate'].nunique(dropna=True) > 1
        )

    explicit_value_col = value_col is not None
    if value_col is None:
        value_col = _value_col(df)
    elif value_col not in df.columns:
        raise KeyError(
            f"value_col={value_col!r} not in df columns: {list(df.columns)}"
        )
    sem_col = f'{value_col}_sem'

    if collapse_replicates:
        df_plot, condition_keys = _collapse_replicates(df, value_col=value_col)
        group_keys = condition_keys
        if show_rates or rates_df is not None:
            rates_df = compute_initial_rates(
                df_plot.drop(columns=sem_col, errors='ignore'),
                t_start=t_start_fit, t_end=t_end_fit,
                group_by=condition_keys, drop_no_enzyme=False,
                window_by=window_by,
                value_col=value_col if explicit_value_col else None,
            )
    else:
        df_plot = df
        # 'Well' alone doesn't identify a trace once several files are stacked
        # (load_folder reuses well IDs across notebooks), so fold the source
        # column into the grouping when present — otherwise same-well traces
        # from different notebooks merge into one corrupted curve.
        nb_col = df.attrs.get('notebook_col') if hasattr(df, 'attrs') else None
        if (nb_col and nb_col in df_plot.columns
                and df_plot[nb_col].nunique(dropna=True) > 1):
            group_keys = [nb_col, 'Well']
        else:
            group_keys = 'Well'
        if show_rates and rates_df is None:
            rates_df = compute_initial_rates(
                df, t_start=t_start_fit, t_end=t_end_fit,
                drop_no_enzyme=False,
                window_by=window_by,
                value_col=value_col if explicit_value_col else None,
            )

    def _default_color_by():
        if 'S (µM)' in df_plot.columns:
            return 'S (µM)'
        if not collapse_replicates and 'Well' in df_plot.columns:
            return 'Well'
        gk = group_keys[0] if isinstance(group_keys, list) else group_keys
        return gk if gk in df_plot.columns else None

    if color_by is not None and color_by not in df_plot.columns:
        fallback = _default_color_by()
        warnings.warn(
            f"color_by={color_by!r} not in df columns "
            f"{list(df_plot.columns)}; falling back to {fallback!r}.",
            stacklevel=2,
        )
        color_by = fallback
    elif color_by is None:
        color_by = _default_color_by()

    if color_by is None or color_by not in df_plot.columns:
        df_plot = df_plot.assign(_all='all')
        color_by = '_all'

    is_numeric = pd.api.types.is_numeric_dtype(df_plot[color_by])
    if color_levels is not None:
        levels_raw = list(color_levels)
    else:
        levels_raw = df_plot[color_by].dropna().unique()
    if is_numeric:
        levels = sorted(levels_raw)
        cmap = plt.get_cmap(cmap_name or 'Blues')
        lo, hi = cmap_range
        # Exclude any zero level from the gradient so it can be drawn in a
        # distinct baseline color (and the gradient spans the active range).
        has_zero_baseline = (
            zero_baseline_color is not None and any(v == 0 for v in levels)
        )
        grad_levels = (
            [v for v in levels if v != 0] if has_zero_baseline else levels
        )
        if len(grad_levels) == 1:
            color_map = {grad_levels[0]: cmap((lo + hi) / 2)}
        else:
            color_map = {
                v: cmap(lo + (hi - lo) * i / (len(grad_levels) - 1))
                for i, v in enumerate(grad_levels)
            }
        if has_zero_baseline:
            for v in levels:
                if v == 0:
                    color_map[v] = zero_baseline_color
        fmt_label = lambda v: f'{v:g}'
    else:
        levels = list(levels_raw)
        cmap = plt.get_cmap(cmap_name or 'tab10')
        color_map = {v: cmap(i % cmap.N) for i, v in enumerate(levels)}
        fmt_label = str

    _apply_color_dict(color_map, color_dict)

    t_end_fit_max = _window_max(t_end_fit)
    if show_inset == 'auto':
        t_max = df_plot['Time [s]'].dropna().max()
        show_inset = bool(pd.notna(t_max) and t_end_fit_max < 0.6 * t_max)

    axins = ax.inset_axes([1.1, 0.55, 0.45, 0.42]) if show_inset else None

    has_hollow = False
    for _, group in df_plot.groupby(group_keys):
        g = group.dropna(subset=['Time [s]', value_col]).sort_values('Time [s]')
        if g.empty:
            continue
        c_val = group[color_by].iloc[0]
        c = color_map.get(c_val, 'gray')

        hollow = bool(hollow_where) and _row_matches(group.iloc[0], hollow_where)
        has_hollow = has_hollow or hollow

        _, te_g = _window_for_row(t_start_fit, t_end_fit, window_by, group.iloc[0])

        if collapse_replicates:
            t_arr = g['Time [s]'].to_numpy(float)
            a_arr = g[value_col].to_numpy(float)
            sem_arr = g[sem_col].to_numpy(float)
            line_kw = dict(color=c, lw=1.4,
                           ls='--' if hollow else '-',
                           alpha=0.6 if hollow else 0.95)
            band_kw = dict(color=c, alpha=0.12 if hollow else 0.18, lw=0)
            ax.plot(t_arr, a_arr, **line_kw)
            ax.fill_between(t_arr, a_arr - sem_arr, a_arr + sem_arr, **band_kw)
            if axins is not None:
                m = t_arr <= te_g
                axins.plot(t_arr[m], a_arr[m], **line_kw)
                axins.fill_between(t_arr[m],
                                   a_arr[m] - sem_arr[m],
                                   a_arr[m] + sem_arr[m], **band_kw)
        else:
            a_hollow = 0.22 if point_alpha is None else point_alpha
            a_solid = 0.15 if point_alpha is None else point_alpha
            s_main = 6 if point_size is None else point_size
            # Inset points are emphasized: larger than the main trace and
            # ringed in black so the fit-range zoom reads clearly.
            s_ins = max(s_main * 3, 18)
            if hollow:
                main_kw = dict(facecolors='none', edgecolors=c, s=s_main,
                               alpha=a_hollow, linewidths=0.6)
                ins_kw = dict(facecolors=c, edgecolors='black', s=s_ins,
                              alpha=a_hollow, linewidths=0.6, zorder=12)
            else:
                main_kw = dict(color=c, s=s_main, alpha=a_solid)
                ins_kw = dict(facecolors=c, edgecolors='black', s=s_ins,
                              alpha=1.0, linewidths=0.6, zorder=12)

            ax.scatter(g['Time [s]'], g[value_col], **main_kw)
            if axins is not None:
                g_in = g[g['Time [s]'] <= te_g]
                axins.scatter(g_in['Time [s]'], g_in[value_col], **ins_kw)

    if clip_y_to_non_hollow and hollow_where:
        active_vals = []
        for _, group in df_plot.groupby(group_keys):
            if _row_matches(group.iloc[0], hollow_where):
                continue
            active_vals.append(group[value_col].dropna())
        if active_vals:
            v = pd.concat(active_vals)
            margin = 0.05 * (v.max() - v.min() or 1.0)
            ax.set_ylim(v.min() - margin, v.max() + margin)
            if axins is not None:
                axins.set_ylim(v.min() - margin, v.max() + margin)
    elif ylim is None:
        # Auto-detect ylim from the actual plotted data (including SEM bands
        # in collapsed mode). matplotlib's autoscale can clip data with
        # sticky-edges or sharey across facets — derive from data directly.
        y_vals = df_plot[value_col].dropna()
        if collapse_replicates and sem_col in df_plot.columns:
            sem_vals = df_plot[sem_col].fillna(0)
            lo_series = df_plot[value_col] - sem_vals
            hi_series = df_plot[value_col] + sem_vals
            y_lo = pd.concat([y_vals, lo_series.dropna()]).min()
            y_hi = pd.concat([y_vals, hi_series.dropna()]).max()
        elif len(y_vals):
            y_lo = y_vals.min()
            y_hi = y_vals.max()
        else:
            y_lo = y_hi = None
        if y_lo is not None and np.isfinite(y_lo) and np.isfinite(y_hi):
            span = float(y_hi - y_lo) or max(abs(float(y_hi)), 1.0)
            margin = 0.08 * span
            # Extra top headroom when the fit-window label is overlaid: it's pinned
            # to the top-left corner, so reserve a clear band (~20% of the axis)
            # above the data for it.
            top_margin = (0.28 if rates_df is not None and len(rates_df)
                          else 0.08) * span
            new_lo = float(y_lo) - margin
            new_hi = float(y_hi) + top_margin
            # Union with existing ylim so sharey facets accumulate range
            # across subplots (each call only sees its own data).
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(min(new_lo, cur_lo), max(new_hi, cur_hi))
            if axins is not None:
                cur_lo_i, cur_hi_i = axins.get_ylim()
                axins.set_ylim(min(new_lo, cur_lo_i), max(new_hi, cur_hi_i))

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)

    line_label_data = []  # for adjustText placement
    endpoint_xy = []
    fit_windows_by_level = {}  # color_by value -> set of (ts, te) seen
    distinct_windows = set()
    legend_windows = False     # set in the annotation block below
    fit_window_by_level = {}
    if rates_df is not None and len(rates_df):
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        for _, row in rates_df.iterrows():
            if hollow_where and _row_matches(row, hollow_where):
                continue
            m, b = row['slope'], row['intercept']
            if 't_start_fit' in row and 't_end_fit' in row \
                    and pd.notna(row['t_start_fit']) and pd.notna(row['t_end_fit']):
                ts_row = float(row['t_start_fit'])
                te_row = float(row['t_end_fit'])
            else:
                ts_row, te_row = _window_for_row(
                    t_start_fit, t_end_fit, window_by, row,
                )
            distinct_windows.add((round(ts_row, 6), round(te_row, 6)))
            if color_by is not None and color_by in row \
                    and pd.notna(row[color_by]):
                fit_windows_by_level.setdefault(row[color_by], set()).add(
                    (ts_row, te_row)
                )
            t_fit_main = np.array([ts_row, te_row])
            y_fit_main = m * t_fit_main + b
            ax.plot(t_fit_main, y_fit_main,
                    color='k', lw=1.0, ls='--', alpha=0.8, zorder=10)
            ax.scatter(t_fit_main, y_fit_main,
                       s=12, c='k', marker='o',
                       linewidths=0, zorder=11)
            endpoint_xy.append((ts_row, float(y_fit_main[0])))
            endpoint_xy.append((te_row, float(y_fit_main[-1])))
            if axins is not None:
                axins.plot(t_fit_main, y_fit_main,
                           color='k', lw=1.0, ls='--', alpha=0.9, zorder=10)
                axins.scatter(t_fit_main, y_fit_main,
                              s=8, c='k', marker='o',
                              linewidths=0, zorder=11)
            line_label_data.append((row, m, b, te_row))

        if endpoint_xy:
            multi_window = len(distinct_windows) > 1
            # Levels mapping to exactly one window can carry their range in the
            # legend; drop any with conflicting windows (window_by != color_by).
            fit_window_by_level = {
                lvl: next(iter(ws))
                for lvl, ws in fit_windows_by_level.items()
                if len(ws) == 1
            }
            # Per-entry legend windows only work when a categorical legend is
            # drawn (a colorbar has no per-level text) and every level maps cleanly.
            legend_windows = (
                multi_window
                and legend
                and not (bool(colorbar) and is_numeric)
                and bool(fit_window_by_level)
                and len(fit_window_by_level) == len(fit_windows_by_level)
            )
            corner = dict(_CORNER_ANCHORS['tl'])
            if legend_windows:
                # Reported per-entry in the legend (see _legend_label); a single
                # min–max corner line would be meaningless across windows.
                pass
            elif multi_window:
                # Can't map to legend entries. Group levels by window so shared
                # windows collapse: state the most common one as the default and
                # list only the exceptions (avoids one cluttered line per level).
                def _fmt_win(ts, te):
                    return (f'{ts / seconds_per_unit:g}–'
                            f'{te / seconds_per_unit:g} {time_unit}')

                if fit_window_by_level:
                    windows_to_levels = {}
                    for lvl, w in fit_window_by_level.items():
                        windows_to_levels.setdefault(w, []).append(lvl)
                    if len(windows_to_levels) == 1:
                        ((ts, te),) = windows_to_levels
                        lines = [f'Fit window: {_fmt_win(ts, te)}']
                    else:
                        # Most-used window is the default; list the rest below.
                        ordered = sorted(windows_to_levels.items(),
                                         key=lambda kv: (-len(kv[1]), kv[0]))
                        (ts, te), _ = ordered[0]
                        lines = [f'Fit window: {_fmt_win(ts, te)} (default)']
                        for (ts, te), lvls in ordered[1:]:
                            lvl_str = ', '.join(fmt_label(l) for l in lvls)
                            lines.append(f'  {lvl_str}: {_fmt_win(ts, te)}')
                else:
                    # No clean per-level mapping (e.g. no color_by) — just list
                    # the distinct windows on one line.
                    lines = ['Fit windows: ' + ', '.join(
                        _fmt_win(ts, te) for ts, te in sorted(distinct_windows)
                    )]
                ax.annotate(
                    '\n'.join(lines),
                    xy=corner['xy'],
                    xycoords='axes fraction',
                    ha=corner['ha'], va=corner['va'],
                    fontsize=8, color='k',
                )
            else:
                # Single shared window — one corner line.
                xs_drawn = [p[0] for p in endpoint_xy]
                t0_disp = min(xs_drawn) / seconds_per_unit
                t1_disp = max(xs_drawn) / seconds_per_unit
                ax.annotate(
                    f"Fit window: {t0_disp:.1f}–{t1_disp:.1f} {time_unit}",
                    xy=corner['xy'],
                    xycoords='axes fraction',
                    ha=corner['ha'], va=corner['va'],
                    fontsize=8, color='k',
                )

    if label_by is not None:
        if label_by not in df_plot.columns:
            warnings.warn(
                f"label_by={label_by!r} not in df columns "
                f"{list(df_plot.columns)}; skipping curve labels.",
                stacklevel=2,
            )
        else:
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())
            _annotate_curve_labels(
                ax, df_plot, group_keys, value_col,
                label_by=label_by, color_by=color_by,
                color_map=color_map,
                is_numeric_label=pd.api.types.is_numeric_dtype(
                    df_plot[label_by]
                ),
            )

    if wavelength is None:
        wavelength = _single_wavelength(df)

    if explicit_value_col:
        ylabel = value_col
    else:
        signal = _guess_signal_kind(df)
        if signal == 'fluorescence':
            ylabel = (
                f'Fluorescence ({wavelength:g} nm, RFU)' if wavelength is not None
                else 'Fluorescence (RFU)'
            )
        else:
            ylabel = (
                f'Absorbance ({wavelength:g} nm)' if wavelength is not None
                else 'Absorbance'
            )
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9.5)
    if time_unit != 's':
        _apply_time_ticks(ax, time_unit)
    if axins is not None:
        axins.set_xlim(0, t_end_fit_max)
        axins.tick_params(labelsize=8)
        if time_unit != 's':
            _apply_time_ticks(axins, time_unit)

    annotate_modes = _resolve_annotate_modes(annotate_rates)

    signal_kind = df.attrs.get('signal_kind') if hasattr(df, 'attrs') else None
    rate_col = _rate_col_label(value_col, signal_kind, _single_wavelength(df))
    # Prefer the rate column actually present in rates_df (its name already
    # encodes the unit, e.g. 'A284/s'); fall back to the reconstructed label.
    if rates_df is not None and rate_col not in getattr(rates_df, 'columns', []):
        present = [c for c in getattr(rates_df, 'columns', [])
                   if isinstance(c, str) and c.startswith('Initial Rate (')]
        if present:
            rate_col = present[0]
    rate_unit = (rate_col[len('Initial Rate ('):-1]
                 if rate_col.startswith('Initial Rate (')
                 else SIGNAL_RATE_UNIT_BY_KIND.get(signal_kind or 'absorbance', 'ΔAbs/s'))

    rate_by_level = {}
    if 'legend' in annotate_modes and rates_df is not None and len(rates_df) \
            and color_by in rates_df.columns and rate_col in rates_df.columns:
        # Only annotate the legend when a level maps to a single rate. With
        # many curves per level (e.g. a substrate series), collapsing to one
        # mean reads as a single arbitrary rate and hides the real spread, so
        # omit the annotation in that case.
        rate_by_level = {
            level: vals.iloc[0]
            for level, vals in rates_df.dropna(subset=[rate_col])
                                       .groupby(color_by)[rate_col]
            if len(vals) == 1
        }

    if 'lines' in annotate_modes and line_label_data:
        try:
            from adjustText import adjust_text
        except ImportError as e:
            raise ImportError(
                "annotate_rates='lines' requires the optional `adjustText` "
                "package — install with `pip install adjustText`"
            ) from e
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        n = len(line_label_data)
        texts = []
        for i, (row, m, b, t_exit) in enumerate(line_label_data):
            frac = 0.25 + (i + 0.5) / n * 0.55
            t_lbl = max(min(frac * t_exit, xmax * 0.95), xmax * 0.05)
            a_lbl = m * t_lbl + b
            rate = row.get(rate_col, -m)
            texts.append(ax.text(t_lbl, a_lbl, f'{rate:.2e}',
                                 fontsize=8, ha='center', va='center',
                                 bbox=dict(facecolor='white', edgecolor='none',
                                           alpha=0.85, pad=1.2),
                                 zorder=12))
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle='-', color='0.4', lw=0.5),
            expand=(1.1, 1.4),
        )

    has_zero_baseline_entry = (
        is_numeric
        and zero_baseline_color is not None
        and any(v == 0 for v in levels)
    )

    def _legend_label(v):
        base = fmt_label(v)
        if has_zero_baseline_entry and v == 0 and zero_baseline_label:
            base = f'{base}  ({zero_baseline_label})'
        if v in rate_by_level:
            base = f'{base}  ({rate_by_level[v]:.2e} {rate_unit})'
        if legend_windows and v in fit_window_by_level:
            ts, te = fit_window_by_level[v]
            base = (f'{base}  [fit {ts / seconds_per_unit:g}–'
                    f'{te / seconds_per_unit:g} {time_unit}]')
        return base

    line_marker_kw = (
        dict(marker='', linestyle='-', linewidth=2.0)
        if collapse_replicates
        # In per-well mode, match the legend swatch alpha to the scatter points
        # when point_alpha is set, so the key reads at the same opacity as the
        # plotted data. Left opaque (alpha unset) when point_alpha is None.
        else dict(marker='o', linestyle='', markersize=6,
                  **({} if point_alpha is None else dict(alpha=point_alpha)))
    )

    draw_colorbar = bool(colorbar) and is_numeric
    if colorbar and not is_numeric:
        warnings.warn(
            f"colorbar=True ignored: color_by={color_by!r} is not numeric.",
            stacklevel=2,
        )

    if legend and has_hollow:
        col_name = next(iter(hollow_where))
        val = hollow_where[col_name]
        if collapse_replicates:
            style_handles = [
                mpl.lines.Line2D([0], [0], color='gray', lw=2.0, ls='-',
                                 label=f'{col_name} ≠ {val}'),
                mpl.lines.Line2D([0], [0], color='gray', lw=1.5, ls='--',
                                 label=f'{col_name} = {val}'),
            ]
        else:
            style_handles = [
                mpl.lines.Line2D([0], [0], marker='o', linestyle='',
                                 mfc='gray', mec='gray', markersize=6,
                                 label=f'{col_name} ≠ {val}'),
                mpl.lines.Line2D([0], [0], marker='o', linestyle='',
                                 mfc='none', mec='gray', mew=0.7, markersize=6,
                                 label=f'{col_name} = {val}'),
            ]
        style_leg = ax.legend(handles=style_handles, loc='upper left',
                              bbox_to_anchor=(1.05, 0.42), frameon=False,
                              fontsize=9)
        ax.add_artist(style_leg)
        color_anchor = (1.05, 0.27)
    else:
        color_anchor = (1.05, 0.42)

    if legend and not draw_colorbar:
        handles = [
            mpl.lines.Line2D([0], [0], color=color_map[v],
                             label=_legend_label(v), **line_marker_kw)
            for v in levels
        ]
        ax.legend(handles=handles, title=color_by, loc='upper left',
                  bbox_to_anchor=color_anchor, frameon=False,
                  fontsize=9, title_fontsize=10)

    if draw_colorbar:
        # Grow the bar with the level count so labels don't collide: extend
        # downward from the legacy top, pin to a floor and grow upward once it
        # would overflow. Small level counts keep the original 0.50 height.
        cbar_top = color_anchor[1] + 0.18  # legacy top: (… - 0.32) + 0.50
        cbar_h = float(np.clip(len(levels) * 0.06, 0.50, 0.96))
        cbar_y = max(0.02, cbar_top - cbar_h)
        if colorbar == 'continuous':
            _draw_continuous_colorbar(
                ax, levels, color_map,
                cmap=cmap, cmap_range=cmap_range,
                title=color_by,
                has_zero_baseline=has_zero_baseline_entry,
                zero_baseline_label=zero_baseline_label,
                fmt_label=fmt_label,
                bbox=(1.05, cbar_y, 0.05, cbar_h),
                alpha=point_alpha,
            )
        else:
            _draw_discrete_colorbar(
                ax, levels, color_map,
                title=color_by,
                has_zero_baseline=has_zero_baseline_entry,
                zero_baseline_label=zero_baseline_label,
                fmt_label=fmt_label,
                bbox=(1.05, cbar_y, 0.05, cbar_h),
                alpha=point_alpha,
            )

    return axins


def plot_initial_rates(
    rates_df,
    x_col='S (µM)',
    group_col='Substrate',
    y_col=None,
    mm_params_df=None,
    fit='auto',
    t_start_fit=0,
    t_end_fit=100,
    split_by=None,
    residuals=False,
    exclude=None,
    fit_range=None,
    title=None,
    fit_color=None,
    color_dict=None,
    xlim=None,
    ylim=None,
    point_alpha=None,
    point_size=None,
    xscale=None,
    yscale=None,
    figsize=None,
    dpi=_style.DEFAULT_DPI,
    transparent=False,
):
    """Scatter rates vs `x_col`, with an optional fit overlay.

    For an MM substrate titration: leave defaults (x_col='S (µM)',
    group_col='Substrate', fit='mm'). If you don't pass mm_params_df, the MM
    fit is computed for you via fit_michaelis_menten(group_by=group_col); pass
    mm_params_df= explicitly to reuse a fit (e.g. with custom exclusions). For
    an enzyme titration at fixed [S]: x_col='E (nM)', group_col='Enzyme',
    fit='linear'. For other numeric x, set x_col and fit=None.

    If a raw kinetic DataFrame is passed (i.e. it has a 'Time [s]' column),
    `compute_initial_rates(rates_df, t_end=t_end_fit)` is run internally so
    you don't have to do that step separately for quick exploration.

    Excluded points (via `exclude`) are drawn as X markers but not used in
    fits. To instead keep points on the plot but drop them from the fit only
    (e.g. fit just the linear region of an enzyme titration), use `fit_range`:
    out-of-range points are still drawn normally but don't feed the fit, and
    the fitted range is reported in the legend.

    If a 'Replicate' column is present with >1 unique value, individual
    replicate points are shown as faint dots and the mean ± SEM (per x_col,
    per group) is overlaid as colored points with error bars.

    Parameters
    ----------
    y_col : str | None
        Initial-rate column name. If None, derived from rates_df.attrs
        ['signal_kind'] (e.g. 'Initial Rate (ΔRFU/s)' for fluorescence)
        with a fallback to whichever 'Initial Rate (…)' column exists.
    fit : 'auto' | 'mm' | 'linear' | None
        Overlay type. 'auto' picks 'mm' if mm_params_df is given, else None.
        'mm' overlays a Michaelis-Menten fit, computing one per group if
        mm_params_df is not supplied. 'linear' fits y vs x per group through
        the origin-extended range.
    t_end_fit : float
        Forwarded to compute_initial_rates when a raw kinetic df is passed.
    fit_range : tuple[float, float] | None
        If set, only points whose `x_col` is within [lo, hi] (inclusive) feed
        the fit. Points outside the range are still plotted normally but
        excluded from the fit (and from residuals); the fitted range is shown
        in the legend. The fit line still extends across the full plotted x
        range. Applies to both 'linear' and an auto-computed 'mm' fit. Default
        None uses every included point.
    fit_color : str | None
        If None and there are multiple groups, each fit gets its own
        `tab10` color. If set, all fits use this single color.
    color_dict : dict | None
        Map of `group_col` value → color, overriding the auto-assigned `tab10`
        color per group (e.g. {'Hs1G': 'green', 'Hs1': 'grey'}). Groups absent
        from the dict keep their default palette color; ignored when
        `fit_color` forces a single color. Keys match by value or string form.
    xlim : tuple[float, float] | None
        If set, the x-axis limits (xmin, xmax) applied to the rate plot (and,
        when faceting/residuals are on, to every shared x-axis). Fit curves are
        drawn across this full range rather than stopping at the last data
        point (MM fits clamp the lower bound to 0). Default None keeps the auto
        limits and fits that span only the observed x range.
    ylim : tuple[float, float] | None
        If set, the y-axis limits (ymin, ymax). Default None derives the limits
        from the plotted data points; fit curves are clipped to that range
        rather than expanding it.
    point_alpha : float | None
        Opacity (0–1) for the data-point markers. Default None uses the
        existing per-style opacity (fully opaque colored points; faint
        replicate dots unchanged).
    point_size : float | None
        Marker area (matplotlib scatter `s`) for the main data points. Default
        None uses the built-in size (42). The mean±SEM marker (when replicates
        are present) scales with it; faint replicate dots are unchanged.
    xscale, yscale : str | None
        Axis scale passed to `ax.set_xscale` / `ax.set_yscale` (e.g. 'log',
        'symlog'). Default None leaves the matplotlib default ('linear'). With
        xscale='log', fit curves are drawn over a log-spaced x range and
        non-positive x points are dropped by matplotlib.
    transparent : bool
        If True, make the figure + axes background transparent (also on save).
        Default False (white).
    """
    if 'Time [s]' in rates_df.columns:
        rates_df = compute_initial_rates(rates_df,
                                         t_start=t_start_fit, t_end=t_end_fit,
                                         drop_no_enzyme=False)

    if x_col not in rates_df.columns:
        raise KeyError(f"x_col={x_col!r} not in rates_df: {list(rates_df.columns)}")

    signal_kind = rates_df.attrs.get('signal_kind') if hasattr(rates_df, 'attrs') else None
    rate_unit = SIGNAL_RATE_UNIT_BY_KIND.get(signal_kind or 'absorbance', 'ΔAbs/s')
    if y_col is None:
        preferred = _rate_col_label(None, signal_kind, _single_wavelength(rates_df))
        if preferred in rates_df.columns:
            y_col = preferred
        else:
            y_col = next(
                (c for c in rates_df.columns if c.startswith('Initial Rate')),
                preferred,
            )
    if y_col not in rates_df.columns:
        raise KeyError(
            f"y_col={y_col!r} not in rates_df: {list(rates_df.columns)}; "
            "did you forget compute_initial_rates()?"
        )

    if fit == 'auto':
        fit = 'mm' if mm_params_df is not None and len(mm_params_df) else None
    if fit not in (None, 'mm', 'linear'):
        raise ValueError(
            f"fit={fit!r}; expected 'auto', 'mm', 'linear', or None"
        )
    if residuals and fit is None:
        raise ValueError("residuals=True requires fit='mm' or fit='linear'")

    if fit_range is not None:
        if len(fit_range) != 2 or fit_range[0] > fit_range[1]:
            raise ValueError(
                f"fit_range={fit_range!r}; expected (lo, hi) with lo <= hi"
            )

    if fit == 'mm' and (mm_params_df is None or not len(mm_params_df)):
        mm_input = rates_df
        if fit_range is not None:
            x_num = pd.to_numeric(rates_df[x_col], errors='coerce')
            mm_input = rates_df[x_num.between(fit_range[0], fit_range[1])]
        mm_params_df = fit_michaelis_menten(
            mm_input, exclude=exclude, group_by=group_col, s_col=x_col,
        )

    if split_by is not None:
        if split_by not in rates_df.columns:
            raise KeyError(
                f"split_by={split_by!r} not in rates_df: {list(rates_df.columns)}"
            )
        levels = list(rates_df[split_by].dropna().unique())
        if not levels:
            raise ValueError(f"split_by={split_by!r} has no non-null values")
        n = len(levels)
        per_panel = figsize or _style.DEFAULT_FIGSIZE_WIDE
        facet_figsize = (per_panel[0] * n * 0.85, per_panel[1])
        if residuals:
            facet_figsize = (facet_figsize[0], facet_figsize[1] * 1.25)
            fig, gs_axes = plt.subplots(
                2, n, figsize=facet_figsize, dpi=dpi,
                sharex='col', sharey='row', squeeze=False,
                gridspec_kw=dict(height_ratios=[3, 1], hspace=0.25),
            )
            main_axes = list(gs_axes[0])
            resid_axes = list(gs_axes[1])
            for ax_m in main_axes:
                ax_m.tick_params(labelbottom=False)
        else:
            fig, axes = plt.subplots(
                1, n, figsize=facet_figsize, dpi=dpi,
                sharey=True, squeeze=False,
            )
            main_axes = list(axes[0])
            resid_axes = [None] * n

        panel_param_infos = []
        for ax_m, ax_r, lvl in zip(main_axes, resid_axes, levels):
            sub = rates_df[rates_df[split_by] == lvl].copy()
            sub.attrs.update(rates_df.attrs)
            sub_mm = (
                mm_params_df[mm_params_df[split_by] == lvl]
                if mm_params_df is not None
                and split_by in mm_params_df.columns
                else mm_params_df
            )
            info = _plot_initial_rates_on_ax(
                ax_m, ax_r, sub,
                x_col=x_col, group_col=group_col, y_col=y_col,
                mm_params_df=sub_mm, fit=fit,
                exclude=exclude, fit_range=fit_range,
                fit_color=fit_color, color_dict=color_dict,
                point_alpha=point_alpha, point_size=point_size,
                xlim=xlim, ylim=ylim,
                xscale=xscale, yscale=yscale,
                legend=False,
            )
            if xlim is not None:
                ax_m.set_xlim(*xlim)
            per_facet_in = facet_figsize[0] / n
            wrap_w = max(10, int(per_facet_in * 6))
            ax_m.set_title(
                textwrap.fill(
                    f'{split_by} = {lvl}', width=wrap_w,
                    break_long_words=False, break_on_hyphens=False,
                ),
                fontsize=10,
            )
            if info is not None:
                panel_param_infos.append(info)
        if panel_param_infos:
            totals = {c: 0 for c in _CORNER_ANCHORS}
            for info in panel_param_infos:
                scores = _score_corners(
                    info['ax'], info['corner_xs'], info['corner_ys'],
                    frac_x=info['frac_x'], frac_y=info['frac_y'],
                )
                for c, s in scores.items():
                    totals[c] += s
            best_corner = min(totals, key=totals.get)
            for info in panel_param_infos:
                _annotate_fit_params(info['ax'], info['text_lines'], best_corner)
        for ax_m in main_axes[1:]:
            ax_m.set_ylabel('')
        for ax_r in resid_axes[1:]:
            if ax_r is not None:
                ax_r.set_ylabel('')
        if title:
            fig.suptitle(title, fontsize=11)
        if not residuals:
            fig.tight_layout()
        else:
            fig.align_ylabels([main_axes[0], resid_axes[0]])
        _style._apply_background(fig, main_axes + [a for a in resid_axes if a is not None],
                          transparent)
        if residuals:
            return fig, main_axes, resid_axes
        return fig, main_axes

    panel_figsize = figsize or _style.DEFAULT_FIGSIZE_WIDE
    if residuals:
        panel_figsize = (panel_figsize[0], panel_figsize[1] * 1.25)
        fig, (ax, ax_resid) = plt.subplots(
            2, 1, figsize=panel_figsize, dpi=dpi,
            sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.25),
        )
        ax.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=panel_figsize, dpi=dpi)
        ax_resid = None

    info = _plot_initial_rates_on_ax(
        ax, ax_resid, rates_df,
        x_col=x_col, group_col=group_col, y_col=y_col,
        mm_params_df=mm_params_df, fit=fit,
        exclude=exclude, fit_range=fit_range,
        fit_color=fit_color, color_dict=color_dict,
        point_alpha=point_alpha, point_size=point_size,
        xlim=xlim, ylim=ylim,
        xscale=xscale, yscale=yscale,
        legend=True,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if info is not None:
        scores = _score_corners(
            info['ax'], info['corner_xs'], info['corner_ys'],
            frac_x=info['frac_x'], frac_y=info['frac_y'],
        )
        best_corner = min(scores, key=scores.get)
        _annotate_fit_params(info['ax'], info['text_lines'], best_corner)
    if title:
        ax.set_title(title)
    fig.subplots_adjust(left=0.14, right=0.55, top=0.93, bottom=0.17)
    _style._apply_background(fig, [ax, ax_resid], transparent)
    if residuals:
        fig.align_ylabels([ax, ax_resid])
        return fig, ax, ax_resid
    return fig, ax


def _plot_initial_rates_on_ax(
    ax, ax_resid, rates_df, *,
    x_col, group_col, y_col,
    mm_params_df, fit,
    exclude=None, fit_range=None, fit_color=None, color_dict=None,
    point_alpha=None, point_size=None, xlim=None, ylim=None,
    xscale=None, yscale=None, legend=True,
):
    """Render the rates scatter + optional fit + optional residuals onto ax(es)."""
    pt_s = point_size if point_size is not None else 42
    pt_ms = pt_s ** 0.5  # scatter area (s) -> errorbar markersize (diameter)
    signal_kind = rates_df.attrs.get('signal_kind') if hasattr(rates_df, 'attrs') else None
    rate_unit = SIGNAL_RATE_UNIT_BY_KIND.get(signal_kind or 'absorbance', 'ΔAbs/s')

    excluded = _build_exclusion_mask(rates_df, exclude)
    incl = rates_df[~excluded]
    excl = rates_df[excluded]

    log_x = xscale == 'log'

    def _fit_x(lo, hi):
        """Fit-curve x samples, log-spaced (over positive x) on a log axis."""
        if log_x:
            pos = pd.to_numeric(rates_df[x_col], errors='coerce')
            pos = pos[pos > 0]
            floor = float(pos.min()) if len(pos) else hi / 1e3
            lo = lo if lo > 0 else floor
            return np.geomspace(lo, hi, 200)
        return np.linspace(lo, hi, 200)

    has_replicates = (
        'Replicate' in rates_df.columns
        and rates_df['Replicate'].nunique(dropna=True) > 1
    )
    has_groups = (
        group_col in rates_df.columns
        and rates_df[group_col].nunique(dropna=True) > 1
    )

    if has_groups:
        groups = list(rates_df[group_col].dropna().unique())
        cmap = plt.get_cmap('tab10')
        group_colors = {g: cmap(i % cmap.N) for i, g in enumerate(groups)}
        _apply_color_dict(group_colors, color_dict)
    else:
        group_colors = {}

    def _color_for(grp):
        if fit_color is not None:
            return fit_color
        return group_colors.get(grp, plt.get_cmap('tab10')(0))

    def _split_fit(sub):
        """(in_fit, out_of_fit) split of `sub` by fit_range on x_col."""
        if fit_range is None:
            return sub, sub.iloc[:0]
        x = pd.to_numeric(sub[x_col], errors='coerce')
        keep = x.between(fit_range[0], fit_range[1])
        return sub[keep], sub[~keep]

    def _scatter_pts(sub, c, label, alpha):
        ax.scatter(sub[x_col], sub[y_col],
                   s=pt_s, marker='o', facecolors=[_style._lighten(c)],
                   edgecolors=_style.POINT_EDGE_COLOR,
                   linewidths=_style.POINT_EDGE_WIDTH,
                   label=label, alpha=alpha, zorder=3)

    def _mean_sem(sub, c, label, alpha):
        agg = (
            sub.dropna(subset=[x_col, y_col])
               .groupby(x_col)[y_col]
               .agg(['mean', 'sem']).reset_index()
        )
        if not len(agg):
            return
        agg['sem'] = agg['sem'].fillna(0)
        ax.errorbar(agg[x_col], agg['mean'], yerr=agg['sem'],
                    fmt='o', markersize=pt_ms, color=c,
                    markerfacecolor=_style._lighten(c),
                    markeredgecolor=_style.POINT_EDGE_COLOR,
                    markeredgewidth=_style.POINT_EDGE_WIDTH,
                    ecolor=c, elinewidth=0.9, capsize=2.5, capthick=0.9,
                    linestyle='none', alpha=alpha, zorder=3, label=label)

    if has_groups:
        for grp, sub_incl in incl.groupby(group_col):
            c = _color_for(grp)
            # Without a fit, the data markers carry the group legend entry;
            # with a fit, the fit line glyph does, so leave markers unlabeled.
            grp_label = str(grp) if fit is None else None
            if has_replicates:
                ax.scatter(sub_incl[x_col], sub_incl[y_col],
                           s=10, marker='o', color=c,
                           alpha=0.30, edgecolors='none', zorder=2)
                _mean_sem(sub_incl, c, grp_label, point_alpha)
            else:
                _scatter_pts(sub_incl, c, grp_label, point_alpha)
    else:
        default_c = _color_for(None)
        if has_replicates:
            ax.scatter(incl[x_col], incl[y_col],
                       s=10, marker='o', facecolors='lightgray',
                       edgecolors='none', alpha=0.85,
                       label='replicates', zorder=2)
            _mean_sem(incl, default_c, 'mean ± SEM', point_alpha)
        else:
            # With a fit, the fit-line glyph carries the legend point, so the
            # bare data markers don't need their own 'data' entry.
            _scatter_pts(incl, default_c, 'data' if fit is None else None,
                         point_alpha)
    if len(excl):
        ax.scatter(excl[x_col], excl[y_col],
                   s=30, marker='x', c='k', linewidths=1.3,
                   label='excluded', zorder=4)

    # The plotted data points alone set the axis limits; fit curves drawn
    # below are clipped to this box rather than expanding it (so a fit drawn
    # across an out-of-fit-range region doesn't push the limits out).
    ax.margins(x=0.03, y=0.05)
    ax.autoscale_view()
    data_xlim, data_ylim = ax.get_xlim(), ax.get_ylim()

    fit_handles = []
    fit_face_by_handle = {}  # fit line -> data-point face color, for legend proxy
    fit_param_entries = []  # list of {grp, lines: [str, ...], color}
    fit_curve_xy = []  # (xs, ys) arrays of every drawn fit curve, for corner picking
    all_resid_ys = []
    if fit == 'mm' and mm_params_df is not None and len(mm_params_df):
        if group_col not in rates_df.columns:
            raise KeyError(
                f"group_col={group_col!r} not in rates_df; cannot overlay MM fits"
            )
        km_col = next(
            (col for col in mm_params_df.columns if col.startswith('Km (')),
            'Km (µM)',
        )
        km_unit_match = re.search(r'\(([^)]+)\)', km_col)
        km_unit = km_unit_match.group(1) if km_unit_match else 'µM'
        for _, row in mm_params_df.iterrows():
            grp = row[group_col]
            c = _color_for(grp)
            fit_c = c if has_groups else 'k'
            grp_mask = rates_df[group_col] == grp
            if xlim is not None:
                fit_lo, fit_hi = max(0.0, xlim[0]), xlim[1]
            else:
                fit_lo, fit_hi = 0.0, rates_df.loc[grp_mask, x_col].max()
            S_fit = _fit_x(fit_lo, fit_hi)
            vmax_col = next(
                (col for col in row.index if col.startswith('Vmax')),
                'Vmax (ΔAbs/s)',
            )
            v_fit = michaelis_menten(S_fit, row[vmax_col], row[km_col])
            label = (
                f"{grp}:\n    $K_M$ = {_fmt_sig(row[km_col])} ± "
                f"{_fmt_sig(row['Km_err'], 2)} {km_unit}\n"
                f"    $V_{{max}}$ = {_fmt_sig(row[vmax_col])} {rate_unit}"
            )
            line, = ax.plot(S_fit, v_fit, color='black', lw=1.5, ls='-',
                            label=label, zorder=2)
            fit_handles.append(line)
            fit_face_by_handle[line] = _style._lighten(c)
            fit_curve_xy.append((S_fit, v_fit))
            fit_param_entries.append({
                'grp': grp, 'color': fit_c,
                'lines': [
                    f"$K_M$ = {_fmt_sig(row[km_col])} ± "
                    f"{_fmt_sig(row['Km_err'], 2)} {km_unit}",
                    f"$V_{{max}}$ = {_fmt_sig(row[vmax_col])} {rate_unit}",
                ],
            })
            if ax_resid is not None:
                sub_pts = incl[incl[group_col] == grp].dropna(subset=[x_col, y_col])
                if not sub_pts.empty:
                    rxs = sub_pts[x_col].to_numpy(float)
                    rys = sub_pts[y_col].to_numpy(float)
                    pred = michaelis_menten(rxs, row[vmax_col], row[km_col])
                    resid = rys - pred
                    ax_resid.scatter(rxs, resid, s=14, color=c,
                                     edgecolors='none', alpha=0.85, zorder=3)
                    all_resid_ys.extend(resid.tolist())

    if fit == 'linear':
        x_unit_match = re.search(r'\(([^)]+)\)', x_col)
        x_unit = x_unit_match.group(1) if x_unit_match else x_col
        slope_unit = f'({rate_unit})/{x_unit}'
        fit_src = _split_fit(incl)[0]
        fit_groups = list(
            fit_src.groupby(group_col) if group_col in fit_src.columns
            else [('all', fit_src)]
        )
        single_group = len(fit_groups) == 1
        for grp, sub in fit_groups:
            sub = sub.dropna(subset=[x_col, y_col])
            if sub[x_col].nunique() < 2:
                continue
            xs = sub[x_col].to_numpy(float)
            ys = sub[y_col].to_numpy(float)
            res = stats.linregress(xs, ys)
            c = _color_for(grp) if group_col in rates_df.columns else _color_for(None)
            fit_c = c if has_groups else 'k'
            if xlim is not None:
                fit_lo, fit_hi = xlim[0], xlim[1]
            elif fit_range is not None:
                # Fit on the subset, but draw the line across the full plotted
                # x range so it extends through the out-of-range points too.
                grp_x = (incl[incl[group_col] == grp][x_col]
                         if has_groups else incl[x_col])
                fit_lo, fit_hi = 0.0, float(pd.to_numeric(grp_x, errors='coerce').max())
            else:
                fit_lo, fit_hi = 0.0, float(xs.max())
            x_fit = _fit_x(fit_lo, fit_hi)
            y_fit = res.slope * x_fit + res.intercept
            entry_lines = [
                f"slope = {_fmt_sig(res.slope)} {slope_unit}",
                f"R² = {res.rvalue ** 2:.3f}",
            ]
            if fit_range is not None:
                entry_lines.append(
                    f"fit range: {_fmt_sig(fit_range[0])}–{_fmt_sig(fit_range[1])} {x_unit}"
                )
            # A single fit needs no group header (it would just repeat the
            # substrate/enzyme name); multiple groups label each one.
            if single_group:
                label = '\n'.join(entry_lines)
            else:
                label = f"{grp}:\n" + '\n'.join(f"    {ln}" for ln in entry_lines)
            line, = ax.plot(x_fit, y_fit, color='black', lw=1.0,
                            ls=(0, (5, 3)), label=label, zorder=2)
            fit_handles.append(line)
            fit_face_by_handle[line] = _style._lighten(c)
            fit_curve_xy.append((x_fit, y_fit))
            fit_param_entries.append({
                'grp': grp, 'color': fit_c,
                'lines': entry_lines,
            })
            if ax_resid is not None:
                pred = res.slope * xs + res.intercept
                resid = ys - pred
                ax_resid.scatter(xs, resid, s=14, color=c,
                                 edgecolors='none', alpha=0.85, zorder=3)
                all_resid_ys.extend(resid.tolist())

    ax.set_ylabel(y_col, fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.locator_params(axis='x', nbins=7)
    if ax_resid is None:
        ax.set_xlabel(x_col, fontsize=11)
    else:
        ax_resid.axhline(0, color='0.4', lw=0.8, ls='-', zorder=1)
        ax_resid.set_xlabel(x_col, fontsize=11)
        ax_resid.set_ylabel('residual', fontsize=11)
        ax_resid.tick_params(labelsize=9.5)
        ax_resid.margins(x=0.03)
        ax_resid.locator_params(axis='x', nbins=7)
        if all_resid_ys:
            max_abs = max(abs(v) for v in all_resid_ys if np.isfinite(v))
            tick = _nice_tick(max_abs) if max_abs > 0 else 1.0
            ax_resid.set_ylim(-tick * 1.5, tick * 1.5)
            ax_resid.set_yticks([-tick, 0.0, tick])

    if xscale is not None:
        ax.set_xscale(xscale)  # shared x propagates to ax_resid
        if xscale == 'log':
            # Decimal tick labels (1, 10, 100) instead of 10^n scientific.
            fmt = mpl.ticker.FuncFormatter(
                lambda v, _pos: f'{v:g}' if v > 0 else ''
            )
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    if yscale is not None:
        ax.set_yscale(yscale)

    # Apply explicit limits, else snap back to the data-driven limits captured
    # before the fit curves were drawn (linear axes only — leave matplotlib's
    # log autoscale alone, where the pre-fit linear capture wouldn't be valid).
    if xlim is not None:
        ax.set_xlim(*xlim)
    elif xscale is None:
        ax.set_xlim(*data_xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif yscale is None:
        ax.set_ylim(*data_ylim)

    param_info = None
    if not legend and fit_param_entries:
        incl_xs = pd.to_numeric(incl[x_col], errors='coerce').to_numpy(float)
        incl_ys = pd.to_numeric(incl[y_col], errors='coerce').to_numpy(float)
        if fit_curve_xy:
            curve_xs = np.concatenate([c[0] for c in fit_curve_xy])
            curve_ys = np.concatenate([c[1] for c in fit_curve_xy])
            corner_xs = np.concatenate([incl_xs, curve_xs])
            corner_ys = np.concatenate([incl_ys, curve_ys])
        else:
            corner_xs, corner_ys = incl_xs, incl_ys
        n_groups = len(fit_param_entries)
        text_lines = []  # (text, color)
        for entry in fit_param_entries:
            if n_groups > 1:
                text_lines.append((entry['grp'], entry['color']))
            text_lines.extend((s, entry['color']) for s in entry['lines'])
            if n_groups > 1:
                text_lines.append(('', entry['color']))  # spacer

        # Measure the actual rendered bbox so the corner picker can avoid
        # any quadrant where data or fit curves overlap the text footprint.
        fig = ax.figure
        sample = '\n'.join(t for t, _ in text_lines if t) or ' '
        draft = ax.text(0.0, 0.0, sample, transform=ax.transAxes,
                        fontsize=7.5, ha='left', va='bottom', alpha=0.0)
        fig.canvas.draw()
        bbox_ax = draft.get_window_extent().transformed(ax.transAxes.inverted())
        draft.remove()
        frac_x = float(np.clip(bbox_ax.width + 0.04, 0.22, 0.6))
        frac_y = float(np.clip(bbox_ax.height + 0.04, 0.22, 0.6))

        param_info = dict(
            ax=ax, text_lines=text_lines,
            corner_xs=corner_xs, corner_ys=corner_ys,
            frac_x=frac_x, frac_y=frac_y,
        )
    if legend and not (
        len(ax.get_legend_handles_labels()[1]) == 1
        and ax.get_legend_handles_labels()[1][0] == 'data'
    ):
        handles, labels = ax.get_legend_handles_labels()
        # Render each fit's legend glyph as a data point sitting on its fit line
        # (marker overlaid on the line) rather than a bare line segment, and lift
        # that glyph to the first text line (the group name) of its label.
        handler_map = {}
        new_handles = []
        for h, lbl in zip(handles, labels):
            if h in fit_face_by_handle:
                proxy = mpl.lines.Line2D(
                    [], [], color=h.get_color(), lw=h.get_linewidth(),
                    ls=h.get_linestyle(), marker='o', markersize=pt_ms,
                    markerfacecolor=fit_face_by_handle[h],
                    markeredgecolor=_style.POINT_EDGE_COLOR,
                    markeredgewidth=_style.POINT_EDGE_WIDTH,
                )
                new_handles.append(proxy)
                # With a group header on the first line, pin the glyph there;
                # for a single header-less fit, let it center on the text block.
                if has_groups:
                    handler_map[proxy] = _TopLineHandler(
                        n_lines=lbl.count('\n') + 1)
            else:
                new_handles.append(h)
        # Multi-line fit-parameter labels need vertical breathing room; plain
        # single-line labels (no fit) read better packed close together.
        multiline = any('\n' in l for l in labels)
        leg = ax.legend(new_handles, labels, fontsize=8, loc='lower left',
                        bbox_to_anchor=(1.08, 0.0), borderaxespad=0.,
                        frameon=False, numpoints=1, handler_map=handler_map,
                        labelspacing=1.4 if multiline else 0.4,
                        title=group_col if has_groups else None)
        if leg.get_title().get_text():
            leg.get_title().set_fontsize(8)
            leg.get_title().set_fontweight('bold')
    return param_info


def _resolve_fold_ref(rates_df, x_col, y_col, fold_ref, medians):
    """Resolve `fold_ref` to a single x_col category for fold-change.

    First tries `fold_ref` as an x_col category directly. Failing that — useful
    when x_col is a composite label like 'Construct · Notebook' but you name the
    reference by its construct — it looks for `fold_ref` among the *other*
    columns (e.g. 'Construct') and returns the x_col category those rows belong
    to. Returns the matching category, None if nothing matches, or raises when
    the match spans several categories (ambiguous — e.g. an 'Hs1G' present in
    two notebooks).
    """
    direct = _match_category(fold_ref, medians.keys())
    if direct is not None:
        return direct

    matched = set()
    for col in rates_df.columns:
        if col in (x_col, y_col):
            continue
        m = _match_category(fold_ref, rates_df[col].dropna().unique())
        if m is None:
            continue
        cats_here = rates_df.loc[rates_df[col] == m, x_col].dropna().unique()
        matched.update(c for c in cats_here if c in medians)
    if not matched:
        return None
    if len(matched) > 1:
        raise KeyError(
            f"fold_ref={fold_ref!r} matches multiple {x_col!r} categories "
            f"{sorted(matched)}; pass the full {x_col!r} value to disambiguate"
        )
    return next(iter(matched))


def _set_categorical_xticklabels(ax, cats, fig_w_in, rotate, wrap):
    """Lay out categorical x tick labels, wrapping/rotating long ones to fit.

    Sizes each label against the per-category slot width (figure inches / n).
    `wrap` and `rotate` accept 'auto' (decide from the available space), an
    explicit value (wrap width in chars / rotation in degrees), or None/False
    to disable. In 'auto' mode: labels that fit are left flat; otherwise they're
    wrapped to the slot, and if even one wrapped line still overflows they're
    rotated 30° instead (rotation reads better than aggressive hyphenation).
    """
    labels = [str(c) for c in cats]
    n = max(len(labels), 1)
    per_cat_in = (fig_w_in / n) if fig_w_in else 1.0
    # rough chars-per-inch at the ~9.5 pt tick font
    chars_per_in = 13.0
    slot_chars = max(4, int(per_cat_in * chars_per_in))
    longest = max((len(s) for s in labels), default=0)

    def _wrapped(width):
        return [
            textwrap.fill(s, width=max(4, width), break_long_words=False,
                          break_on_hyphens=False)
            for s in labels
        ]

    if wrap == 'auto' and rotate == 'auto':
        if longest <= slot_chars:
            disp, angle = labels, 0
        else:
            wrapped = _wrapped(slot_chars)
            max_line = max(
                (max(len(ln) for ln in w.split('\n')) for w in wrapped),
                default=0,
            )
            if max_line <= slot_chars:
                disp, angle = wrapped, 0
            else:
                disp, angle = labels, 30
    else:
        if wrap in (None, False, 'auto'):
            disp = labels
        else:
            disp = _wrapped(int(wrap))
        if rotate in (None, False, 'auto'):
            angle = 0 if (rotate != 'auto' or longest <= slot_chars) else 30
        else:
            angle = float(rotate)

    ax.set_xticks(range(len(labels)))
    if angle:
        ax.set_xticklabels(disp, rotation=angle, ha='right',
                           rotation_mode='anchor')
    else:
        ax.set_xticklabels(disp)


def _categorical_items(rates_df, x_cols, y_col, order, group_gap=0.7):
    """Lay out one item per (group, inner-level) combination along the x-axis.

    `x_cols` is one or more columns. The last is the inner level (its own tick);
    any preceding columns form the group key (drawn as a bracket below). Returns
    `(items, group_spans)` — each item carries its rate values, boolean row mask,
    x position, inner-level label, and compound label; `group_spans` pairs each
    group key with the x positions it covers, for bracket drawing.
    """
    inner_col = x_cols[-1]
    group_cols = list(x_cols[:-1])
    grouped = bool(group_cols)

    valid = rates_df.dropna(subset=list(x_cols))
    group_order, inner_by_group = [], {}
    for _, r in valid.iterrows():
        g = tuple(r[c] for c in group_cols)
        iv = r[inner_col]
        if g not in inner_by_group:
            inner_by_group[g] = []
            group_order.append(g)
        if iv not in inner_by_group[g]:
            inner_by_group[g].append(iv)

    if order is not None:
        if grouped:
            ordered = []
            for o in order:
                for g in group_order:
                    if g and (g[0] == o or str(g[0]) == str(o)) and g not in ordered:
                        ordered.append(g)
            group_order = ordered + [g for g in group_order if g not in ordered]
        else:
            g = group_order[0] if group_order else ()
            present = inner_by_group.get(g, [])
            inner_by_group[g] = (
                [o for o in order if o in present]
                + [v for v in present if v not in order]
                + [o for o in order if o not in present]  # empty ticks, as before
            )
            if g not in inner_by_group:  # order given but no data at all
                group_order, inner_by_group[g] = [g], list(order)

    pos = 0.0
    items, group_spans = [], []
    for g in group_order:
        xs = []
        for iv in inner_by_group[g]:
            mask = pd.Series(True, index=rates_df.index)
            for c, v in zip(group_cols, g):
                mask &= (rates_df[c] == v)
            mask &= (rates_df[inner_col] == iv)
            items.append({
                'group': g,
                'inner': iv,
                'x': pos,
                'vals': rates_df.loc[mask, y_col].dropna().to_numpy(),
                'mask': mask,
                'inner_label': _fmt_level(inner_col, iv),
                'compound': _compound_label(x_cols, list(g) + [iv]),
            })
            xs.append(pos)
            pos += 1.0
        group_spans.append((g, xs))
        pos += group_gap
    return items, group_spans


def _item_color(item, color_dict, default):
    """Resolve an item's color from `color_dict`, trying its compound label, its
    primary (group) value, then its inner value, before falling back."""
    candidates = [item['compound']]
    if item['group']:
        candidates.append(item['group'][0])
    candidates.append(item['inner'])
    for key in candidates:
        c = _color_dict_get(color_dict, key, '\0miss')
        if c != '\0miss':
            return c
    return default


def _resolve_fold_ref_item(rates_df, x_cols, y_col, fold_ref, items):
    """Resolve `fold_ref` to a single item index (one with data), matching its
    compound label, any of its x_cols values, or — failing that — a value in
    some other column. Raises if the match spans several items."""
    valid = [i for i, it in enumerate(items) if len(it['vals'])]
    for i in valid:
        if str(items[i]['compound']) == str(fold_ref):
            return i
    matched = set()
    for i in valid:
        vals = list(items[i]['group']) + [items[i]['inner']]
        if any(v == fold_ref or str(v) == str(fold_ref) for v in vals):
            matched.add(i)
    if not matched:
        for col in rates_df.columns:
            if col in x_cols or col == y_col:
                continue
            m = _match_category(fold_ref, rates_df[col].dropna().unique())
            if m is None:
                continue
            for i in valid:
                if (items[i]['mask'] & (rates_df[col] == m)).any():
                    matched.add(i)
    if len(matched) == 1:
        return next(iter(matched))
    if len(matched) > 1:
        labels = [items[i]['compound'] for i in sorted(matched)]
        raise KeyError(
            f"fold_ref={fold_ref!r} matches multiple categories {labels}; "
            "pass the full compound label to disambiguate"
        )
    return None


def _draw_group_brackets(ax, group_spans, group_cols):
    """Draw an outer-level bracket + label beneath the inner ticks for each
    group (the leading x_cols when several are passed)."""
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    y_line = -0.135
    for g, xs in group_spans:
        if not xs:
            continue
        left, right = min(xs) - 0.35, max(xs) + 0.35
        ax.plot([left, right], [y_line, y_line], color='0.35', lw=1.0,
                transform=trans, clip_on=False)
        ax.text((left + right) / 2, y_line - 0.025, _compound_label(group_cols, g),
                transform=trans, ha='center', va='top', fontsize=10, clip_on=False)


def plot_rates_categorical(
    rates_df,
    x_col,
    y_col='Initial Rate (ΔAbs/s)',
    order=None,
    point_color=None,
    color_dict=None,
    line_color='black',
    jitter=0.15,
    annotate=True,
    fold_change=False,
    fold_ref=None,
    rotate_labels='auto',
    wrap_labels='auto',
    xlim=None,
    ylim=None,
    figsize=_style.DEFAULT_FIGSIZE,
    dpi=_style.DEFAULT_DPI,
    ax=None,
    transparent=False,
):
    """Strip plot of rates by a categorical column, with a median bar per group.

    Each replicate is shown as a jittered point (outlined lighter-`tab10`, the
    same per-category scheme as the progress-curve / rates plots); a horizontal
    black bar marks the median, with SEM whiskers when n > 1. The median bar is
    omitted for categories with a single rate. Useful for comparing
    variants/conditions at a single [S].

    Parameters
    ----------
    x_col : str | list[str]
        Categorical column to group by (e.g. 'Enzyme'). Pass a list of columns
        (e.g. ['Construct', 'E (nM)']) for a two-level x-axis: the last column
        is the inner level (one tick each), the preceding column(s) form an
        outer group drawn as a labeled bracket below — so a construct measured
        at several enzyme concentrations gets one bracket spanning its
        concentration sub-ticks. Numeric inner values carry their unit when the
        column name has one (e.g. '20 nM').
    order : list | None
        Optional explicit ordering. With a single `x_col`, orders the categories;
        with a list `x_col`, orders the outer groups (by their primary value).
    point_color : color | None
        If None (default), each category gets its own `tab10` color (lightened
        face + dark outline). Pass a single color to use it for every category.
    color_dict : dict | None
        Map of `x_col` category → color, overriding the auto-assigned `tab10`
        color per category (e.g. {'Hs1G': 'green', 'Hs1': 'grey'}). Takes
        precedence over `point_color` for matching categories; categories
        absent from the dict fall back to `point_color` or the default
        palette. Keys match by value or string form.
    line_color : color
        Color of the per-category median bar and SEM whiskers.
    annotate : bool
        If True, write the median rate value above each column.
    fold_change : bool
        If True, also annotate each column with its median fold-change relative
        to a reference category (e.g. '3.2×'); the reference itself shows '1×'.
    fold_ref : category | None
        Reference category for `fold_change`. If None (default), the first
        category with data is used. Matches an `x_col` category by value or
        string form; failing that, it's matched against the other columns — so
        with a composite `x_col` (e.g. 'Construct · Notebook') you can still
        name the reference by its construct (`fold_ref='Hs1G'`), as long as that
        resolves to a single category (it raises if it spans several, e.g. the
        same construct in two notebooks).
    rotate_labels : 'auto' | float | None
        X tick-label rotation in degrees. 'auto' (default) rotates 30° only when
        labels are too long to fit their slot (after wrapping); pass a number to
        force an angle, or None/0 to keep them flat.
    wrap_labels : 'auto' | int | None
        Wrap long x tick labels onto multiple lines. 'auto' (default) wraps to
        the per-category slot width; pass an int for an explicit character
        width, or None to disable wrapping. Useful for composite labels like
        'Construct · Notebook'.
    xlim, ylim : tuple[float, float] | None
        Axis limits (min, max), overriding the auto-computed limits. Default
        None keeps the auto limits (ylim adds headroom for the annotations).
    transparent : bool
        If True, make the figure + axes background transparent (also on save).
        Default False (white). Ignored when `ax` is passed.
    """
    x_cols = [x_col] if isinstance(x_col, str) else list(x_col)
    missing = [c for c in x_cols if c not in rates_df.columns]
    if missing:
        raise KeyError(f"x_col {missing} not in rates_df: {list(rates_df.columns)}")
    if y_col not in rates_df.columns:
        raise KeyError(f"y_col={y_col!r} not in rates_df: {list(rates_df.columns)}")
    grouped = len(x_cols) > 1
    group_cols = x_cols[:-1]

    items, group_spans = _categorical_items(rates_df, x_cols, y_col, order)
    for it in items:
        it['median'] = float(np.median(it['vals'])) if len(it['vals']) else None

    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    # Resolve the fold-change reference up front so every column can annotate
    # its fold relative to it.
    ref_val = None
    if fold_change and any(it['median'] is not None for it in items):
        if fold_ref is None:
            ref_idx = next(i for i, it in enumerate(items)
                           if it['median'] is not None)
        else:
            ref_idx = _resolve_fold_ref_item(rates_df, x_cols, y_col, fold_ref, items)
            if ref_idx is None:
                raise KeyError(
                    f"fold_ref={fold_ref!r} not found among categories with "
                    f"data or values of any other column"
                )
        ref_val = items[ref_idx]['median']

    # Default colors: one per group when several levels are passed (so sub-items
    # share a hue), else one per item (the single-column behavior).
    group_order = []
    for it in items:
        if it['group'] not in group_order:
            group_order.append(it['group'])
    group_idx = {g: i for i, g in enumerate(group_order)}

    cmap = plt.get_cmap('tab10')
    rng = np.random.default_rng(0)
    for i, it in enumerate(items):
        vals = it['vals']
        x = it['x']
        if len(vals) == 0:
            continue
        default_i = group_idx[it['group']] if grouped else i
        default_c = point_color if point_color is not None else cmap(default_i % cmap.N)
        base_c = _item_color(it, color_dict, default_c)
        if len(vals) > 1:
            x_pos = x + (rng.random(len(vals)) - 0.5) * 2 * jitter
        else:
            x_pos = np.full(len(vals), float(x))  # no jitter for a lone point
        ax.scatter(x_pos, vals, s=42, marker='o',
                   facecolors=[_style._lighten(base_c)],
                   edgecolors=_style.POINT_EDGE_COLOR, linewidths=_style.POINT_EDGE_WIDTH,
                   zorder=3)
        med_val = it['median']
        sem_val = 0.0
        # A single rate has no spread to summarize — skip the median bar.
        if len(vals) > 1:
            ax.hlines(med_val, x - 0.25, x + 0.25,
                      colors=line_color, lw=2, zorder=4)
            sem_val = float(stats.sem(vals))
            ax.errorbar(x, med_val, yerr=sem_val,
                        fmt='none', ecolor=line_color,
                        elinewidth=1.0, capsize=3, capthick=1.0,
                        zorder=2)
        # Stack labels above the point, nearest-first: the fold-change (the
        # takeaway) sits closest in bold, the raw median reads as a smaller grey
        # subtitle above it.
        entries = []  # (text, fontsize, color, weight), nearest-point first
        if fold_change and ref_val is not None:
            if ref_val != 0 and np.isfinite(med_val / ref_val):
                fold = med_val / ref_val
                fold_txt = '1×' if abs(fold - 1) < 5e-3 else f'{fold:.2g}×'
            else:
                fold_txt = '—'
            entries.append((fold_txt, 9, 'k', 'bold'))
        if annotate:
            # Demote the raw median to a small grey subtitle only when it shares
            # the label with a fold-change; alone it keeps the original look.
            if fold_change:
                entries.append((f'{med_val:.3g}', 7.5, '0.45', 'normal'))
            else:
                entries.append((f'{med_val:.3g}', 8, 'k', 'normal'))
        if entries:
            top_y = max(float(np.max(vals)), med_val + sem_val)
            y_off = 7
            for txt, fs, color, weight in entries:
                ax.annotate(txt,
                            xy=(x, top_y),
                            xytext=(0, y_off),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=fs, color=color, fontweight=weight,
                            zorder=5)
                y_off += fs + 3.5  # advance past this line for the next one

    positions = [it['x'] for it in items]
    inner_labels = [it['inner_label'] for it in items]
    if grouped:
        ax.set_xticks(positions)
        ax.set_xticklabels(inner_labels)
        if rotate_labels not in ('auto', None, False, 0):
            for t in ax.get_xticklabels():
                t.set_rotation(float(rotate_labels))
                t.set_ha('right')
        _draw_group_brackets(ax, group_spans, group_cols)
        ax.set_xlabel('')  # the two tick levels are self-labeling
    else:
        _set_categorical_xticklabels(
            ax, inner_labels, fig.get_size_inches()[0], rotate_labels, wrap_labels,
        )
        ax.set_xlabel(x_cols[0], fontsize=11)
    ax.set_xlim(min(positions) - 0.6, max(positions) + 0.6)
    ax.set_ylabel(y_col, fontsize=11)
    ax.tick_params(labelsize=9.5)
    all_vals = pd.to_numeric(rates_df[y_col], errors='coerce').dropna().to_numpy()
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif len(all_vals):
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        span = (hi - lo) or max(abs(hi), 1.0)
        # Extra headroom on top for the value annotation sitting above each
        # point; a two-line label (value + fold-change) needs a bit more.
        n_lines = int(annotate) + int(fold_change)
        top_pad = (0.12, 0.22, 0.30)[min(n_lines, 2)]
        ax.set_ylim(lo - 0.12 * span, hi + top_pad * span)
    else:
        ax.margins(y=0.15)
    if xlim is not None:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    if grouped and owns_fig:
        # Reserve room below the inner ticks for the group brackets + labels.
        fig.subplots_adjust(bottom=max(0.30, fig.subplotpars.bottom))
    if owns_fig:
        _style._apply_background(fig, ax, transparent)
    return fig, ax


def plot_spectra(
    scan_df,
    wells=None,
    n_timepoints=None,
    cmap_name='viridis',
    xlim=None,
    ylim=None,
    figsize_per_panel=_style.DEFAULT_FIGSIZE,
    dpi=_style.DEFAULT_DPI,
    sharey=True,
    legend_max=12,
    transparent=False,
):
    """Plot A vs wavelength colored by time, one panel per well.

    Useful for picking a probe wavelength before running extract_wavelength().

    Parameters
    ----------
    wells : str | list[str] | None
        Well(s) to plot. Default: all wells in scan_df.
    n_timepoints : int | None
        If None, draw every timepoint. If an int, downsample to that many
        evenly-spaced timepoints per panel.
    legend_max : int
        If number of timepoints drawn is ≤ this, show a discrete legend;
        otherwise show a continuous colorbar.
    xlim, ylim : tuple[float, float] | None
        Axis limits (min, max) applied to every panel. Default None keeps the
        auto limits.
    transparent : bool
        If True, make the figure + axes background transparent (also on save).
        Default False (white).
    """
    if wells is None:
        wells = sorted(scan_df['Well'].unique())
    elif isinstance(wells, str):
        wells = [wells]

    n = len(wells)
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
        dpi=dpi, sharey=sharey,
    )
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap(cmap_name)

    has_time = (
        'Time [s]' in scan_df.columns
        and scan_df['Time [s]'].dropna().nunique() > 1
    )

    for ax, w in zip(axes, wells):
        sub = scan_df[scan_df['Well'] == w]
        if not has_time:
            spec = sub.sort_values('Wavelength [nm]')
            ax.plot(spec['Wavelength [nm]'], spec['Absorbance'],
                    color=cmap(0.5), lw=1.2)
        else:
            times = np.array(sorted(sub['Time [s]'].dropna().unique()))
            if len(times) == 0:
                warnings.warn(
                    f"well {w!r} has no valid Time [s] data — skipping panel "
                    "(likely an aborted scan)",
                    stacklevel=2,
                )
                ax.set_axis_off()
                ax.set_title(f'Well {w} (no data)', fontsize=10)
                continue
            if n_timepoints is not None and len(times) > n_timepoints:
                idxs = np.linspace(0, len(times) - 1, n_timepoints).astype(int)
                sel_times = times[idxs]
            else:
                sel_times = times

            t_min, t_max = sel_times.min(), sel_times.max()
            norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)

            for t in sel_times:
                spec = sub[sub['Time [s]'] == t].sort_values('Wavelength [nm]')
                color = cmap(norm(t))
                label = f'{t:.0f} s' if len(sel_times) <= legend_max else None
                ax.plot(spec['Wavelength [nm]'], spec['Absorbance'],
                        color=color, lw=1.0, label=label)

        ax.set_xlabel('Wavelength (nm)', fontsize=11)
        ax.set_ylabel('Absorbance', fontsize=11)
        ax.tick_params(labelsize=9.5)
        ax.set_title(f'Well {w}', fontsize=10)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if has_time:
            if len(sel_times) <= legend_max:
                ax.legend(fontsize=7, frameon=False, ncol=2,
                          title='time', title_fontsize=8)
            else:
                sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
                cbar.set_label('time (s)', fontsize=9)
                cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    _style._apply_background(fig, axes, transparent)
    return fig, axes  # always a numpy array of Axes (length n, n >= 1)
