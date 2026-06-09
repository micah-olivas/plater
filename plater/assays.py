"""plater.assays: colorimetric protein assays (BCA 4PL standard curve + back-calculation)."""

import re

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from . import _style
from ._common import _fourPL, _inv_fourPL


def _trailing_unit(col_name):
    """Return the trailing ' (...)' unit fragment of a column name, or '' if absent."""
    m = re.search(r'\s*\([^)]+\)\s*$', col_name)
    return m.group(0).rstrip() if m else ''


def fit_bca_standard(stds_df, conc_col='[Std] (mg/mL)', value_col='Absorbance',
                     blank_subtract=True):
    """Fit a 4-parameter logistic to a BCA BSA standard curve.

    Fits on raw replicate-level points (R² is reported on the same).

    Parameters
    ----------
    stds_df : DataFrame
        Long-form standards, one row per well: a `conc_col` (BSA concentration;
        rows where this is 0 are treated as blanks) and `value_col` (absorbance).
    blank_subtract : bool
        If True, subtract the mean absorbance of zero-standard rows before
        fitting. The corrected values are not written back to `stds_df`.

    Returns
    -------
    dict
        4PL params (a, b, c, d), R², blank, conc_lo / conc_hi (curve range),
        and the column names used. Pass to `back_calculate_bca` /
        `pick_bca_dilution`.
    """
    sub = stds_df[[conc_col, value_col]].dropna()
    if blank_subtract:
        zero_mask = sub[conc_col] == 0
        blank = float(sub.loc[zero_mask, value_col].mean()) if zero_mask.any() else 0.0
    else:
        blank = 0.0
    x = sub[conc_col].to_numpy(float)
    y = sub[value_col].to_numpy(float) - blank

    nonzero = x[x > 0]
    if len(np.unique(nonzero)) < 3:
        raise ValueError(
            f'4PL fit needs ≥3 distinct non-zero standard concentrations; '
            f'got {len(np.unique(nonzero))}'
        )

    p0 = [0.0, 1.0, float(np.median(nonzero)), float(y.max())]
    bounds = (
        [-np.inf, 0.1, 1e-6, 0.0],
        [np.inf, 5.0, float(nonzero.max()) * 100, np.inf],
    )
    popt, _ = curve_fit(_fourPL, x, y, p0=p0, bounds=bounds, maxfev=20000)
    a, b, c, d = popt
    yhat = _fourPL(x, *popt)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return {
        'a': float(a), 'b': float(b), 'c': float(c), 'd': float(d),
        'r2': r2, 'blank': blank,
        'conc_lo': float(nonzero.min()),
        'conc_hi': float(nonzero.max()),
        'conc_col': conc_col, 'value_col': value_col,
    }


def back_calculate_bca(samples_df, fit, dilution_col='Dilution factor',
                       value_col=None, out_prefix='Stock'):
    """Back-calculate sample stock concentrations from a BCA 4PL fit.

    Returns a copy of `samples_df` with three new columns:
      - 'In-well <unit>' : interpolated concentration in the well
        (NaN if the absorbance falls outside the fit's asymptotes).
      - '<out_prefix> <unit>' : in-well × `dilution_col`. Output unit
        matches the standards' concentration column — no unit conversion
        is applied.
      - 'In Range' : True if in-well lies between `conc_lo` and `conc_hi`.

    `dilution_col` is treated as a unitless multiplier (e.g. 25 for 1:25).
    """
    value_col = value_col or fit['value_col']
    sub = samples_df.copy()
    a_corr = sub[value_col].to_numpy(float) - fit['blank']
    in_well = _inv_fourPL(a_corr, fit['a'], fit['b'], fit['c'], fit['d'])

    unit = _trailing_unit(fit['conc_col'])
    in_well_col = f'In-well{unit}'
    stock_col = f'{out_prefix}{unit}'

    sub[in_well_col] = in_well
    sub[stock_col] = in_well * sub[dilution_col].to_numpy(float)
    sub['In Range'] = (in_well >= fit['conc_lo']) & (in_well <= fit['conc_hi'])
    return sub


def pick_bca_dilution(back_df, fit, sample_col='Sample',
                      dilution_col='Dilution factor',
                      stock_col=None, in_well_col=None, flag_pct=15.0):
    """Pick the most reliable dilution per sample and check cross-dilution agreement.

    For each sample, picks the dilution whose mean in-well concentration is
    closest (in log space) to the geometric midpoint of the standard curve,
    preferring dilutions where every replicate is in-range.

    Cross-dilution agreement is the per-sample percent difference between the
    highest and lowest stock means across dilutions; samples exceeding
    `flag_pct` are tagged 'CHECK' in the 'Flag' column.
    """
    unit = _trailing_unit(fit['conc_col'])
    stock_col = stock_col or f'Stock{unit}'
    in_well_col = in_well_col or f'In-well{unit}'

    stock_mean_col = f'Stock Mean{unit}'
    stock_std_col = f'Stock Std{unit}'
    cv_col = 'CV (%)'
    diff_col = 'Diff (%)'

    agg = (
        back_df.groupby([sample_col, dilution_col], as_index=False)
               .agg(
                   in_well_mean=(in_well_col, 'mean'),
                   stock_mean=(stock_col, 'mean'),
                   stock_std=(stock_col, 'std'),
                   in_range=('In Range', 'all'),
               )
               .rename(columns={
                   'stock_mean': stock_mean_col,
                   'stock_std': stock_std_col,
               })
    )
    agg[cv_col] = 100 * agg[stock_std_col] / agg[stock_mean_col]

    target = float(np.sqrt(fit['conc_lo'] * fit['conc_hi']))
    agg['log_dist'] = np.abs(
        np.log10(agg['in_well_mean'].clip(lower=1e-12)) - np.log10(target)
    )

    picked = (
        agg.sort_values(
            [sample_col, 'in_range', 'log_dist'],
            ascending=[True, False, True],
        )
        .groupby(sample_col, as_index=False).head(1)
        .rename(columns={dilution_col: 'Used Dilution'})
        [[sample_col, 'Used Dilution', stock_mean_col, stock_std_col, cv_col]]
    )

    wide = agg.pivot(index=sample_col, columns=dilution_col, values=stock_mean_col)
    wide.columns = [f'Stock 1:{int(c)}{unit}' for c in wide.columns]
    if wide.shape[1] >= 2:
        wide[diff_col] = (
            100 * (wide.max(axis=1) - wide.min(axis=1)) / wide.mean(axis=1)
        )
        wide['Flag'] = np.where(wide[diff_col] > flag_pct, 'CHECK', '')
    else:
        wide[diff_col] = 0.0
        wide['Flag'] = ''

    return picked.merge(wide.reset_index(), on=sample_col)


def plot_bca_standard(fit, stds_df, samples_df=None, sample_col='Sample',
                      dilution_col=None, in_well_col=None, value_col=None,
                      ax=None, xlim=None, ylim=None,
                      figsize=(7.0, 3.0), dpi=_style.DEFAULT_DPI,
                      legend=True, transparent=False):
    """Plot a BCA 4PL fit with standards, optionally overlaying samples.

    Sample points are plotted at (back-calculated in-well, blank-subtracted A);
    they lie on the fit curve by construction — the overlay is a position
    check (where each sample reads vs the standard range), not a fit check.

    Parameters
    ----------
    fit : dict
        Output of `fit_bca_standard`.
    stds_df : DataFrame
        Standards used for the fit (long-form).
    samples_df : DataFrame | None
        Output of `back_calculate_bca`. Must contain the in-well column and
        the absorbance column. If None, only standards are plotted.
    dilution_col : str | None
        If given, samples get distinct markers per dilution.
    ax : matplotlib axis | None
        If None, creates a 2-panel figure (linear x | log x). If passed,
        plots one panel and respects whatever scale the axis already has.
    xlim, ylim : tuple[float, float] | None
        Axis limits (min, max) applied to each panel. Default None keeps the
        auto limits.
    transparent : bool
        If True, make the figure + axes background transparent (also on save).
        Default False (white). Ignored when `ax` is passed.
    """
    value_col = value_col or fit['value_col']
    unit = _trailing_unit(fit['conc_col'])
    in_well_col = in_well_col or f'In-well{unit}'

    sx = stds_df[fit['conc_col']].to_numpy(float)
    sy = stds_df[value_col].to_numpy(float) - fit['blank']
    a, b, c, d = fit['a'], fit['b'], fit['c'], fit['d']

    sample_names = []
    dil_to_marker = {}
    if samples_df is not None:
        sample_names = list(samples_df[sample_col].dropna().unique())
        if dilution_col is not None:
            dils = sorted(samples_df[dilution_col].dropna().unique())
            markers = ['o', '^', 's', 'D', 'v', 'P', 'X']
            dil_to_marker = {d_: markers[i % len(markers)]
                             for i, d_ in enumerate(dils)}

    def _draw(_ax, scale):
        nonzero = sx[sx > 0]
        if scale == 'log' and len(nonzero):
            xx = np.logspace(np.log10(nonzero.min()),
                             np.log10(sx.max() * 1.05), 300)
        else:
            xx = np.linspace(0, sx.max() * 1.05, 300)
        _ax.plot(xx, _fourPL(xx, a, b, c, d), 'r-', lw=1.2, zorder=1,
                 label=f"4PL (R²={fit['r2']:.3f})")
        _ax.scatter(sx, sy, s=15, c='k', alpha=0.7, zorder=2, label='BSA std')

        if samples_df is not None:
            cmap = plt.get_cmap('tab10')
            for i, name in enumerate(sample_names):
                grp = samples_df[samples_df[sample_col] == name]
                xv = grp[in_well_col].to_numpy(float)
                yv = grp[value_col].to_numpy(float) - fit['blank']
                if dilution_col is not None:
                    for dil, sub_idx in grp.groupby(dilution_col).groups.items():
                        sub = grp.loc[sub_idx]
                        _ax.scatter(
                            sub[in_well_col],
                            sub[value_col].to_numpy(float) - fit['blank'],
                            s=22, c=[cmap(i % 10)],
                            marker=dil_to_marker[dil],
                            alpha=0.85, zorder=3, edgecolors='none',
                            label=f'{name} 1:{int(dil)}',
                        )
                else:
                    _ax.scatter(xv, yv, s=22, c=[cmap(i % 10)],
                                alpha=0.85, zorder=3, edgecolors='none',
                                label=name)
        _ax.set_xscale(scale)
        if scale == 'log':
            from matplotlib.ticker import ScalarFormatter, NullFormatter
            fmt = ScalarFormatter()
            fmt.set_scientific(False)
            _ax.xaxis.set_major_formatter(fmt)
            _ax.xaxis.set_minor_formatter(NullFormatter())
        _ax.set_xlabel(fit['conc_col'])
        _ax.set_ylabel(f'{value_col} (blank-subtracted)')
        if xlim is not None:
            _ax.set_xlim(*xlim)
        if ylim is not None:
            _ax.set_ylim(*ylim)

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        _draw(axes[0], 'linear')
        _draw(axes[1], 'log')
        if legend:
            axes[1].legend(fontsize=7, loc='lower right')
        plt.tight_layout()
        _style._apply_background(fig, axes, transparent)
        return fig, axes
    else:
        _draw(ax, ax.get_xscale())
        if legend:
            ax.legend(fontsize=7, loc='lower right')
        return ax
