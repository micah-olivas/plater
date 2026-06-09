"""plater.standards: substrate/product standard curves and concentration conversion."""

import re
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._common import _inv_fourPL, _value_col


def compute_standard_curve(df, conc_col='S (µM)', value_col='Absorbance',
                           t_window=None, group_by=None, label=None):
    """Aggregate A vs [S] into a standard curve (mean ± SEM per [S]).

    Substrate / product standards have no enzyme, so absorbance is roughly
    flat in time and we collapse the time axis to one point per [S].

    Parameters
    ----------
    t_window : (t_min, t_max) | None
        Restrict to this time range before averaging. None uses all rows.
    group_by : str | list[str] | None
        Extra grouping columns kept alongside `conc_col` (e.g. 'Substrate').
    label : str | None
        If given, added as a 'Dataset' column — handy when concatenating
        curves from multiple experiments for comparison.
    """
    for col in (conc_col, value_col):
        if col not in df.columns:
            raise KeyError(f"{col!r} not in df: {list(df.columns)}")

    sub = df
    if t_window is not None:
        if 'Time [s]' not in sub.columns:
            raise KeyError("t_window requires a 'Time [s]' column")
        t0, t1 = t_window
        sub = sub[(sub['Time [s]'] >= t0) & (sub['Time [s]'] <= t1)]

    keys = []
    if group_by is not None:
        keys += [group_by] if isinstance(group_by, str) else list(group_by)
    keys.append(conc_col)

    sem_col = f'{value_col}_sem'
    agg = (
        sub.dropna(subset=[conc_col, value_col])
           .groupby(keys)[value_col]
           .agg(['mean', 'sem', 'count'])
           .rename(columns={'mean': value_col, 'sem': sem_col, 'count': 'n'})
           .reset_index()
    )
    agg[sem_col] = agg[sem_col].fillna(0)
    for wl_col in ('Wavelength (nm)', 'Wavelength [nm]'):
        if wl_col in sub.columns:
            unique_wl = sub[wl_col].dropna().unique()
            if len(unique_wl) == 1:
                agg[wl_col] = float(unique_wl[0])
            break
    if label is not None:
        agg['Dataset'] = label
    return agg


def fit_standard_curve(df, conc_col='S (µM)', value_col='Absorbance',
                       max_conc=None, group_by=None):
    """Linear fit A vs [S]. Slope is the extinction coefficient (per pathlength).

    Parameters
    ----------
    max_conc : float | None
        Drop points above this [S] before fitting (clip nonlinear high end).
    group_by : str | list[str] | None
        Fit one curve per group (e.g. group_by='Dataset' to compare runs).
    """
    sub = df.dropna(subset=[conc_col, value_col]).copy()
    if max_conc is not None:
        sub = sub[sub[conc_col] <= max_conc]

    if group_by is None:
        groups = [((), sub)]
        key_cols = []
    else:
        key_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        groups = [
            (k if isinstance(k, tuple) else (k,), g)
            for k, g in sub.groupby(key_cols)
        ]

    rows = []
    for keys, g in groups:
        x = g[conc_col].to_numpy(float)
        y = g[value_col].to_numpy(float)
        if len(x) < 2:
            continue
        res = stats.linregress(x, y)
        rows.append({
            **dict(zip(key_cols, keys)),
            'slope': float(res.slope),
            'slope_err': float(res.stderr),
            'intercept': float(res.intercept),
            'r2': float(res.rvalue ** 2),
            'n': int(len(x)),
        })
    return pd.DataFrame(rows)


def fit_single_point_standard(stds_df=None, *, conc=None, signal=None,
                              conc_col='S (µM)', value_col='Absorbance',
                              blank_subtract=True, intercept=None):
    """Single-point standard curve: slope = (signal − intercept) / conc.

    With one calibrator you can't fit both slope and intercept, so the
    intercept is *given* (defaulting to 0, i.e. through the origin) and the
    slope falls out of the calibrator ratio. The result is a `{slope,
    intercept}` dict drop-in compatible with `apply_standard_curve`.

    Parameters
    ----------
    stds_df : DataFrame | None
        Long-form standards with `conc_col` and `value_col`. Rows at
        `conc_col == 0` are treated as blanks; their mean is used as the
        intercept when `blank_subtract=True` (and `intercept` is not given
        explicitly). The remaining non-zero rows define the calibrator —
        if multiple unique concentrations remain, the one with the most
        replicates is used (warns).
    conc, signal : float, float
        Alternative to `stds_df`: scalar calibrator values. Pass these or
        `stds_df`, not both.
    intercept : float | None
        Explicit intercept (overrides `blank_subtract`). Defaults to 0.
    blank_subtract : bool
        If True (default) and `stds_df` has rows at `conc_col == 0`, their
        mean signal becomes the intercept.

    Examples
    --------
    >>> fit = fit_single_point_standard(conc=500, signal=0.85)
    >>> df_with_conc = apply_standard_curve(df, fit, product_name='P',
    ...                                     conc_unit='µM')

    >>> fit = fit_single_point_standard(stds_df)              # auto-blank
    >>> rates_conc = apply_standard_curve(rates_df, fit, conc_unit='µM')
    """
    if stds_df is not None:
        if conc is not None or signal is not None:
            raise ValueError(
                "pass either stds_df= or scalar conc=/signal=, not both"
            )
        sub = stds_df[[conc_col, value_col]].dropna()
        if sub.empty:
            raise ValueError(
                f"stds_df has no usable rows in columns "
                f"({conc_col!r}, {value_col!r})"
            )
        if intercept is None and blank_subtract:
            zero_rows = sub[sub[conc_col] == 0]
            if not zero_rows.empty:
                intercept = float(zero_rows[value_col].mean())
        nonzero = sub[sub[conc_col] != 0]
        if nonzero.empty:
            raise ValueError(
                f"stds_df has no non-zero {conc_col} rows to use as a "
                "calibrator"
            )
        unique_concs = nonzero[conc_col].unique()
        if len(unique_concs) > 1:
            counts = nonzero.groupby(conc_col).size()
            best = counts.sort_values(ascending=False).index[0]
            warnings.warn(
                f"stds_df has {len(unique_concs)} unique non-zero {conc_col} "
                f"values {sorted(unique_concs)}; using {best!r} (most "
                "replicates) for the single-point fit",
                stacklevel=2,
            )
            cal = nonzero[nonzero[conc_col] == best]
            conc = float(best)
        else:
            cal = nonzero
            conc = float(unique_concs[0])
        signal = float(cal[value_col].mean())
    else:
        if conc is None or signal is None:
            raise ValueError(
                "fit_single_point_standard needs either stds_df= or "
                "scalar conc=/signal="
            )
        conc = float(conc)
        signal = float(signal)

    if conc == 0:
        raise ValueError(
            "calibrator conc must be nonzero (got 0); a single-point "
            "standard needs one known nonzero concentration"
        )

    intercept = float(intercept) if intercept is not None else 0.0
    slope = (signal - intercept) / conc
    return {
        'slope': slope,
        'intercept': intercept,
        'conc': conc,
        'signal': signal,
    }


def _normalize_fit(fit):
    """Coerce a fit (DataFrame row or dict) into a (kind, params) pair."""
    if isinstance(fit, pd.DataFrame):
        if fit.empty:
            raise ValueError('fit DataFrame is empty')
        params = fit.iloc[0].to_dict()
    elif isinstance(fit, pd.Series):
        params = fit.to_dict()
    elif isinstance(fit, dict):
        params = dict(fit)
    else:
        raise TypeError(
            f"fit must be a DataFrame, Series, or dict; got {type(fit).__name__}"
        )
    kind = params.get('fit')
    if kind is None:
        if {'a', 'b', 'c', 'd'} <= params.keys():
            kind = '4pl'
        elif {'a', 'd', 'k'} <= params.keys():
            kind = 'exponential'
        elif 'slope' in params:
            kind = 'linear'
        else:
            raise KeyError(
                f"could not infer fit kind from params {list(params)}; "
                "expected slope (linear), {a,d,k} (exponential), or "
                "{a,b,c,d} (4pl)"
            )
    return kind, params


def apply_standard_curve(df, fit, product_name='P', conc_unit='µM',
                          signal_col=None):
    """Convert RFU / absorbance to product concentration via a standard-curve fit.

    Inverts the standard curve row-wise. Dispatches on what's in `df`:
      - long-form data (with a signal column from load()): adds
        '[<product>] (<unit>)' with concentrations interpolated from the fit.
      - rates DataFrame (linear fits only): adds
        'd[<product>]/dt (<unit>/<t-unit>)' = rate / slope. The intercept
        drops out of the time derivative.

    For nonlinear fits (exponential, 4pl), rate-to-concentration conversion
    isn't well-defined globally, so you must apply the fit to long-form
    signal data *first*, then re-run `compute_initial_rates` on the new
    concentration column.

    Parameters
    ----------
    fit : DataFrame | dict
        Output of `fit_standard_curve`, `fit_single_point_standard`, or a
        row of the `fits` DataFrame returned by `plot_standard_curves`. The
        fit kind is inferred from the present keys; pass `'fit'` in the dict
        to override.
    product_name : str
        Label inserted into the new column header, e.g. 'P' → '[P] (µM)'.
    conc_unit : str
        Concentration unit for the new column header. Should match the
        concentration axis of the standard-curve fit.
    signal_col : str | None
        Override signal-column auto-detection in long-form mode.
    """
    kind, params = _normalize_fit(fit)

    out = df.copy()
    if hasattr(df, 'attrs'):
        out.attrs.update(df.attrs)

    rate_cols = [c for c in out.columns if c.startswith('Initial Rate (')]
    if rate_cols:
        if kind != 'linear':
            raise ValueError(
                f"apply_standard_curve to a rates DataFrame needs a linear "
                f"fit (rate / slope); got kind={kind!r}. Apply the fit to "
                "long-form signal data first, then re-run "
                "compute_initial_rates on the converted concentration column."
            )
        slope = float(params['slope'])
        if slope == 0 or not np.isfinite(slope):
            raise ValueError(f"standard-curve slope must be nonzero and finite; got {slope}")
        for rc in rate_cols:
            unit_match = re.search(r'\(([^)]+)\)', rc)
            inner = unit_match.group(1) if unit_match else 'ΔAbs/s'
            time_unit = inner.split('/')[-1] if '/' in inner else 's'
            out[f'd[{product_name}]/dt ({conc_unit}/{time_unit})'] = out[rc] / slope
        return out

    if signal_col is None:
        signal_col = _value_col(out)
    sig = pd.to_numeric(out[signal_col], errors='coerce').to_numpy(float)
    if kind == 'linear':
        slope = float(params['slope'])
        intercept = float(params.get('intercept', 0.0))
        if slope == 0 or not np.isfinite(slope):
            raise ValueError(f"standard-curve slope must be nonzero and finite; got {slope}")
        conc = (sig - intercept) / slope
    elif kind == 'exponential':
        a, d, k = float(params['a']), float(params['d']), float(params['k'])
        # invert y = a + (d − a)(1 − exp(−k·x))  →  x = −ln(1 − (y − a)/(d − a)) / k
        ratio = (sig - a) / (d - a)
        with np.errstate(invalid='ignore', divide='ignore'):
            conc = np.where((ratio < 1) & (ratio > 0), -np.log1p(-ratio) / k, np.nan)
    else:  # '4pl'
        a, b, c, d = (float(params['a']), float(params['b']),
                      float(params['c']), float(params['d']))
        conc = _inv_fourPL(sig, a, b, c, d)

    out[f'[{product_name}] ({conc_unit})'] = conc
    return out


def adjust_stock_concentration(fits, reference, nominal_stock,
                                label_col='Dataset'):
    """Infer true stock concentration from a slope ratio against a reference.

    If your absorbance vs labeled-[S] standard curves use the same dilution
    scheme but different stocks, the slope ratio reveals how off the new
    stock is from the trusted one:

        adjusted_stock = nominal_stock × slope / slope_reference

    Parameters
    ----------
    fits : DataFrame
        Output of `fit_standard_curve` or `plot_standard_curves`. Must
        contain `label_col` and 'slope' columns.
    reference : str
        Value in `label_col` whose slope is taken as ground truth.
    nominal_stock : float | dict[str, float]
        Labeled stock concentration for the non-reference dataset(s).
        Either a single value applied to all non-reference rows, or a
        {label: conc} mapping.
    """
    out = fits.copy()
    ref = out[out[label_col] == reference]
    if ref.empty:
        raise KeyError(
            f"reference={reference!r} not in {label_col} values "
            f"{list(out[label_col].unique())}"
        )
    ref_slope = float(ref.iloc[0]['slope'])
    out['slope_ratio'] = out['slope'] / ref_slope

    if isinstance(nominal_stock, dict):
        nominal = out[label_col].map(nominal_stock)
    else:
        nominal = pd.Series(
            np.where(out[label_col] == reference, np.nan, nominal_stock),
            index=out.index,
        )
    out['nominal_stock'] = nominal
    out['adjusted_stock'] = out['nominal_stock'] * out['slope_ratio']
    return out
