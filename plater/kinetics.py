"""plater.kinetics: Michaelis-Menten model and fitting."""

import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ._common import _rate_col_name


def michaelis_menten(S, Vmax, Km):
    return (Vmax * S) / (Km + S)


def _build_exclusion_mask(df, exclude):
    """Boolean mask: True where any condition dict matches the row."""
    mask = pd.Series(False, index=df.index)
    if not exclude:
        return mask
    for cond in exclude:
        m = pd.Series(True, index=df.index)
        for k, v in cond.items():
            m &= (df[k] == v)
        mask |= m
    return mask


def fit_michaelis_menten(rates_df, exclude=None, group_by='Substrate',
                         s_col='S (µM)'):
    """Fit MM kinetics per group (substrate by default; pass group_by='Enzyme'
    to compare variants on a shared substrate).

    If a 'Replicate' column is present (with >1 unique value), fits on the
    raw per-replicate points so Vmax_err / Km_err reflect biological + technical
    noise. Otherwise averages duplicate measurements at each [S] before fitting.

    Parameters
    ----------
    exclude : list[dict] | None
        Points to drop before fitting, e.g.
        [{'Substrate': 'pNPA', 'S (µM)': 1250}] or
        [{'Replicate': 2, 'S (µM)': 625}]
    group_by : str
        Column to fit per. The output uses this column name as the group key
        and should be passed as `group_col=` to `plot_initial_rates`.
    s_col : str
        Varied-concentration column to fit against (e.g. '[NAD+] (mM)' for a
        cofactor titration). The output Km column is named after its unit
        (e.g. 'Km (mM)'), so pass the same column as `x_col=` to
        `plot_initial_rates`.
    """
    if group_by not in rates_df.columns:
        raise KeyError(
            f"group_by={group_by!r} not in rates_df: {list(rates_df.columns)}"
        )
    if s_col not in rates_df.columns:
        raise KeyError(
            f"s_col={s_col!r} not in rates_df: {list(rates_df.columns)}"
        )
    unit_match = re.search(r'\(([^)]+)\)', s_col)
    km_col = f"Km ({unit_match.group(1)})" if unit_match else 'Km'

    fit_input = rates_df[~_build_exclusion_mask(rates_df, exclude)]
    has_replicates = (
        'Replicate' in fit_input.columns
        and fit_input['Replicate'].nunique(dropna=True) > 1
    )

    signal_kind = rates_df.attrs.get('signal_kind') if hasattr(rates_df, 'attrs') else None
    preferred = _rate_col_name(signal_kind)
    if preferred in rates_df.columns:
        rate_col = preferred
    else:
        rate_col = next(
            (c for c in rates_df.columns if c.startswith('Initial Rate')),
            None,
        )
    if rate_col is None or rate_col not in rates_df.columns:
        raise KeyError(
            f"no 'Initial Rate (…)' column found in rates_df: "
            f"{list(rates_df.columns)}; did you forget compute_initial_rates()?"
        )
    rate_unit = rate_col[len('Initial Rate ('):-1] if rate_col.startswith('Initial Rate (') else 'ΔAbs/s'

    rows = []
    for group_val, sub in fit_input.groupby(group_by):
        sub = sub[[s_col, rate_col]].dropna()
        if has_replicates:
            ordered = sub.sort_values(s_col)
            S = ordered[s_col].to_numpy(float)
            v = ordered[rate_col].to_numpy(float)
            n_S = ordered[s_col].nunique()
        else:
            agg = (
                sub.groupby(s_col, as_index=False)[rate_col]
                   .mean()
                   .sort_values(s_col)
            )
            S = agg[s_col].to_numpy(float)
            v = agg[rate_col].to_numpy(float)
            n_S = len(S)
        if n_S < 3:
            continue
        
        # provide initial guess:
        # vmax = max rate
        # KM = any S over 0, otherwise 1 
        p0 = [np.nanmax(v),
              np.nanmedian(S[S > 0]) if np.any(S > 0) else 1.0]
        
        # fit Michaelis-Menten
        try:
            popt, pcov = curve_fit(
                michaelis_menten, S, v,
                p0=p0,
                maxfev=10000,
            )
        except Exception:
            continue
        # Error estimates from the covariance matrix diagonal. 
        # Note that these are approximate and assume the model is correct.
        perr = np.sqrt(np.diag(pcov))
        rows.append({
            group_by: group_val,
            f'Vmax ({rate_unit})': popt[0],
            km_col: popt[1],
            'Vmax_err': perr[0],
            'Km_err': perr[1],
            'n_points': len(S),
            'n_S': n_S,
        })
    out = pd.DataFrame(rows)
    out.attrs['signal_kind'] = signal_kind
    return out
