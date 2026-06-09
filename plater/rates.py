"""plater.rates: initial-rate computation from progress curves."""

import numpy as np
import pandas as pd
import scipy.stats as stats

from ._common import DATA_COLUMNS, _rate_col_label, _single_wavelength, _value_col


def _resolve_group_window(t_start, t_end, window_by, group_by, keys):
    """Resolve scalar (t_start, t_end) for one group's `keys` tuple.

    Either `t_start` or `t_end` (or both) may be a dict mapping a
    `window_by` value (e.g. an enzyme name) to a scalar; the remaining
    parameter is taken as-is. If both are scalars, `window_by` is ignored.
    """
    needs_lookup = isinstance(t_start, dict) or isinstance(t_end, dict)
    key_val = None
    if needs_lookup:
        if window_by is None:
            raise ValueError(
                "dict t_start/t_end requires window_by= to name the column "
                "whose values match the dict keys"
            )
        if window_by not in group_by:
            raise ValueError(
                f"window_by={window_by!r} not in group_by={group_by}; "
                "include it in group_by or pass scalar t_start/t_end"
            )
        key_val = keys[group_by.index(window_by)]

    def _pick(spec, label):
        if isinstance(spec, dict):
            if key_val not in spec:
                raise KeyError(
                    f"{label} dict has no entry for {window_by}={key_val!r}; "
                    f"available keys: {list(spec)}"
                )
            return float(spec[key_val])
        return float(spec)

    return _pick(t_start, 't_start'), _pick(t_end, 't_end')


def compute_initial_rates(df, t_end=100, t_start=0, group_by=None,
                          drop_no_enzyme=True, exclude=None,
                          direction='auto', window_by=None,
                          value_col=None):
    """Linear fit of A vs t over [t_start, t_end] for each group.

    Parameters
    ----------
    t_end : float | dict
        Upper bound of the linear fit window in seconds. Pass a dict keyed
        by `window_by` values (e.g. ``{'EnzA': 75, 'EnzB': 200}``) to use
        a different window per reaction identity.
    t_start : float | dict
        Lower bound of the linear fit window in seconds. Useful when the
        first few timepoints are noisy (mixing artifact, lag phase, etc.).
        Also supports the per-identity dict form (see `t_end`).
    window_by : str | None
        Column name whose values match the keys of any dict `t_start` /
        `t_end`. Must be one of the `group_by` columns. Required when
        either bound is a dict; ignored otherwise.
    group_by : str | list[str] | None
        Columns that identify a single trace. Default: every column in `df`
        except the time-varying data columns (Time [s], Absorbance,
        Absorbance_raw, Temp [°C]). A constant wavelength column ('Wavelength
        (nm)' / 'Wavelength [nm]') is kept as a group key so the probe
        wavelength carries through to the output. This typically gives one fit
        per (well × condition).
    drop_no_enzyme : bool
        If True and an 'E (nM)' column exists in the result, drop rows where
        E (nM) == 0. Silently skipped if no such column.
    exclude : list[dict] | None
        Each dict is a condition to drop, e.g.
        [{'Replicate': 2, 'S (µM)': 625}] or [{'Well': 'G2'}].
    direction : 'auto' | 'decrease' | 'increase'
        Sign convention for `Initial Rate (ΔAbs/s)`:
          - 'decrease' : substrate-disappearance (A goes down) → rate = -slope
          - 'increase' : product-accumulation   (A goes up)   → rate = +slope
          - 'auto'     : pick the sign whose median fitted slope across all
                         groups has greater magnitude (default)
        The `slope` column is always the raw fit slope.

    Returns
    -------
    DataFrame with one row per fitted group. Includes 't_start_fit' and
    't_end_fit' columns recording the window actually used — these flow
    into plot_progress_curves so per-reaction windows are drawn correctly.
    """
    if direction not in ('auto', 'decrease', 'increase'):
        raise ValueError(
            f"direction={direction!r}; expected 'auto', 'decrease', or 'increase'"
        )
    if group_by is None:
        # A constant wavelength is a per-measurement identity (unlike the
        # time-varying Time/Temp/signal columns), so keep it as a group key —
        # it then travels through to the rates output without splitting groups.
        # ('Wavelength (nm)' isn't in DATA_COLUMNS; drop the scan-axis form too.)
        excluded = set(DATA_COLUMNS) - {'Wavelength [nm]'}
        if value_col is not None:
            excluded.add(value_col)
        else:
            # auto-detected value column should also be excluded
            try:
                excluded.add(_value_col(df))
            except KeyError:
                pass
        group_by = [c for c in df.columns if c not in excluded]
    elif isinstance(group_by, str):
        group_by = [group_by]
    if not group_by:
        raise ValueError(
            "no group_by columns inferred; pass group_by= explicitly"
        )

    signal_kind = df.attrs.get('signal_kind') if hasattr(df, 'attrs') else None
    wavelength = _single_wavelength(df)
    if value_col is None:
        value_col = _value_col(df)
    elif value_col not in df.columns:
        raise KeyError(
            f"value_col={value_col!r} not in df columns: {list(df.columns)}"
        )
    rate_col = _rate_col_label(value_col, signal_kind, wavelength)

    fits = []
    for keys, group in df.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ts, te = _resolve_group_window(t_start, t_end, window_by, group_by, keys)
        t_mask = (group['Time [s]'] >= ts) & (group['Time [s]'] <= te)
        sub = group.loc[t_mask, ['Time [s]', value_col]].dropna()
        if len(sub) < 2:
            continue
        t = sub['Time [s]'].to_numpy(float)
        a = sub[value_col].to_numpy(float)
        res = stats.linregress(t, a)
        fits.append((keys, res, ts, te))

    if direction == 'auto':
        slopes = np.array([res.slope for _, res, _, _ in fits])
        sign = -1 if slopes.size and np.nanmedian(slopes) < 0 else 1
    else:
        sign = -1 if direction == 'decrease' else 1

    rows = []
    for keys, res, ts, te in fits:
        rows.append({
            **dict(zip(group_by, keys)),
            rate_col: sign * res.slope,
            'slope': res.slope,
            'intercept': res.intercept,
            'r2': res.rvalue ** 2,
            't_start_fit': ts,
            't_end_fit': te,
        })
    rates = pd.DataFrame(rows)
    if rates.empty:
        return rates

    sort_cols = [c for c in ('Substrate', 'S (µM)', 'Well') if c in rates.columns]
    if sort_cols:
        rates = rates.sort_values(sort_cols)
    rates = rates.reset_index(drop=True)

    if drop_no_enzyme and 'E (nM)' in rates.columns:
        rates = rates[rates['E (nM)'] > 0]
    if exclude:
        for cond in exclude:
            mask = pd.Series(True, index=rates.index)
            for k, v in cond.items():
                if k not in rates.columns:
                    raise KeyError(
                        f"exclude key {k!r} not found; available columns: "
                        f"{list(rates.columns)}"
                    )
                mask &= (rates[k] == v)
            rates = rates[~mask]
    rates = rates.reset_index(drop=True)
    if 'signal_kind' in getattr(df, 'attrs', {}):
        rates.attrs['signal_kind'] = df.attrs['signal_kind']
    return rates
