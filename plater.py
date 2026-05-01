"""Plate reader kinetics analysis: initial rates, control subtraction, MM fits.

Quick usage:
    import plater as pl

    df = pl.load_plate_reader_excel('myfile.xlsx', conditions={...})
    df_corr = pl.subtract_paired_control(df, keep_controls=True)
    rates = pl.compute_initial_rates(df_corr, t_end=75)
    pl.plot_progress_curves(df_corr, rates_df=rates, t_end_fit=75)

    mm = pl.fit_michaelis_menten(rates, exclude=[{'Substrate': 'BzP', 'S (µM)': 1250}])
    pl.plot_initial_rates(rates, mm_params_df=mm, exclude=[...])
"""

import re
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------
# defaults — override on the module to retheme all plots
# ----------------------------------------------------------------------
DEFAULT_FIGSIZE = (4.0, 3.0)         # single-panel plots
DEFAULT_FIGSIZE_WIDE = (5.8, 3.0)    # plots with side legend / inset
DEFAULT_DPI = 120

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Arial',
    'mathtext.it': 'Arial:italic',
    'mathtext.bf': 'Arial:bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
})


# ----------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------
DEFAULT_CONDITION_TAGS = ('Replicate', 'Substrate', 'S (µM)', 'E (nM)')
WELL_RE = re.compile(r'^[A-H]\d{1,2}$')


def _build_conditions_df(conditions, condition_tags):
    """Validate length and return a 'Well'-indexed DataFrame of conditions."""
    tags = list(condition_tags)
    bad = {w: vals for w, vals in conditions.items() if len(vals) != len(tags)}
    if bad:
        ex_well, ex_vals = next(iter(bad.items()))
        raise ValueError(
            f"condition values must match condition_tags length "
            f"({len(tags)} tags: {tags}); "
            f"e.g. well {ex_well!r} has {len(ex_vals)} values: {ex_vals}"
        )
    return (
        pd.DataFrame.from_dict(conditions, orient='index', columns=tags)
        .rename_axis('Well')
        .reset_index()
    )


def _resolve_sheet(filename, sheet_name):
    """Return a valid sheet name, with a helpful error if the requested one is missing."""
    xl = pd.ExcelFile(filename)
    if sheet_name is None:
        return 'Result sheet' if 'Result sheet' in xl.sheet_names else xl.sheet_names[0]
    if sheet_name not in xl.sheet_names:
        raise ValueError(
            f"sheet {sheet_name!r} not found in {filename!r}; "
            f"available sheets: {xl.sheet_names}"
        )
    return sheet_name


def _find_simple_kinetic_header(raw):
    """Index of the [..., Time [s], A1, A2, ...] header row, or None."""
    for i in range(len(raw)):
        time_col = None
        well_count = 0
        for j in range(raw.shape[1]):
            cell = raw.iat[i, j]
            if not isinstance(cell, str):
                continue
            s = cell.strip()
            if s.lower() == 'time [s]':
                time_col = j
            elif WELL_RE.match(s):
                well_count += 1
        if time_col is not None and well_count >= 1:
            return i
    return None


def _find_kinetic_scan_starts(raw):
    """Row indices where a well-block starts in column 0."""
    return [
        i for i in range(len(raw))
        if isinstance(raw.iat[i, 0], str) and WELL_RE.match(raw.iat[i, 0].strip())
    ]


def _detect_format(raw):
    """Return 'simple_kinetic' or 'kinetic_scan' from a raw (header=None) sheet."""
    if _find_simple_kinetic_header(raw) is not None:
        return 'simple_kinetic'
    if _find_kinetic_scan_starts(raw):
        return 'kinetic_scan'
    raise ValueError(
        "could not detect plate-reader data layout — expected either a "
        "'Time [s]' header row with well-ID columns (simple kinetic) or "
        "well IDs as block markers in column 0 (kinetic scan)"
    )


def _attach_conditions(df, conditions, condition_tags):
    """Filter to wells in `conditions` (if given) and merge metadata columns."""
    if conditions is None:
        return df
    df = df[df['Well'].isin(conditions.keys())].copy()
    cond_df = _build_conditions_df(conditions, condition_tags)
    return df.merge(cond_df, on='Well', how='left')


def _coerce_numeric(df, condition_tags):
    """Cast intrinsic data columns + non-string condition tags to numeric."""
    base_cols = ['Time [s]', 'Absorbance', 'Absorbance_raw',
                 'Temp [°C]', 'Wavelength [nm]']
    tag_cols = [c for c in (condition_tags or [])
                if c.lower() not in ('substrate', 'enzyme')]
    for col in (*base_cols, *tag_cols):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _parse_simple_kinetic(raw, header_row, conditions, condition_tags):
    """Long-format DataFrame from a simple-kinetic sheet (single wavelength)."""
    header = raw.iloc[header_row].astype('object').tolist()
    body = raw.iloc[header_row + 1:].copy()
    body.columns = header
    body = body.loc[:, [c for c in body.columns if not pd.isna(c)]]
    body = body.dropna(axis=1, how='all')

    well_cols = [
        c for c in body.columns
        if isinstance(c, str) and WELL_RE.match(c.strip())
    ]
    if not well_cols:
        raise ValueError("simple-kinetic header row had no well columns")
    if 'Time [s]' not in body.columns:
        raise ValueError("simple-kinetic header row had no 'Time [s]' column")

    df = body[['Time [s]', *well_cols]].copy()
    df = df.dropna(subset=['Time [s]'])
    df = df.melt(id_vars='Time [s]', var_name='Well', value_name='Absorbance')
    df = _attach_conditions(df, conditions, condition_tags)
    return _coerce_numeric(df, condition_tags).reset_index(drop=True)


def _parse_kinetic_scan(raw, well_starts, conditions, condition_tags):
    """Long-format DataFrame from a kinetic-scan sheet (spectrum × time per well)."""
    rows = []
    for idx, start in enumerate(well_starts):
        well = raw.iat[start, 0].strip()
        if conditions is not None and well not in conditions:
            continue
        end = well_starts[idx + 1] if idx + 1 < len(well_starts) else len(raw)
        block = raw.iloc[start:end]

        if str(block.iat[1, 0]).strip() != 'Time [s]':
            raise ValueError(
                f"expected 'Time [s]' as row 1 of {well} block, "
                f"got {block.iat[1, 0]!r}"
            )
        times = pd.to_numeric(block.iloc[1, 1:], errors='coerce').to_numpy()
        temps = pd.to_numeric(block.iloc[2, 1:], errors='coerce').to_numpy()

        for r in range(3, len(block)):
            wv_raw = block.iat[r, 0]
            if pd.isna(wv_raw):
                continue
            try:
                wv = float(wv_raw)
            except (TypeError, ValueError):
                continue
            abs_vals = pd.to_numeric(block.iloc[r, 1:], errors='coerce').to_numpy()
            n = min(len(times), len(abs_vals))
            for t_i in range(n):
                t = times[t_i]
                a = abs_vals[t_i]
                if pd.isna(t) or pd.isna(a):
                    continue
                rows.append({
                    'Well': well,
                    'Time [s]': t,
                    'Temp [°C]': temps[t_i] if t_i < len(temps) else np.nan,
                    'Wavelength [nm]': wv,
                    'Absorbance': a,
                })

    df = pd.DataFrame(rows)
    df = _attach_conditions(df, conditions, condition_tags)
    return _coerce_numeric(df, condition_tags).reset_index(drop=True)


def load(
    filename,
    conditions=None,
    condition_tags=DEFAULT_CONDITION_TAGS,
    sheet_name=None,
    format='auto',
    wavelength=None,
    tolerance=None,
):
    """Load a Tecan Spark plate-reader Excel export.

    Auto-detects the data layout (simple kinetic vs. kinetic scan) and the
    position of the data block within the sheet, so it tolerates the variable-
    length Tecan metadata header without manual `skiprows`.

    Parameters
    ----------
    filename : str
        Path to the .xlsx file.
    conditions : dict[str, list] | None
        Mapping {well: [tag values...]} per `condition_tags`. If None, all
        detected wells are returned with no metadata merge — useful for
        first-pass inspection of an unfamiliar file.
    condition_tags : sequence of str
        Names for the per-well metadata columns. Default:
        ('Replicate', 'Substrate', 'S (µM)', 'E (nM)'). Downstream functions
        reference these default names; if you rename, you'll need to update
        those calls accordingly.
    sheet_name : str | None
        Defaults to 'Result sheet' if present, else the workbook's first
        sheet. If a name is given that doesn't exist, the available sheets
        are listed in the error.
    format : 'auto' | 'simple_kinetic' | 'kinetic_scan'
        Override format detection.
    wavelength : float | None
        Only used for kinetic-scan files. If set, the spectrum is collapsed
        to a single wavelength via extract_wavelength, so the result drops
        directly into compute_initial_rates.
    tolerance : float | None
        Passed to extract_wavelength when `wavelength` is given.
    """
    sheet_name = _resolve_sheet(filename, sheet_name)
    raw = pd.read_excel(filename, sheet_name=sheet_name, header=None)

    if format == 'auto':
        format = _detect_format(raw)

    if format == 'simple_kinetic':
        header_row = _find_simple_kinetic_header(raw)
        if header_row is None:
            raise ValueError(
                "format='simple_kinetic' but no [Time [s], A1, ...] header row "
                f"was found in sheet {sheet_name!r}"
            )
        return _parse_simple_kinetic(raw, header_row, conditions, condition_tags)

    if format == 'kinetic_scan':
        well_starts = _find_kinetic_scan_starts(raw)
        if not well_starts:
            raise ValueError(
                "format='kinetic_scan' but no well-block markers (A1..H12) "
                f"were found in column 0 of sheet {sheet_name!r}"
            )
        df = _parse_kinetic_scan(raw, well_starts, conditions, condition_tags)
        if wavelength is not None:
            df = extract_wavelength(df, wavelength, tolerance=tolerance)
        return df

    raise ValueError(
        f"format={format!r}; expected 'auto', 'simple_kinetic', or 'kinetic_scan'"
    )


def load_plate_reader_excel(
    filename,
    conditions,
    condition_tags=DEFAULT_CONDITION_TAGS,
    sheet_name=None,
    skiprows=None,
):
    """Backward-compatible shim around `load(format='simple_kinetic')`.

    If `skiprows` is given, the legacy fixed-offset path is used; otherwise
    the data block is auto-detected. New code should call `load()` directly.
    """
    if skiprows is not None:
        df = pd.read_excel(filename,
                           sheet_name=sheet_name or 'Result sheet',
                           skiprows=skiprows, header=0)
        df = df.dropna(axis=1, how='all')
        well_cols = [c for c in df.columns
                     if isinstance(c, str) and WELL_RE.match(c)]
        df = df[['Time [s]', *well_cols]]
        df = df.melt(id_vars='Time [s]', var_name='Well', value_name='Absorbance')
        df = _attach_conditions(df, conditions, condition_tags)
        return _coerce_numeric(df, condition_tags).reset_index(drop=True)
    return load(filename, conditions, condition_tags=condition_tags,
                sheet_name=sheet_name, format='simple_kinetic')


def load_kinetic_scan_excel(
    filename,
    conditions,
    condition_tags=DEFAULT_CONDITION_TAGS,
    sheet_name=None,
):
    """Backward-compatible shim around `load(format='kinetic_scan')`."""
    return load(filename, conditions, condition_tags=condition_tags,
                sheet_name=sheet_name, format='kinetic_scan')


def extract_wavelength(scan_df, wavelength, tolerance=None):
    """Reduce a kinetic-scan DataFrame to a single-wavelength kinetic trace.

    Picks the closest available wavelength (warns if not exact). The result
    has the same schema as load_plate_reader_excel output and works directly
    with compute_initial_rates / subtract_paired_control / plot_progress_curves.

    Parameters
    ----------
    wavelength : float
        Target wavelength (nm).
    tolerance : float | None
        If set, raise if no available wavelength is within `tolerance` nm.
    """
    available = sorted(scan_df['Wavelength [nm]'].dropna().unique())
    if not available:
        raise ValueError("scan_df has no Wavelength [nm] data")
    closest = min(available, key=lambda w: abs(w - wavelength))
    if tolerance is not None and abs(closest - wavelength) > tolerance:
        raise ValueError(
            f"no wavelength within {tolerance} nm of {wavelength}; "
            f"closest available = {closest}"
        )
    if closest != wavelength:
        warnings.warn(
            f"using closest available wavelength {closest:g} nm "
            f"(requested {wavelength:g} nm)",
            stacklevel=2,
        )
    out = (
        scan_df[scan_df['Wavelength [nm]'] == closest]
        .reset_index(drop=True)
        .copy()
    )
    return out


# ----------------------------------------------------------------------
# initial rates
# ----------------------------------------------------------------------
DATA_COLUMNS = {
    'Time [s]', 'Absorbance', 'Absorbance_raw',
    'Temp [°C]', 'Wavelength [nm]',
}


def compute_initial_rates(df, t_end=100, group_by=None,
                          drop_no_enzyme=True, exclude=None,
                          direction='auto'):
    """Linear fit of A vs t over [0, t_end] for each group.

    Parameters
    ----------
    t_end : float
        Upper bound of the linear fit window in seconds.
    group_by : str | list[str] | None
        Columns that identify a single trace. Default: every column in `df`
        except the data columns (Time [s], Absorbance, Absorbance_raw,
        Temp [°C], Wavelength [nm]). This typically gives one fit per
        (well × condition).
    drop_no_enzyme : bool
        If True and an 'E (nM)' column exists in the result, drop rows where
        E (nM) == 0. Silently skipped if no such column.
    exclude : list[dict] | None
        Each dict is a condition to drop, e.g.
        [{'Replicate': 2, 'S (µM)': 625}] or [{'Well': 'G2'}].
    direction : 'auto' | 'decrease' | 'increase'
        Sign convention for `Initial Rate (Abs/s)`:
          - 'decrease' : substrate-disappearance (A goes down) → rate = -slope
          - 'increase' : product-accumulation   (A goes up)   → rate = +slope
          - 'auto'     : pick the sign whose median fitted slope across all
                         groups has greater magnitude (default)
        The `slope` column is always the raw fit slope.
    """
    if direction not in ('auto', 'decrease', 'increase'):
        raise ValueError(
            f"direction={direction!r}; expected 'auto', 'decrease', or 'increase'"
        )
    if group_by is None:
        group_by = [c for c in df.columns if c not in DATA_COLUMNS]
    elif isinstance(group_by, str):
        group_by = [group_by]
    if not group_by:
        raise ValueError(
            "no group_by columns inferred; pass group_by= explicitly"
        )

    fits = []
    for keys, group in df.groupby(group_by):
        sub = group.loc[group['Time [s]'] <= t_end, ['Time [s]', 'Absorbance']].dropna()
        if len(sub) < 2:
            continue
        t = sub['Time [s]'].to_numpy(float)
        a = sub['Absorbance'].to_numpy(float)
        res = stats.linregress(t, a)
        if not isinstance(keys, tuple):
            keys = (keys,)
        fits.append((keys, res))

    if direction == 'auto':
        slopes = np.array([res.slope for _, res in fits])
        sign = -1 if slopes.size and np.nanmedian(slopes) < 0 else 1
    else:
        sign = -1 if direction == 'decrease' else 1

    rows = []
    for keys, res in fits:
        rows.append({
            **dict(zip(group_by, keys)),
            'Initial Rate (Abs/s)': sign * res.slope,
            'slope': res.slope,
            'intercept': res.intercept,
            'r2': res.rvalue ** 2,
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
    return rates.reset_index(drop=True)


# ----------------------------------------------------------------------
# control subtraction
# ----------------------------------------------------------------------
def subtract_paired_control(df, pair_keys=('Substrate', 'S (µM)'),
                             keep_controls=False,
                             control_where=None):
    """Pair-match no-enzyme controls to +E wells by [S] (and substrate)
    and subtract the time-dependent drift of the control.

        A_corrected(t) = A_+E(t) - [A_-E(t) - A_-E(0)]

    Subtracting the t=0 baseline of the control preserves the absolute
    absorbance scale of the +E trace — only the *change* in the control
    over time is removed.

    Parameters
    ----------
    pair_keys : tuple of str
        Columns used to pair +E and -E samples. Default pairs by
        (Substrate, [S]).
    keep_controls : bool
        If True, controls are also returned (corrected, so they collapse
        to ~flat lines at A_ctrl(0)) — useful as a sanity check that the
        correction did what it should. Default False.
    control_where : dict | callable | None
        How to identify control rows within each (Substrate, [S]) group.
          - None (default): treat rows with `E (nM) == 0` as controls (and
            `E (nM) > 0` as samples)
          - dict: every {col: val} pair must match (e.g. {'Variant': 'WT'})
          - callable: receives the (full) DataFrame and returns a boolean
            Series the same length as `df`

    Returns
    -------
    DataFrame with:
        - 'Absorbance'      : drift-corrected values
        - 'Absorbance_raw'  : original values
    Rows in [S] groups with no matched control are dropped.
    """
    out = []
    keys = list(pair_keys)

    if control_where is None:
        ctrl_mask = df['E (nM)'] == 0
    elif callable(control_where):
        ctrl_mask = control_where(df).astype(bool)
    elif isinstance(control_where, dict):
        ctrl_mask = pd.Series(True, index=df.index)
        for k, v in control_where.items():
            ctrl_mask &= (df[k] == v)
    else:
        raise TypeError(
            f"control_where={control_where!r}; expected None, dict, or callable"
        )

    df = df.assign(_is_control=ctrl_mask.to_numpy())

    for _, group in df.groupby(keys):
        ctrl = group[group['_is_control']]
        if ctrl.empty:
            continue

        ctrl_mean = (
            ctrl.dropna(subset=['Time [s]', 'Absorbance'])
                .groupby('Time [s]', as_index=False)['Absorbance'].mean()
                .rename(columns={'Absorbance': 'A_ctrl'})
                .sort_values('Time [s]')
        )
        if ctrl_mean.empty:
            continue

        a0 = ctrl_mean.iloc[0]['A_ctrl']
        ctrl_mean['drift'] = ctrl_mean['A_ctrl'] - a0

        sample = group if keep_controls else group[~group['_is_control']]
        if sample.empty:
            continue

        merged = sample.merge(ctrl_mean[['Time [s]', 'drift']],
                              on='Time [s]', how='left')
        merged['Absorbance_raw'] = merged['Absorbance']
        merged['Absorbance'] = merged['Absorbance'] - merged['drift']
        out.append(merged.drop(columns=['drift', '_is_control']))

    if not out:
        return df.drop(columns='_is_control').iloc[0:0].copy()

    corrected = pd.concat(out, ignore_index=True)

    n_unmatched = corrected['Absorbance'].isna().sum()
    if n_unmatched:
        warnings.warn(
            f"{n_unmatched} sample timepoints had no matching control "
            "and were left as NaN",
            stacklevel=2,
        )

    return corrected


# ----------------------------------------------------------------------
# Michaelis-Menten
# ----------------------------------------------------------------------
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


def fit_michaelis_menten(rates_df, exclude=None):
    """Fit MM kinetics per substrate.

    If a 'Replicate' column is present (with >1 unique value), fits on the
    raw per-replicate points so Vmax_err / Km_err reflect biological + technical
    noise. Otherwise averages duplicate measurements at each [S] before fitting.

    Parameters
    ----------
    exclude : list[dict] | None
        Points to drop before fitting, e.g.
        [{'Substrate': 'pNPA', 'S (µM)': 1250}] or
        [{'Replicate': 2, 'S (µM)': 625}]
    """
    fit_input = rates_df[~_build_exclusion_mask(rates_df, exclude)]
    has_replicates = (
        'Replicate' in fit_input.columns
        and fit_input['Replicate'].nunique(dropna=True) > 1
    )

    rows = []
    for substrate, sub in fit_input.groupby('Substrate'):
        sub = sub[['S (µM)', 'Initial Rate (Abs/s)']].dropna()
        if has_replicates:
            ordered = sub.sort_values('S (µM)')
            S = ordered['S (µM)'].to_numpy(float)
            v = ordered['Initial Rate (Abs/s)'].to_numpy(float)
            n_S = ordered['S (µM)'].nunique()
        else:
            agg = (
                sub.groupby('S (µM)', as_index=False)['Initial Rate (Abs/s)']
                   .mean()
                   .sort_values('S (µM)')
            )
            S = agg['S (µM)'].to_numpy(float)
            v = agg['Initial Rate (Abs/s)'].to_numpy(float)
            n_S = len(S)
        if n_S < 3:
            continue

        p0 = [np.nanmax(v),
              np.nanmedian(S[S > 0]) if np.any(S > 0) else 1.0]
        try:
            popt, pcov = curve_fit(
                michaelis_menten, S, v,
                p0=p0, bounds=(0, np.inf), maxfev=10000,
            )
        except Exception:
            continue
        perr = np.sqrt(np.diag(pcov))
        rows.append({
            'Substrate': substrate,
            'Vmax (Abs/s)': popt[0],
            'Km (µM)': popt[1],
            'Vmax_err': perr[0],
            'Km_err': perr[1],
            'n_points': len(S),
            'n_S': n_S,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# plotting
# ----------------------------------------------------------------------
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


def _collapse_replicates(df, condition_keys=None):
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
    agg = (
        df.dropna(subset=['Time [s]', 'Absorbance'])
          .groupby(grp_cols, as_index=False, dropna=False)['Absorbance']
          .agg(['mean', 'sem'])
          .rename(columns={'mean': 'Absorbance', 'sem': 'Absorbance_sem'})
    )
    agg['Absorbance_sem'] = agg['Absorbance_sem'].fillna(0)
    return agg, condition_keys


def plot_progress_curves(
    df,
    rates_df=None,
    show_rates=False,
    annotate_rates=False,
    color_by=None,
    hollow_where=None,
    t_end_fit=100,
    wavelength=None,
    cmap_name=None,
    cmap_range=(0.25, 1.0),
    figsize=DEFAULT_FIGSIZE_WIDE,
    dpi=DEFAULT_DPI,
    show_inset='auto',
    collapse_replicates='auto',
    clip_y_to_non_hollow=False,
):
    """A vs t per well (or per condition, if replicates are pooled).

    Inset shows the linear fit range [0, t_end_fit]. Traces matching
    `hollow_where` are drawn with hollow markers / dashed lines and skipped
    from fit overlays.

    Parameters
    ----------
    rates_df : DataFrame | None
        Pre-computed rates (from compute_initial_rates) to overlay as linear
        fits. In collapse mode, fits are recomputed from the collapsed mean
        trace so they match the displayed line.
    show_rates : bool
        If True and rates_df is None, compute_initial_rates is run internally
        with t_end=t_end_fit and the fits are overlaid.
    annotate_rates : False | True | 'legend' | 'lines' | 'both'
        Where to display the per-trace initial-rate value:
          - False : no annotation
          - True / 'legend' : append the rate (Abs/s) to each legend entry
            (mean rate per `color_by` level)
          - 'lines' : place labels directly on each fit line, with collision-
            resistant positioning via `adjustText`
          - 'both' : both
        'lines'/'both' require the optional `adjustText` package.
    collapse_replicates : 'auto' | bool
        If True, average traces across Replicate per (condition × Time) and
        draw one mean line + shaded SEM band per condition. 'auto' enables
        this whenever a 'Replicate' column with >1 unique value is present.
    show_inset : 'auto' | bool
        If True, draw a side inset of the linear fit range. 'auto' draws it
        only when t_end_fit is meaningfully shorter than the full trace.
    clip_y_to_non_hollow : bool
        If True (and hollow_where is set), set y-axis limits from the
        non-hollow traces only — useful when controls are flat near A0 and
        would otherwise compress the active dynamic range.
    wavelength : float | None
        Probe wavelength (nm) shown in the y-axis label. If None, auto-
        detected from a single-valued 'Wavelength [nm]' column in `df`.
    color_by : str | None
        Column to color traces by. Numeric → sequential colormap (default
        'Blues'); categorical → qualitative ('tab10'). Default: 'S (µM)' if
        present, else 'Well' (per-well mode) / first condition key (collapsed).
    hollow_where : dict | None
        Wells/conditions where every {col: val} pair matches are drawn with
        hollow markers (per-well mode) or a dashed line (collapsed mode), and
        skipped from fit overlays. Useful for highlighting controls
        (e.g. {'E (nM)': 0}).
    cmap_name : str | None
        Override the auto-selected colormap.
    """
    if collapse_replicates == 'auto':
        collapse_replicates = (
            'Replicate' in df.columns
            and df['Replicate'].nunique(dropna=True) > 1
        )

    if collapse_replicates:
        df_plot, condition_keys = _collapse_replicates(df)
        group_keys = condition_keys
        if show_rates or rates_df is not None:
            rates_df = compute_initial_rates(
                df_plot.drop(columns='Absorbance_sem', errors='ignore'),
                t_end=t_end_fit, group_by=condition_keys,
                drop_no_enzyme=False,
            )
    else:
        df_plot = df
        group_keys = 'Well'
        if show_rates and rates_df is None:
            rates_df = compute_initial_rates(
                df, t_end=t_end_fit, drop_no_enzyme=False,
            )

    if color_by is None:
        if 'S (µM)' in df_plot.columns:
            color_by = 'S (µM)'
        elif not collapse_replicates and 'Well' in df_plot.columns:
            color_by = 'Well'
        else:
            color_by = group_keys[0] if isinstance(group_keys, list) else group_keys
    if color_by not in df_plot.columns:
        raise KeyError(f"color_by={color_by!r} not in df columns: {list(df_plot.columns)}")

    is_numeric = pd.api.types.is_numeric_dtype(df_plot[color_by])
    levels_raw = df_plot[color_by].dropna().unique()
    if is_numeric:
        levels = sorted(levels_raw)
        cmap = plt.get_cmap(cmap_name or 'Blues')
        lo, hi = cmap_range
        color_map = {
            v: cmap(lo + (hi - lo) * i / max(1, len(levels) - 1))
            for i, v in enumerate(levels)
        }
        fmt_label = lambda v: f'{v:g}'
    else:
        levels = list(levels_raw)
        cmap = plt.get_cmap(cmap_name or 'tab10')
        color_map = {v: cmap(i % cmap.N) for i, v in enumerate(levels)}
        fmt_label = str

    if show_inset == 'auto':
        t_max = df_plot['Time [s]'].dropna().max()
        show_inset = bool(pd.notna(t_max) and t_end_fit < 0.6 * t_max)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    axins = ax.inset_axes([1.1, 0.55, 0.45, 0.42]) if show_inset else None

    has_hollow = False
    for _, group in df_plot.groupby(group_keys):
        g = group.dropna(subset=['Time [s]', 'Absorbance']).sort_values('Time [s]')
        if g.empty:
            continue
        c_val = group[color_by].iloc[0]
        c = color_map.get(c_val, 'gray')

        hollow = bool(hollow_where) and _row_matches(group.iloc[0], hollow_where)
        has_hollow = has_hollow or hollow

        if collapse_replicates:
            t_arr = g['Time [s]'].to_numpy(float)
            a_arr = g['Absorbance'].to_numpy(float)
            sem_arr = g['Absorbance_sem'].to_numpy(float)
            line_kw = dict(color=c, lw=1.4,
                           ls='--' if hollow else '-',
                           alpha=0.6 if hollow else 0.95)
            band_kw = dict(color=c, alpha=0.12 if hollow else 0.18, lw=0)
            ax.plot(t_arr, a_arr, **line_kw)
            ax.fill_between(t_arr, a_arr - sem_arr, a_arr + sem_arr, **band_kw)
            if axins is not None:
                m = t_arr <= t_end_fit
                axins.plot(t_arr[m], a_arr[m], **line_kw)
                axins.fill_between(t_arr[m],
                                   a_arr[m] - sem_arr[m],
                                   a_arr[m] + sem_arr[m], **band_kw)
        else:
            if hollow:
                main_kw = dict(facecolors='none', edgecolors=c, s=10,
                               alpha=0.8, linewidths=0.7)
                ins_kw = dict(facecolors='none', edgecolors=c, s=6,
                              alpha=0.8, linewidths=0.5)
            else:
                main_kw = dict(color=c, s=10, alpha=0.7)
                ins_kw = dict(color=c, s=6, alpha=0.7)

            ax.scatter(g['Time [s]'], g['Absorbance'], **main_kw)
            if axins is not None:
                g_in = g[g['Time [s]'] <= t_end_fit]
                axins.scatter(g_in['Time [s]'], g_in['Absorbance'], **ins_kw)

    if clip_y_to_non_hollow and hollow_where:
        active_vals = []
        for _, group in df_plot.groupby(group_keys):
            if _row_matches(group.iloc[0], hollow_where):
                continue
            active_vals.append(group['Absorbance'].dropna())
        if active_vals:
            v = pd.concat(active_vals)
            margin = 0.05 * (v.max() - v.min() or 1.0)
            ax.set_ylim(v.min() - margin, v.max() + margin)
            if axins is not None:
                axins.set_ylim(v.min() - margin, v.max() + margin)

    line_label_data = []  # for adjustText placement
    if rates_df is not None and len(rates_df):
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        for _, row in rates_df.iterrows():
            if hollow_where and _row_matches(row, hollow_where):
                continue
            m, b = row['slope'], row['intercept']
            t_candidates = [xmax]
            if m < 0:
                t_candidates.append((ymin - b) / m)
            elif m > 0:
                t_candidates.append((ymax - b) / m)
            t_exit = min(t for t in t_candidates if t > 0)
            t_fit_main = np.array([0, t_exit])
            ax.plot(t_fit_main, m * t_fit_main + b,
                    color='k', lw=1.0, ls='--', alpha=0.8, zorder=10)
            if axins is not None:
                t_fit_ins = np.array([0, min(t_exit, t_end_fit)])
                axins.plot(t_fit_ins, m * t_fit_ins + b,
                           color='k', lw=1.0, ls='--', alpha=0.9, zorder=10)
            line_label_data.append((row, m, b, t_exit))


    if wavelength is None and 'Wavelength [nm]' in df.columns:
        unique_wl = df['Wavelength [nm]'].dropna().unique()
        if len(unique_wl) == 1:
            wavelength = float(unique_wl[0])

    ylabel = (
        f'Absorbance ({wavelength:g} nm)' if wavelength is not None
        else 'Absorbance'
    )
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=9.5)
    if axins is not None:
        axins.set_xlim(0, t_end_fit)
        axins.tick_params(labelsize=8)
        axins.set_title(f'linear fit range (0–{t_end_fit:g} s)',
                        fontsize=9, style='italic', pad=4)

    annotate_modes = _resolve_annotate_modes(annotate_rates)

    rate_by_level = {}
    if 'legend' in annotate_modes and rates_df is not None and len(rates_df) \
            and color_by in rates_df.columns:
        rate_by_level = (
            rates_df.groupby(color_by)['Initial Rate (Abs/s)']
                    .mean().to_dict()
        )

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
            rate = row.get('Initial Rate (Abs/s)', -m)
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

    def _legend_label(v):
        base = fmt_label(v)
        if v in rate_by_level:
            return f'{base}  ({rate_by_level[v]:.2e} Abs/s)'
        return base

    line_marker_kw = (
        dict(marker='', linestyle='-', linewidth=2.0)
        if collapse_replicates
        else dict(marker='o', linestyle='', markersize=6)
    )

    if has_hollow:
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

    handles = [
        mpl.lines.Line2D([0], [0], color=color_map[v],
                         label=_legend_label(v), **line_marker_kw)
        for v in levels
    ]
    ax.legend(handles=handles, title=color_by, loc='upper left',
              bbox_to_anchor=color_anchor, frameon=False,
              fontsize=9, title_fontsize=10)

    fig.subplots_adjust(left=0.10, right=0.60, top=0.92, bottom=0.16)
    return fig, ax, axins


def plot_initial_rates(
    rates_df,
    x_col='S (µM)',
    group_col='Substrate',
    y_col='Initial Rate (Abs/s)',
    mm_params_df=None,
    exclude=None,
    title=None,
    fit_color=None,
    figsize=DEFAULT_FIGSIZE,
    dpi=DEFAULT_DPI,
):
    """Scatter rates vs `x_col` with optional MM fit overlay.

    For an MM titration: leave defaults (x_col='S (µM)', group_col='Substrate').
    For other numeric x: set x_col; pass mm_params_df=None to skip the fit.
    Excluded points are drawn as X markers but not used in fits.

    If a 'Replicate' column is present with >1 unique value, individual replicate
    points are shown as faint gray dots and the mean ± SEM (per [S], per group)
    is overlaid as black points with error bars.

    With multiple substrates / groups in `mm_params_df`, each gets its own
    `tab10` color (data points colored to match) and its Km/Vmax show up in
    the legend rather than as a corner annotation.

    Parameters
    ----------
    fit_color : str | None
        If None and there are multiple groups, each MM fit gets its own
        `tab10` color. If set, all fits use this single color.
    """
    if x_col not in rates_df.columns:
        raise KeyError(f"x_col={x_col!r} not in rates_df: {list(rates_df.columns)}")

    excluded = _build_exclusion_mask(rates_df, exclude)
    incl = rates_df[~excluded]
    excl = rates_df[excluded]

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
    else:
        group_colors = {}

    def _color_for(grp):
        if fit_color is not None:
            return fit_color
        return group_colors.get(grp, 'k')

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if has_groups:
        for grp, sub_incl in incl.groupby(group_col):
            c = _color_for(grp)
            if has_replicates:
                ax.scatter(sub_incl[x_col], sub_incl[y_col],
                           s=10, marker='o', color=c,
                           alpha=0.30, edgecolors='none', zorder=2)
                agg = (
                    sub_incl.dropna(subset=[x_col, y_col])
                            .groupby(x_col)[y_col]
                            .agg(['mean', 'sem']).reset_index()
                )
                agg['sem'] = agg['sem'].fillna(0)
                ax.errorbar(agg[x_col], agg['mean'], yerr=agg['sem'],
                            fmt='o', markersize=4.5, color=c,
                            ecolor=c, elinewidth=0.9, capsize=2.5, capthick=0.9,
                            linestyle='none', zorder=3)
            else:
                ax.scatter(sub_incl[x_col], sub_incl[y_col],
                           s=18, marker='o', color=c, zorder=3)
    else:
        if has_replicates:
            ax.scatter(incl[x_col], incl[y_col],
                       s=10, marker='o', facecolors='lightgray',
                       edgecolors='none', alpha=0.85,
                       label='replicates', zorder=2)
            agg_keys = [x_col]
            agg = (
                incl.dropna(subset=[x_col, y_col])
                    .groupby(agg_keys)[y_col]
                    .agg(['mean', 'sem']).reset_index()
            )
            agg['sem'] = agg['sem'].fillna(0)
            ax.errorbar(agg[x_col], agg['mean'], yerr=agg['sem'],
                        fmt='o', markersize=4.5, color='k',
                        ecolor='k', elinewidth=0.9, capsize=2.5, capthick=0.9,
                        linestyle='none', label='mean ± SEM', zorder=3)
        else:
            ax.scatter(incl[x_col], incl[y_col],
                       s=18, marker='o', c='k', label='data', zorder=3)
    if len(excl):
        ax.scatter(excl[x_col], excl[y_col],
                   s=30, marker='x', c='k', linewidths=1.3,
                   label='excluded', zorder=4)

    fit_handles = []
    if mm_params_df is not None and len(mm_params_df):
        if group_col not in rates_df.columns:
            raise KeyError(
                f"group_col={group_col!r} not in rates_df; cannot overlay MM fits"
            )
        for _, row in mm_params_df.iterrows():
            grp = row[group_col]
            c = _color_for(grp)
            x_max = rates_df.loc[rates_df[group_col] == grp, x_col].max()
            S_fit = np.linspace(0, x_max, 200)
            v_fit = michaelis_menten(S_fit, row['Vmax (Abs/s)'], row['Km (µM)'])
            label = (
                f"{grp}: $K_M$={row['Km (µM)']:.0f}±{row['Km_err']:.0f} µM, "
                f"$V_{{max}}$={row['Vmax (Abs/s)']:.2e} Abs/s"
            )
            line, = ax.plot(S_fit, v_fit, color=c, lw=1.5,
                            label=label, zorder=2)
            fit_handles.append(line)

    if title:
        ax.set_title(title)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.margins(x=0.03, y=0.05)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    return fig, ax


def plot_rates_categorical(
    rates_df,
    x_col,
    y_col='Initial Rate (Abs/s)',
    order=None,
    point_color='k',
    mean_color='red',
    jitter=0.15,
    annotate_means=True,
    figsize=DEFAULT_FIGSIZE,
    dpi=DEFAULT_DPI,
    ax=None,
):
    """Strip plot of rates by a categorical column, with mean bars per group.

    Each replicate is shown as a jittered point; horizontal red bar marks
    the mean, with SEM whiskers when n > 1. Useful for comparing
    variants/conditions at a single [S].

    Parameters
    ----------
    x_col : str
        Categorical column to group by (e.g. 'Enzyme').
    order : list | None
        Optional explicit ordering of categories along the x-axis.
    annotate_means : bool
        If True, write the mean rate value above each column.
    """
    if x_col not in rates_df.columns:
        raise KeyError(f"x_col={x_col!r} not in rates_df: {list(rates_df.columns)}")
    if y_col not in rates_df.columns:
        raise KeyError(f"y_col={y_col!r} not in rates_df: {list(rates_df.columns)}")

    cats = list(order) if order is not None else list(
        rates_df[x_col].dropna().unique()
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    rng = np.random.default_rng(0)
    for i, c in enumerate(cats):
        sub = rates_df[rates_df[x_col] == c]
        vals = sub[y_col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        x_jitter = i + (rng.random(len(vals)) - 0.5) * 2 * jitter
        ax.scatter(x_jitter, vals, s=24, c=point_color,
                   alpha=0.75, zorder=3, edgecolors='none')
        mean_val = float(np.mean(vals))
        ax.hlines(mean_val, i - 0.25, i + 0.25,
                  colors=mean_color, lw=2, zorder=4)
        sem_val = 0.0
        if len(vals) > 1:
            sem_val = float(stats.sem(vals))
            ax.errorbar(i, mean_val, yerr=sem_val,
                        fmt='none', ecolor=mean_color,
                        elinewidth=1.0, capsize=3, capthick=1.0,
                        zorder=2)
        if annotate_means:
            top_y = max(float(np.max(vals)), mean_val + sem_val)
            ax.annotate(f'{mean_val:.2e}',
                        xy=(i, top_y),
                        xytext=(0, 7),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8, color='k', zorder=5)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.margins(y=0.20 if annotate_means else 0.15)
    fig.tight_layout()
    return fig, ax


def plot_spectra(
    scan_df,
    wells=None,
    n_timepoints=None,
    cmap_name='viridis',
    figsize_per_panel=DEFAULT_FIGSIZE,
    dpi=DEFAULT_DPI,
    sharey=True,
    legend_max=12,
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

    for ax, w in zip(axes, wells):
        sub = scan_df[scan_df['Well'] == w]
        times = np.array(sorted(sub['Time [s]'].dropna().unique()))
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

        if len(sel_times) <= legend_max:
            ax.legend(fontsize=7, frameon=False, ncol=2,
                      title='time', title_fontsize=8)
        else:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
            cbar.set_label('time (s)', fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig, axes  # always a numpy array of Axes (length n, n >= 1)
