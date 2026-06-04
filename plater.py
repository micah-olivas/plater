"""Plate reader kinetics analysis: initial rates, control subtraction, MM fits.

Quick usage:
    import plater as pl

    df = pl.load_plate_reader_excel('myfile.xlsx', conditions={...})
    df_corr = pl.subtract_paired_control(df, keep_controls=True)
    # or, to remove an enzyme-alone background instead of a no-enzyme drift:
    # df_corr = pl.subtract_no_substrate_baseline(df)
    rates = pl.compute_initial_rates(df_corr, t_end=75)
    pl.plot_progress_curves(df_corr, rates_df=rates, t_end_fit=75)

    mm = pl.fit_michaelis_menten(rates, exclude=[{'Substrate': 'BzP', 'S (µM)': 1250}])
    pl.plot_initial_rates(rates, mm_params_df=mm, exclude=[...])
"""

import glob
import os
import re
import textwrap
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
DEFAULT_DPI = 160

# Visual defaults for data markers in scatter-style plots (standard curves,
# initial-rate plots). Lighter face + thin black outline so points read
# clearly against fit overlays. Progress-curve points use raw series colors
# without an outline since many overlapping traces would muddy together.
POINT_EDGE_COLOR = 'black'
POINT_EDGE_WIDTH = 0.6
POINT_FACE_LIGHTEN = 0.22


def _lighten(color, amount=POINT_FACE_LIGHTEN):
    """Blend `color` toward white by `amount` ∈ [0, 1]; keeps alpha."""
    r, g, b, a = mpl.colors.to_rgba(color)
    return (r + amount * (1 - r),
            g + amount * (1 - g),
            b + amount * (1 - b),
            a)

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
    # transparent backgrounds by default (figure + axes, on screen and on save)
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
})


# ----------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------
DEFAULT_CONDITION_TAGS = ('Replicate', 'Substrate', 'S (µM)', 'E (nM)')
WELL_RE = re.compile(r'^[A-P]\d{1,2}$')


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


def _find_wavelength_scan_header(raw):
    """Index of the [Wavel., G1, G2, ...] header row, or None.

    A wavelength scan has wavelengths in column 0 (header cell starts with
    'Wavel') and well IDs across the remaining columns — one timepoint, many
    wavelengths.
    """
    for i in range(len(raw)):
        cell0 = raw.iat[i, 0]
        if not isinstance(cell0, str):
            continue
        if not cell0.strip().lower().startswith('wavel'):
            continue
        well_count = sum(
            1 for j in range(1, raw.shape[1])
            if isinstance(raw.iat[i, j], str) and WELL_RE.match(raw.iat[i, j].strip())
        )
        if well_count >= 1:
            return i
    return None


# Plate shapes the endpoint-grid detector recognizes, largest first so a 384
# plate isn't misread as the 96-well subgrid hiding in its top-left corner.
_ENDPOINT_GRID_SHAPES = (
    (16, 24, 'ABCDEFGHIJKLMNOP'),
    (8, 12, 'ABCDEFGH'),
)


def _find_endpoint_grid(raw):
    """Locate a plate-grid header row (cols labeled 1..N), or None.

    Returns ``(header_row, j0, n_rows, n_cols)`` — the row of the column
    header, the column where the '1' sits, and the plate shape (8×12 or
    16×24). The N rows below the header start with row labels A..H or A..P.
    """
    for i in range(len(raw)):
        cols = []
        for j in range(raw.shape[1]):
            v = raw.iat[i, j]
            try:
                cols.append(int(float(v)))
            except (TypeError, ValueError):
                cols.append(None)
        for n_rows, n_cols, row_alphabet in _ENDPOINT_GRID_SHAPES:
            target = list(range(1, n_cols + 1))
            for j0 in range(raw.shape[1] - n_cols + 1):
                if cols[j0:j0 + n_cols] != target:
                    continue
                if i + n_rows >= len(raw):
                    continue
                row_labels = []
                for r in range(i + 1, i + 1 + n_rows):
                    cell = raw.iat[r, j0 - 1] if j0 - 1 >= 0 else None
                    row_labels.append(
                        cell.strip() if isinstance(cell, str) else None
                    )
                if row_labels == list(row_alphabet):
                    return (i, j0, n_rows, n_cols)
    return None


def _find_endpoint_row_header(raw):
    """Index of a well-ID header row with no 'Time [s]'/'Wavel.' column, or None."""
    for i in range(len(raw)):
        well_count = 0
        has_time_or_wavel = False
        for j in range(raw.shape[1]):
            cell = raw.iat[i, j]
            if not isinstance(cell, str):
                continue
            s = cell.strip()
            if s.lower() == 'time [s]' or s.lower().startswith('wavel'):
                has_time_or_wavel = True
                break
            if WELL_RE.match(s):
                well_count += 1
        if has_time_or_wavel or well_count < 1:
            continue
        # require at least one numeric data row below
        for r in range(i + 1, len(raw)):
            for j in range(raw.shape[1]):
                v = raw.iat[r, j]
                try:
                    float(v)
                    return i
                except (TypeError, ValueError):
                    continue
        return None
    return None


def _detect_format(raw):
    """Return 'simple_kinetic', 'wavelength_scan', 'kinetic_scan', or 'endpoint'."""
    if _find_simple_kinetic_header(raw) is not None:
        return 'simple_kinetic'
    if _find_wavelength_scan_header(raw) is not None:
        return 'wavelength_scan'
    if _find_kinetic_scan_starts(raw):
        return 'kinetic_scan'
    if _find_endpoint_grid(raw) is not None:
        return 'endpoint'
    if _find_endpoint_row_header(raw) is not None:
        return 'endpoint'
    raise ValueError(
        "could not detect plate-reader data layout — expected a "
        "'Time [s]' header row with well-ID columns (simple kinetic), a "
        "'Wavel.' header row with well-ID columns (wavelength scan), "
        "well IDs as block markers in column 0 (kinetic scan), or "
        "an 8×12 or 16×24 plate grid / well-ID header row (endpoint)"
    )


def _attach_conditions(df, conditions, condition_tags):
    """Filter to wells in `conditions` (if given) and merge metadata columns."""
    if conditions is None:
        return df
    df = df[df['Well'].isin(conditions.keys())].copy()
    cond_df = _build_conditions_df(conditions, condition_tags)
    return df.merge(cond_df, on='Well', how='left')


def _coerce_numeric(df, condition_tags):
    """Cast intrinsic data columns to numeric; cast tag columns only when their
    values are entirely numeric (so string-valued tags like 'stock' survive)."""
    base_cols = ['Time [s]', 'Absorbance', 'Absorbance_raw',
                 'Temp [°C]', 'Wavelength [nm]']
    for col in base_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in (condition_tags or []):
        if col not in df.columns:
            continue
        original = df[col]
        coerced = pd.to_numeric(original, errors='coerce')
        # only adopt the numeric cast if no non-null values were lost
        if coerced.notna().equals(original.notna()):
            df[col] = coerced
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


def _parse_wavelength_scan(raw, header_row, conditions, condition_tags):
    """Long-format DataFrame from a single-timepoint wavelength scan."""
    header = raw.iloc[header_row].astype('object').tolist()
    body = raw.iloc[header_row + 1:].copy()
    body.columns = header
    body = body.loc[:, [c for c in body.columns if not pd.isna(c)]]
    body = body.dropna(axis=1, how='all')

    wv_col = header[0]
    well_cols = [
        c for c in body.columns
        if isinstance(c, str) and WELL_RE.match(c.strip())
    ]
    if not well_cols:
        raise ValueError("wavelength-scan header row had no well columns")

    df = body[[wv_col, *well_cols]].copy()
    df = df.rename(columns={wv_col: 'Wavelength [nm]'})
    df['Wavelength [nm]'] = pd.to_numeric(df['Wavelength [nm]'], errors='coerce')
    df = df.dropna(subset=['Wavelength [nm]'])
    df = df.melt(id_vars='Wavelength [nm]', var_name='Well', value_name='Absorbance')
    df = _attach_conditions(df, conditions, condition_tags)
    return _coerce_numeric(df, condition_tags).reset_index(drop=True)


def _parse_endpoint(raw, conditions, condition_tags):
    """Long-format DataFrame from a single-read endpoint sheet.

    Handles two common layouts: a plate grid (8×12 / 96-well, or 16×24 /
    384-well — rows A–H or A–P, cols 1–12 or 1–24) and a row-headered list
    (well IDs as column headers + a single data row).
    """
    grid = _find_endpoint_grid(raw)
    if grid is not None:
        i, j0, n_rows, n_cols = grid
        row_alphabet = 'ABCDEFGHIJKLMNOP'[:n_rows]
        rows = []
        for r_off, row_label in enumerate(row_alphabet):
            for c_off in range(n_cols):
                v = raw.iat[i + 1 + r_off, j0 + c_off]
                try:
                    a = float(v)
                except (TypeError, ValueError):
                    continue
                rows.append({'Well': f'{row_label}{c_off + 1}', 'Absorbance': a})
        df = pd.DataFrame(rows)
        df = _attach_conditions(df, conditions, condition_tags)
        return _coerce_numeric(df, condition_tags).reset_index(drop=True)

    header_row = _find_endpoint_row_header(raw)
    if header_row is None:
        raise ValueError("endpoint format requested but no recognizable layout found")

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
        raise ValueError("endpoint header row had no well columns")

    df = body[well_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='all')
    if df.empty:
        raise ValueError("endpoint sheet had no numeric data rows below the header")
    # collapse to one row per well: take the first non-null value
    series = df.iloc[0]
    if len(df) > 1:
        warnings.warn(
            f"endpoint sheet had {len(df)} data rows; using the first",
            stacklevel=2,
        )
    out = pd.DataFrame({'Well': series.index, 'Absorbance': series.values})
    out = _attach_conditions(out, conditions, condition_tags)
    return _coerce_numeric(out, condition_tags).reset_index(drop=True)


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
    filename=None,
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
    filename : str | None
        Path to the .xlsx file. If None, looks for a single .xlsx file in the
        current working directory and uses it (erroring if zero or multiple
        are found).
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
    format : 'auto' | 'simple_kinetic' | 'kinetic_scan' | 'wavelength_scan' | 'endpoint'
        Override format detection.
    wavelength : float | None
        Used for kinetic-scan and wavelength-scan files. If set, the spectrum
        is collapsed to a single wavelength via extract_wavelength.
    tolerance : float | None
        Passed to extract_wavelength when `wavelength` is given.
    """
    if filename is None:
        candidates = sorted(
            f for f in glob.glob('*.xlsx') if not os.path.basename(f).startswith('~$')
        )
        if not candidates:
            raise FileNotFoundError(
                "load() called with no filename and no .xlsx file found in "
                f"the current directory ({os.getcwd()!r})"
            )
        if len(candidates) > 1:
            raise ValueError(
                "load() called with no filename but multiple .xlsx files found "
                f"in {os.getcwd()!r}: {candidates}. Pass filename= explicitly."
            )
        filename = candidates[0]

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
        df = _parse_simple_kinetic(raw, header_row, conditions, condition_tags)
    elif format == 'wavelength_scan':
        header_row = _find_wavelength_scan_header(raw)
        if header_row is None:
            raise ValueError(
                "format='wavelength_scan' but no [Wavel., G1, ...] header row "
                f"was found in sheet {sheet_name!r}"
            )
        df = _parse_wavelength_scan(raw, header_row, conditions, condition_tags)
        if wavelength is not None:
            df = extract_wavelength(df, wavelength, tolerance=tolerance)
    elif format == 'endpoint':
        df = _parse_endpoint(raw, conditions, condition_tags)
    elif format == 'kinetic_scan':
        well_starts = _find_kinetic_scan_starts(raw)
        if not well_starts:
            raise ValueError(
                "format='kinetic_scan' but no well-block markers (A1..H12) "
                f"were found in column 0 of sheet {sheet_name!r}"
            )
        df = _parse_kinetic_scan(raw, well_starts, conditions, condition_tags)
        if wavelength is not None:
            df = extract_wavelength(df, wavelength, tolerance=tolerance)
    else:
        raise ValueError(
            f"format={format!r}; expected 'auto', 'simple_kinetic', "
            "'wavelength_scan', 'kinetic_scan', or 'endpoint'"
        )

    mode = _extract_measurement_mode(raw)
    if mode is not None:
        df.attrs['signal_kind'] = mode
        signal_col = SIGNAL_COL_BY_KIND[mode]
        if signal_col != 'Absorbance':
            rename = {}
            if 'Absorbance' in df.columns:
                rename['Absorbance'] = signal_col
            if 'Absorbance_raw' in df.columns:
                rename['Absorbance_raw'] = f'{signal_col}_raw'
            if rename:
                df = df.rename(columns=rename)
                df.attrs['signal_kind'] = mode

    n_wells = df['Well'].nunique() if 'Well' in df.columns else 0
    print(
        f"loaded {os.path.basename(filename)!r} "
        f"(sheet={sheet_name!r}, format={format}, "
        f"mode={mode or 'unknown'}, "
        f"{n_wells} wells, {len(df)} rows)"
    )
    return df


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


_MODE_KEYWORDS = (
    ('fluorescence', 'fluorescence'),
    ('luminescence', 'luminescence'),
    ('absorbance', 'absorbance'),
)

SIGNAL_COL_BY_KIND = {
    'absorbance': 'Absorbance',
    'fluorescence': 'RFU',
    'luminescence': 'Luminescence',
}

SIGNAL_RATE_UNIT_BY_KIND = {
    'absorbance': 'ΔAbs/s',
    'fluorescence': 'ΔRFU/s',
    'luminescence': 'ΔLum/s',
}

# Every column name the loader has ever emitted for the per-measurement value.
# Used to locate the signal column on long-form DataFrames whose attrs may be
# missing (e.g. a frame that was concat'd, filtered, or built by older code).
_VALUE_COL_CANDIDATES = ('Absorbance', 'RFU', 'Luminescence', 'Signal')


def _value_col(df):
    """Return the signal column name in a long-form DataFrame.

    Prefers the column implied by `df.attrs['signal_kind']`; falls back to
    the first known signal-column name actually present in the frame.
    """
    kind = df.attrs.get('signal_kind') if hasattr(df, 'attrs') else None
    if kind:
        col = SIGNAL_COL_BY_KIND.get(kind)
        if col and col in df.columns:
            return col
    for candidate in _VALUE_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        f"no signal column found in df (looked for {list(_VALUE_COL_CANDIDATES)}); "
        f"got columns {list(df.columns)}"
    )


def _rate_col_name(signal_kind):
    """Initial-rate column label for a given signal_kind (defaults to ΔAbs/s)."""
    unit = SIGNAL_RATE_UNIT_BY_KIND.get(signal_kind or 'absorbance', 'ΔAbs/s')
    return f'Initial Rate ({unit})'


def _extract_measurement_mode(raw, header_row=None):
    """Scan a Tecan metadata block for the read mode (returned lowercased).

    Looks for a row whose column-0 cell is 'Mode' and whose neighboring cells
    contain a recognizable read-mode keyword (Absorbance, Fluorescence,
    Luminescence). Returns None if no such row is found above `header_row`.
    """
    last_row = header_row if header_row is not None else min(len(raw), 60)
    for i in range(last_row):
        cell0 = raw.iat[i, 0] if raw.shape[1] else None
        if not isinstance(cell0, str) or cell0.strip().lower() != 'mode':
            continue
        for j in range(1, raw.shape[1]):
            cell = raw.iat[i, j]
            if not isinstance(cell, str):
                continue
            low = cell.strip().lower()
            for needle, label in _MODE_KEYWORDS:
                if needle in low:
                    return label
    return None


def _guess_signal_kind(df, threshold=10.0):
    """Best-effort read-mode for a long-form DataFrame.

    Prefers `df.attrs['signal_kind']` (set by load() from the Tecan header).
    Falls back to a value-magnitude heuristic: absorbance reads cap near 4–5
    even on saturated samples; Tecan fluorescence channels return integer
    counts in the thousands+. The (Absorbance) column name is reused for
    both modes because the loader melts the well grid into one value column.
    """
    tagged = df.attrs.get('signal_kind') if hasattr(df, 'attrs') else None
    if tagged:
        return tagged
    col = 'Absorbance_raw' if 'Absorbance_raw' in df.columns else 'Absorbance'
    if col not in df.columns:
        return 'absorbance'
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    if vals.empty:
        return 'absorbance'
    return 'fluorescence' if float(vals.abs().max()) > threshold else 'absorbance'


# ----------------------------------------------------------------------
# initial rates
# ----------------------------------------------------------------------
DATA_COLUMNS = {
    'Time [s]',
    'Absorbance', 'Absorbance_raw',
    'RFU', 'RFU_raw',
    'Luminescence', 'Luminescence_raw',
    'Temp [°C]', 'Wavelength [nm]',
}


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
        excluded = set(DATA_COLUMNS)
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
    if value_col is None:
        value_col = _value_col(df)
        rate_col = _rate_col_name(signal_kind)
    else:
        if value_col not in df.columns:
            raise KeyError(
                f"value_col={value_col!r} not in df columns: {list(df.columns)}"
            )
        unit_match = re.search(r'\(([^)]+)\)\s*$', value_col)
        unit = unit_match.group(1) if unit_match else value_col
        rate_col = f'Initial Rate ({unit}/s)'

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
        - signal column      : drift-corrected values (e.g. 'Absorbance', 'RFU')
        - '<signal>_raw'     : original values
    Rows in [S] groups with no matched control are dropped.
    """
    out = []
    keys = list(pair_keys)
    value_col = _value_col(df)
    raw_col = f'{value_col}_raw'

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
            ctrl.dropna(subset=['Time [s]', value_col])
                .groupby('Time [s]', as_index=False)[value_col].mean()
                .rename(columns={value_col: 'A_ctrl'})
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
        merged[raw_col] = merged[value_col]
        merged[value_col] = merged[value_col] - merged['drift']
        out.append(merged.drop(columns=['drift', '_is_control']))

    if not out:
        return df.drop(columns='_is_control').iloc[0:0].copy()

    corrected = pd.concat(out, ignore_index=True)
    if 'signal_kind' in getattr(df, 'attrs', {}):
        corrected.attrs['signal_kind'] = df.attrs['signal_kind']

    n_unmatched = corrected[value_col].isna().sum()
    if n_unmatched:
        warnings.warn(
            f"{n_unmatched} sample timepoints had no matching control "
            "and were left as NaN",
            stacklevel=2,
        )

    return corrected


def subtract_no_substrate_baseline(df, pair_keys=('Substrate',),
                                    keep_baseline=False,
                                    baseline_where=None):
    """Subtract the time-dependent drift of no-substrate wells from each
    +substrate trace.

        A_corrected(t) = A_+S(t) - [A_baseline(t) - A_baseline(0)]

    Use when enzyme alone (or buffer + cofactor) produces a steady
    background signal — NADH oxidation, autofluorescence drift, etc. —
    that inflates the measured initial rates. Within each Substrate group,
    the S=0 wells are averaged at every timepoint, their drift relative to
    t=0 is computed, and that drift is subtracted from the +substrate
    traces of the same Substrate.

    Thin wrapper around `subtract_paired_control` with defaults appropriate
    for no-substrate baselines (pair by Substrate, treat S=0 wells as the
    baseline). For no-enzyme controls or arbitrary control criteria, use
    `subtract_paired_control` directly.

    Parameters
    ----------
    pair_keys : tuple of str
        Columns used to pair baseline and sample rows. Default ('Substrate',)
        — every baseline well labeled with a given Substrate is matched to
        all +S wells with the same Substrate. If you have a separate
        baseline per replicate, pass ('Substrate', 'Replicate').
    keep_baseline : bool
        If True, the baseline wells are returned alongside the +substrate
        traces (drift-corrected, so they collapse to ~flat lines at
        A_baseline(0)) — useful as a sanity check.
    baseline_where : dict | callable | None
        How to identify baseline rows. Default None → rows with S (µM) == 0.
    """
    if baseline_where is None:
        baseline_where = {'S (µM)': 0}
    return subtract_paired_control(
        df,
        pair_keys=pair_keys,
        control_where=baseline_where,
        keep_controls=keep_baseline,
    )


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
                # p0=p0,
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


# ----------------------------------------------------------------------
# substrate / product standard curves
# ----------------------------------------------------------------------
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
    if 'Wavelength [nm]' in sub.columns:
        unique_wl = sub['Wavelength [nm]'].dropna().unique()
        if len(unique_wl) == 1:
            agg['Wavelength [nm]'] = float(unique_wl[0])
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


# ----------------------------------------------------------------------
# BCA assay helpers (4PL standard curve + back-calculation)
# ----------------------------------------------------------------------
def _fourPL(x, a, b, c, d):
    """4-parameter logistic. a=low asymptote, d=high asymptote, c=EC50, b=Hill slope."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return d + (a - d) / (1 + (x / c) ** b)


def _inv_fourPL(y, a, b, c, d):
    y = np.asarray(y, dtype=float)
    lo, hi = min(a, d), max(a, d)
    out = np.full_like(y, np.nan)
    m = (y > lo) & (y < hi)
    with np.errstate(divide='ignore', invalid='ignore'):
        out[m] = c * ((a - d) / (y[m] - d) - 1) ** (1.0 / b)
    return out


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
                      ax=None, figsize=(7.0, 3.0), dpi=DEFAULT_DPI,
                      legend=True):
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

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        _draw(axes[0], 'linear')
        _draw(axes[1], 'log')
        if legend:
            axes[1].legend(fontsize=7, loc='lower right')
        plt.tight_layout()
        return fig, axes
    else:
        _draw(ax, ax.get_xscale())
        if legend:
            ax.legend(fontsize=7, loc='lower right')
        return ax


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
                         fit='linear',
                         wavelength=None, title=None,
                         figsize=DEFAULT_FIGSIZE_WIDE, dpi=DEFAULT_DPI,
                         ax=None):
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
    wavelength : float | None
        Probe wavelength (nm) shown in the y-axis label. If None, auto-
        detected from a single-valued 'Wavelength [nm]' column in the input.
    """
    if fit not in ('linear', 'exponential', '4pl'):
        raise ValueError(
            f"fit={fit!r}; expected 'linear', 'exponential', or '4pl'"
        )
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

    if wavelength is None and 'Wavelength [nm]' in df.columns:
        unique_wl = df['Wavelength [nm]'].dropna().unique()
        if len(unique_wl) == 1:
            wavelength = float(unique_wl[0])

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
        face = _lighten(c)
        ax.errorbar(g[conc_col], g[value_col], yerr=g[sem_col],
                    fmt='o', markersize=6, color=c,
                    markerfacecolor=face,
                    markeredgecolor=POINT_EDGE_COLOR,
                    markeredgewidth=POINT_EDGE_WIDTH,
                    ecolor=c, elinewidth=0.9, capsize=2.5, capthick=0.9,
                    linestyle='none', zorder=3)
        data_handles.append(
            mpl.lines.Line2D([0], [0], marker='o', color=c, linestyle='',
                             markerfacecolor=face,
                             markeredgecolor=POINT_EDGE_COLOR,
                             markeredgewidth=POINT_EDGE_WIDTH,
                             markersize=6, label=str(lbl))
        )

        if not show_fit:
            continue

        x_max = max_conc if max_conc is not None else g[conc_col].max()
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
                f"{lbl} linear\n"
                f"  y = {row['slope']:.3e}·x  {sign} {abs(row['intercept']):.2f}\n"
                f"  r²={row['r2']:.3f}"
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
                f"{lbl} exponential\n"
                f"  a={row['a']:.0f}, d={row['d']:.0f}\n"
                f"  k={row['k']:.2e} /{unit}\n"
                f"  r²={row['r2']:.3f}"
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
                f"{lbl} 4PL\n"
                f"  a={row['a']:.0f}, d={row['d']:.0f}\n"
                f"  b={row['b']:.2f}, c={row['c']:.1f} {unit}\n"
                f"  r²={row['r2']:.3f}"
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
    return fig, ax, pd.DataFrame(fits)


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
    dpi=DEFAULT_DPI,
    show_inset=False,
    collapse_replicates='auto',
    clip_y_to_non_hollow=False,
    ylim=None,
    time_unit='s',
    value_col=None,
    point_alpha=None,
    point_size=None,
):
    """A vs t per well (or per condition, if replicates are pooled).

    Inset shows the linear fit range [0, t_end_fit]. Traces matching
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
          - True / 'legend' : append the rate (ΔAbs/s) to each legend entry
            (mean rate per `color_by` level)
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
    value_col : str | None
        Column to plot on the y-axis (and to fit when computing rates). If
        None, auto-detected from `df.attrs['signal_kind']` (typically RFU /
        Absorbance / Luminescence). Pass a name like '[NADH] (µM)' to plot
        a converted-concentration column from `apply_standard_curve`. The
        column header should end in '(<unit>)' so the y-axis label and rate
        units render correctly.
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
    colorbar : bool
        If True (and `color_by` is numeric), replace the right-side
        categorical legend with a discrete colorbar. The value-0 baseline,
        when present, is shown as a separated grey tile below the gradient.
    point_alpha : float | None
        Override the scatter-point alpha in per-well mode. Defaults to 0.15
        for filled points and 0.22 for hollow points. Ignored in collapsed
        (replicate-averaged) mode, which draws lines + SEM bands.
    point_size : float | None
        Override the scatter-point area (matplotlib `s`) in per-well mode.
        Defaults to 6 (inset points scale to 2/3 of this). Ignored in
        collapsed (replicate-averaged) mode, which draws lines + SEM bands.
    """
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
        sharey = bool(
            ranges
            and min(ranges) > 0
            and max(ranges) / min(ranges) <= 3.0
        )

        if figsize is None:
            facet_figsize = (n * 3.0, 3.0)
        else:
            facet_figsize = figsize

        fig, axes = plt.subplots(
            1, n,
            figsize=facet_figsize,
            dpi=dpi, sharey=sharey,
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
                label_by=label_by,
                colorbar=colorbar and is_last,
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
        return fig, axes

    fig, ax = plt.subplots(figsize=figsize or DEFAULT_FIGSIZE_WIDE, dpi=dpi)
    axins = _plot_progress_curves_on_ax(
        ax, df,
        rates_df=rates_df,
        show_rates=show_rates,
        annotate_rates=annotate_rates,
        color_by=color_by,
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
        time_unit=time_unit,
        value_col=value_col,
        point_alpha=point_alpha,
        point_size=point_size,
    )
    fig.subplots_adjust(left=0.10, right=0.60, top=0.92, bottom=0.16)
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
):
    """Draw a discrete colorbar in an inset axis to the right of `ax`.

    Numeric levels are stacked low→high from bottom to top. When
    `has_zero_baseline=True`, the 0 level is rendered as a separated grey
    tile beneath a small gap so the no-substrate baseline reads as distinct
    from the dose-response gradient.
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
            facecolor=color_map[v], edgecolor='white', lw=0.4,
        ))
        cax.text(1.35, y0 + tile_h / 2, fmt_label(v),
                 ha='left', va='center', fontsize=8,
                 clip_on=False)
        top_y = y0 + tile_h

    if has_zero:
        cax.add_patch(mpl.patches.Rectangle(
            (0, 0), 1, zero_h,
            facecolor=color_map.get(0, '0.4'), edgecolor='white', lw=0.4,
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
            s_ins = s_main * (4 / 6)  # inset points scale with the main size
            if hollow:
                main_kw = dict(facecolors='none', edgecolors=c, s=s_main,
                               alpha=a_hollow, linewidths=0.6)
                ins_kw = dict(facecolors='none', edgecolors=c, s=s_ins,
                              alpha=a_hollow, linewidths=0.5)
            else:
                main_kw = dict(color=c, s=s_main, alpha=a_solid)
                ins_kw = dict(color=c, s=s_ins, alpha=a_solid)

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
                # Can't map to legend entries — list each window on its own line.
                lines = ['Fit windows:'] + [
                    f'  {fmt_label(lvl)}: {ts / seconds_per_unit:g}–'
                    f'{te / seconds_per_unit:g} {time_unit}'
                    for lvl, (ts, te) in fit_window_by_level.items()
                ]
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

    if wavelength is None and 'Wavelength [nm]' in df.columns:
        unique_wl = df['Wavelength [nm]'].dropna().unique()
        if len(unique_wl) == 1:
            wavelength = float(unique_wl[0])

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
        ins_t_disp = t_end_fit_max / seconds_per_unit
        if isinstance(t_end_fit, dict):
            ins_title = (
                f'linear fit range (0–{ins_t_disp:.1f} {time_unit}, '
                f'per {window_by})'
            )
        else:
            ins_title = f'linear fit range (0–{ins_t_disp:.1f} {time_unit})'
        axins.set_title(ins_title, fontsize=9, style='italic', pad=4)
        if time_unit != 's':
            _apply_time_ticks(axins, time_unit)

    annotate_modes = _resolve_annotate_modes(annotate_rates)

    signal_kind = df.attrs.get('signal_kind') if hasattr(df, 'attrs') else None
    rate_col = _rate_col_name(signal_kind)
    rate_unit = SIGNAL_RATE_UNIT_BY_KIND.get(signal_kind or 'absorbance', 'ΔAbs/s')

    rate_by_level = {}
    if 'legend' in annotate_modes and rates_df is not None and len(rates_df) \
            and color_by in rates_df.columns and rate_col in rates_df.columns:
        rate_by_level = (
            rates_df.groupby(color_by)[rate_col]
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
        else dict(marker='o', linestyle='', markersize=6)
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
        _draw_discrete_colorbar(
            ax, levels, color_map,
            title=color_by,
            has_zero_baseline=has_zero_baseline_entry,
            zero_baseline_label=zero_baseline_label,
            fmt_label=fmt_label,
            bbox=(1.05, color_anchor[1] - 0.32, 0.05, 0.50),
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
    title=None,
    fit_color=None,
    figsize=None,
    dpi=DEFAULT_DPI,
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

    Excluded points are drawn as X markers but not used in fits.

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
    fit_color : str | None
        If None and there are multiple groups, each fit gets its own
        `tab10` color. If set, all fits use this single color.
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
        preferred = _rate_col_name(signal_kind)
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

    if fit == 'mm' and (mm_params_df is None or not len(mm_params_df)):
        mm_params_df = fit_michaelis_menten(
            rates_df, exclude=exclude, group_by=group_col, s_col=x_col,
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
        per_panel = figsize or DEFAULT_FIGSIZE_WIDE
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
                exclude=exclude, fit_color=fit_color, legend=False,
            )
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
        if residuals:
            return fig, main_axes, resid_axes
        return fig, main_axes

    panel_figsize = figsize or DEFAULT_FIGSIZE_WIDE
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
        exclude=exclude, fit_color=fit_color, legend=True,
    )
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
    if residuals:
        fig.align_ylabels([ax, ax_resid])
        return fig, ax, ax_resid
    return fig, ax


def _plot_initial_rates_on_ax(
    ax, ax_resid, rates_df, *,
    x_col, group_col, y_col,
    mm_params_df, fit,
    exclude=None, fit_color=None, legend=True,
):
    """Render the rates scatter + optional fit + optional residuals onto ax(es)."""
    signal_kind = rates_df.attrs.get('signal_kind') if hasattr(rates_df, 'attrs') else None
    rate_unit = SIGNAL_RATE_UNIT_BY_KIND.get(signal_kind or 'absorbance', 'ΔAbs/s')

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
        return group_colors.get(grp, plt.get_cmap('tab10')(0))

    if has_groups:
        for grp, sub_incl in incl.groupby(group_col):
            c = _color_for(grp)
            face = _lighten(c)
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
                            fmt='o', markersize=6, color=c,
                            markerfacecolor=face,
                            markeredgecolor=POINT_EDGE_COLOR,
                            markeredgewidth=POINT_EDGE_WIDTH,
                            ecolor=c, elinewidth=0.9, capsize=2.5, capthick=0.9,
                            linestyle='none', zorder=3)
            else:
                ax.scatter(sub_incl[x_col], sub_incl[y_col],
                           s=42, marker='o', facecolors=[face],
                           edgecolors=POINT_EDGE_COLOR,
                           linewidths=POINT_EDGE_WIDTH, zorder=3)
    else:
        default_c = _color_for(None)
        face = _lighten(default_c)
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
                        fmt='o', markersize=6, color=default_c,
                        markerfacecolor=face,
                        markeredgecolor=POINT_EDGE_COLOR,
                        markeredgewidth=POINT_EDGE_WIDTH,
                        ecolor=default_c, elinewidth=0.9,
                        capsize=2.5, capthick=0.9,
                        linestyle='none', label='mean ± SEM', zorder=3)
        else:
            ax.scatter(incl[x_col], incl[y_col],
                       s=42, marker='o', facecolors=[face],
                       edgecolors=POINT_EDGE_COLOR,
                       linewidths=POINT_EDGE_WIDTH,
                       label='data', zorder=3)
    if len(excl):
        ax.scatter(excl[x_col], excl[y_col],
                   s=30, marker='x', c='k', linewidths=1.3,
                   label='excluded', zorder=4)

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
            x_max = rates_df.loc[grp_mask, x_col].max()
            S_fit = np.linspace(0, x_max, 200)
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
            fit_face_by_handle[line] = _lighten(c)
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
        fit_groups = (
            incl.groupby(group_col) if group_col in incl.columns
            else [('all', incl)]
        )
        for grp, sub in fit_groups:
            sub = sub.dropna(subset=[x_col, y_col])
            if sub[x_col].nunique() < 2:
                continue
            xs = sub[x_col].to_numpy(float)
            ys = sub[y_col].to_numpy(float)
            res = stats.linregress(xs, ys)
            c = _color_for(grp) if group_col in rates_df.columns else _color_for(None)
            fit_c = c if has_groups else 'k'
            x_max = float(xs.max())
            x_fit = np.linspace(0, x_max, 200)
            y_fit = res.slope * x_fit + res.intercept
            label = (
                f"{grp}:\n    slope = {_fmt_sig(res.slope)} {slope_unit}\n"
                f"    R² = {res.rvalue ** 2:.3f}"
            )
            line, = ax.plot(x_fit, y_fit, color='black', lw=1.4, ls='-',
                            label=label, zorder=2)
            fit_handles.append(line)
            fit_face_by_handle[line] = _lighten(c)
            fit_curve_xy.append((x_fit, y_fit))
            fit_param_entries.append({
                'grp': grp, 'color': fit_c,
                'lines': [
                    f"slope = {_fmt_sig(res.slope)} {slope_unit}",
                    f"R² = {res.rvalue ** 2:.3f}",
                ],
            })
            if ax_resid is not None:
                pred = res.slope * xs + res.intercept
                resid = ys - pred
                ax_resid.scatter(xs, resid, s=14, color=c,
                                 edgecolors='none', alpha=0.85, zorder=3)
                all_resid_ys.extend(resid.tolist())

    ax.set_ylabel(y_col, fontsize=11)
    ax.tick_params(labelsize=9.5)
    ax.margins(x=0.03, y=0.05)
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
    if legend:
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
                    ls=h.get_linestyle(), marker='o', markersize=6,
                    markerfacecolor=fit_face_by_handle[h],
                    markeredgecolor=POINT_EDGE_COLOR,
                    markeredgewidth=POINT_EDGE_WIDTH,
                )
                new_handles.append(proxy)
                handler_map[proxy] = _TopLineHandler(n_lines=lbl.count('\n') + 1)
            else:
                new_handles.append(h)
        ax.legend(new_handles, labels, fontsize=8, loc='upper left',
                  bbox_to_anchor=(1.08, 1.0), borderaxespad=0.,
                  frameon=False, numpoints=1, handler_map=handler_map,
                  labelspacing=1.4)
    return param_info


def plot_rates_categorical(
    rates_df,
    x_col,
    y_col='Initial Rate (ΔAbs/s)',
    order=None,
    point_color=None,
    line_color='black',
    jitter=0.15,
    annotate=True,
    figsize=DEFAULT_FIGSIZE,
    dpi=DEFAULT_DPI,
    ax=None,
):
    """Strip plot of rates by a categorical column, with a median bar per group.

    Each replicate is shown as a jittered point (outlined lighter-`tab10`, the
    same per-category scheme as the progress-curve / rates plots); a horizontal
    black bar marks the median, with SEM whiskers when n > 1. The median bar is
    omitted for categories with a single rate. Useful for comparing
    variants/conditions at a single [S].

    Parameters
    ----------
    x_col : str
        Categorical column to group by (e.g. 'Enzyme').
    order : list | None
        Optional explicit ordering of categories along the x-axis.
    point_color : color | None
        If None (default), each category gets its own `tab10` color (lightened
        face + dark outline). Pass a single color to use it for every category.
    line_color : color
        Color of the per-category median bar and SEM whiskers.
    annotate : bool
        If True, write the median rate value above each column.
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

    cmap = plt.get_cmap('tab10')
    rng = np.random.default_rng(0)
    for i, c in enumerate(cats):
        sub = rates_df[rates_df[x_col] == c]
        vals = sub[y_col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        base_c = point_color if point_color is not None else cmap(i % cmap.N)
        if len(vals) > 1:
            x_pos = i + (rng.random(len(vals)) - 0.5) * 2 * jitter
        else:
            x_pos = np.full(len(vals), float(i))  # no jitter for a lone point
        ax.scatter(x_pos, vals, s=42, marker='o',
                   facecolors=[_lighten(base_c)],
                   edgecolors=POINT_EDGE_COLOR, linewidths=POINT_EDGE_WIDTH,
                   zorder=3)
        med_val = float(np.median(vals))
        sem_val = 0.0
        # A single rate has no spread to summarize — skip the median bar.
        if len(vals) > 1:
            ax.hlines(med_val, i - 0.25, i + 0.25,
                      colors=line_color, lw=2, zorder=4)
            sem_val = float(stats.sem(vals))
            ax.errorbar(i, med_val, yerr=sem_val,
                        fmt='none', ecolor=line_color,
                        elinewidth=1.0, capsize=3, capthick=1.0,
                        zorder=2)
        if annotate:
            top_y = max(float(np.max(vals)), med_val + sem_val)
            ax.annotate(f'{med_val:.3g}',
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
    all_vals = pd.to_numeric(rates_df[y_col], errors='coerce').dropna().to_numpy()
    if len(all_vals):
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        span = (hi - lo) or max(abs(hi), 1.0)
        # Extra headroom on top for the value annotation sitting above each point.
        ax.set_ylim(lo - 0.12 * span, hi + (0.22 if annotate else 0.12) * span)
    else:
        ax.margins(y=0.15)
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
    return fig, axes  # always a numpy array of Axes (length n, n >= 1)
