"""plater._common: shared column/signal resolution + curve models (internal)."""

import re

import numpy as np
import pandas as pd


# Wavelength may arrive as a soft-bracket per-measurement tag ('Wavelength (nm)',
# added for single-wavelength reads) or the square-bracket scan axis
# ('Wavelength [nm]'); accept either when reading it back.
WAVELENGTH_COLS = ('Wavelength (nm)', 'Wavelength [nm]')


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


def _single_wavelength(df):
    """The frame's probe wavelength when it pins down a single value, else None.

    Reads either wavelength column form (see WAVELENGTH_COLS). Returns a float
    only when exactly one non-null wavelength is present, so it's safe to fold
    into a rate-column label.
    """
    if not hasattr(df, 'columns'):
        return None
    for col in WAVELENGTH_COLS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().unique()
            if len(vals) == 1:
                return float(vals[0])
    return None


def _rate_col_label(value_col, signal_kind, wavelength=None):
    """Initial-rate column label, wavelength-aware for absorbance.

    An explicit unit on `value_col` (e.g. '[NADH] (µM)') wins. Otherwise an
    absorbance signal becomes 'A{wavelength}' when the wavelength is known
    (e.g. 'Initial Rate (A284/s)'), falling back to 'ΔAbs'. Non-absorbance
    signals defer to `_rate_col_name`.
    """
    m = re.search(r'\(([^)]+)\)\s*$', value_col) if isinstance(value_col, str) else None
    if m:
        return f'Initial Rate ({m.group(1)}/s)'
    is_abs = (signal_kind in (None, 'absorbance')
              and value_col in (None, 'Absorbance', 'Absorbance_raw'))
    if is_abs:
        unit = f'A{wavelength:g}' if wavelength is not None else 'ΔAbs'
        return f'Initial Rate ({unit}/s)'
    return _rate_col_name(signal_kind)


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


DATA_COLUMNS = {
    'Time [s]',
    'Absorbance', 'Absorbance_raw',
    'RFU', 'RFU_raw',
    'Luminescence', 'Luminescence_raw',
    'Temp [°C]', 'Wavelength [nm]',
}


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
