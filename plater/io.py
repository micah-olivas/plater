"""plater.io: plate-reader file loading and format parsing."""

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

from ._common import SIGNAL_COL_BY_KIND


DEFAULT_CONDITION_TAGS = ('Replicate', 'Substrate', 'S (µM)', 'E (nM)')


WELL_RE = re.compile(r'^[A-P]\d{1,2}$')
WELL_RANGE_RE = re.compile(r'^([A-Pa-p]\d{1,2})\s*[-:]\s*([A-Pa-p]\d{1,2})$')


def _parse_well_id(s):
    """Return (row_letter, col_int) for a well ID like 'G1'."""
    m = re.match(r'^([A-Pa-p])(\d{1,2})$', s.strip())
    if not m:
        raise ValueError(f"not a well ID: {s!r}")
    return m.group(1).upper(), int(m.group(2))


def _ordered_unique(items):
    """De-duplicate while preserving first-seen order."""
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def expand_well_range(spec):
    """Expand a well range / list spec into a list of well IDs.

    Accepts a single well ('G1'), a rectangular range ('G1-G2', 'B2:D5'), a
    comma-separated list ('G1, G2, H3'), or any iterable of well IDs. A range
    is interpreted as the rectangle bounded by its two corners — matching the
    Tecan 'Part of Plate' convention, so 'B2-D5' expands to rows B–D × cols
    2–5. Useful for subsetting a load to the measured section of a plate.
    """
    if spec is None:
        return None
    if not isinstance(spec, str):
        return _ordered_unique(w.strip().upper() for w in spec)

    s = spec.strip()
    if ',' in s:
        out = []
        for part in s.split(','):
            part = part.strip()
            if part:
                out.extend(expand_well_range(part))
        return _ordered_unique(out)

    m = WELL_RANGE_RE.match(s)
    if m:
        (r1, c1), (r2, c2) = _parse_well_id(m.group(1)), _parse_well_id(m.group(2))
        rows = range(min(ord(r1), ord(r2)), max(ord(r1), ord(r2)) + 1)
        cols = range(min(c1, c2), max(c1, c2) + 1)
        return [f'{chr(r)}{c}' for r in rows for c in cols]

    r, c = _parse_well_id(s)
    return [f'{r}{c}']


_PLATE_AREA_LABELS = ('part of plate', 'plate area')


def _parse_plate_area(raw):
    """Well list from the Tecan 'Part of Plate' / 'Plate area' metadata, or None.

    These rows record the rectangular region actually measured (e.g. 'G1-G2'),
    so they pinpoint the section of interest without scanning the data block.
    """
    for i in range(min(len(raw), 60)):
        cell0 = raw.iat[i, 0] if raw.shape[1] else None
        if not isinstance(cell0, str) or cell0.strip().lower() not in _PLATE_AREA_LABELS:
            continue
        for j in range(1, raw.shape[1]):
            cell = raw.iat[i, j]
            if not isinstance(cell, str):
                continue
            s = cell.strip()
            if WELL_RANGE_RE.match(s) or WELL_RE.match(s):
                return expand_well_range(s)
    return None


def _subset_wells(df, wells, raw, drop_empty_wells):
    """Restrict a long-format frame to an explicit well set and/or drop empties.

    `wells` may be 'auto'/'plate_area' (use the Tecan metadata region), a range
    string, a list of well IDs, or None. `drop_empty_wells` additionally removes
    wells whose signal column is entirely NaN.
    """
    if 'Well' not in df.columns:
        return df

    signal_col = next(
        (c for c in ('Absorbance', *SIGNAL_COL_BY_KIND.values()) if c in df.columns),
        None,
    )

    if wells is not None:
        if isinstance(wells, str) and wells.strip().lower() in ('auto', 'plate_area'):
            target = _parse_plate_area(raw)
            if target is None:
                warnings.warn(
                    "wells='auto' but no 'Part of Plate'/'Plate area' metadata "
                    "was found; keeping all wells",
                    stacklevel=3,
                )
        else:
            target = expand_well_range(wells)
        if target is not None:
            # wells carrying actual data, so typos and empty-region requests warn
            if signal_col is not None:
                with_data = set(df.loc[df[signal_col].notna(), 'Well'])
            else:
                with_data = set(df['Well'])
            missing = [w for w in target if w not in with_data]
            if missing:
                warnings.warn(
                    f"requested wells have no data in the sheet: {missing}",
                    stacklevel=3,
                )
            df = df[df['Well'].isin(target)]

    if drop_empty_wells and signal_col is not None:
        has_data = df.groupby('Well')[signal_col].transform(lambda s: s.notna().any())
        if 'Saturated' in df.columns:
            # Saturated wells have a NaN signal (the over-range reading was
            # voided) but were genuinely measured — keep them so they stay
            # visible and flagged instead of being dropped as "empty".
            keep = has_data | df.groupby('Well')['Saturated'].transform('any')
        else:
            keep = has_data
        df = df[keep]

    return df.reset_index(drop=True)


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


def _is_simple_kinetic_header(raw, i):
    """True if row `i` is a [..., Time [s], A1, A2, ...] kinetic header."""
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
    return time_col is not None and well_count >= 1


def _find_simple_kinetic_header(raw):
    """Index of the first [..., Time [s], A1, A2, ...] header row, or None."""
    for i in range(len(raw)):
        if _is_simple_kinetic_header(raw, i):
            return i
    return None


def _table_title(raw, header_row):
    """Label naming the table above its header row, or None.

    Path-corrected reads stack several kinetic tables in one sheet, each
    introduced by a single-cell title row ('BzP', 'Pathlength corrected BzP
    [OD/cm]', …) sitting just above the 'Cycle Nr.' header.
    """
    for i in range(header_row - 1, max(-1, header_row - 4), -1):
        cell0 = raw.iat[i, 0] if raw.shape[1] else None
        if not isinstance(cell0, str) or not cell0.strip():
            continue
        rest = [raw.iat[i, j] for j in range(1, raw.shape[1])]
        if all(pd.isna(x) or (isinstance(x, str) and not x.strip()) for x in rest):
            return cell0.strip()
        return None
    return None


def _simple_kinetic_tables(raw):
    """List every stacked kinetic table as {title, header_row, end_row}.

    Each table's body runs from the row after its header until the first row
    whose column 0 is no longer a cycle number — i.e. the next title row, a
    blank, or an 'End Time' footer — so tables aren't melted into each other.
    """
    tables = []
    for i in range(len(raw)):
        if not _is_simple_kinetic_header(raw, i):
            continue
        end = len(raw)
        for r in range(i + 1, len(raw)):
            try:
                int(float(raw.iat[r, 0]))
            except (TypeError, ValueError):
                end = r
                break
        tables.append({
            'title': _table_title(raw, i),
            'header_row': i,
            'end_row': end,
        })
    return tables


def _select_table(tables, table):
    """Pick one table by name/index; return (selected, all_titles).

    `table=None` uses the sole table, or — when several are stacked — prefers a
    'pathlength corrected' table (the analysis-ready quantity for a corrected
    read), falling back to the first. A string matches a title case-insensitively
    (exact wins over substring); an int indexes the tables in sheet order.
    """
    titles = [t['title'] or f'table {i + 1}' for i, t in enumerate(tables)]

    if table is None:
        if len(tables) == 1:
            return tables[0], titles
        for t, name in zip(tables, titles):
            if 'corrected' in name.lower():
                return t, titles
        return tables[0], titles

    if isinstance(table, int) and not isinstance(table, bool):
        try:
            return tables[table], titles
        except IndexError:
            raise ValueError(
                f"table index {table} out of range; {len(tables)} tables: {titles}"
            )

    low = str(table).strip().lower()
    exact = [t for t, n in zip(tables, titles) if n.lower() == low]
    if exact:
        return exact[0], titles
    subs = [(t, n) for t, n in zip(tables, titles) if low in n.lower()]
    if len(subs) == 1:
        return subs[0][0], titles
    if not subs:
        raise ValueError(f"table {table!r} not found; available tables: {titles}")
    raise ValueError(
        f"table {table!r} is ambiguous; matches {[n for _, n in subs]}"
    )


def _find_measurements(raw):
    """List the measurement actions in a sheet as {name, start_row, end_row}.

    A Tecan read mode that stacks several measurements in one 'Result sheet'
    introduces each with a 'Mode | <read mode>' row (Fluorescence / Absorbance
    / Luminescence) followed by a 'Name | <label>' row; that action's data
    block then runs until the next action's Mode row. This lets one sheet
    holding several reads — e.g. an 'EGFP' endpoint grid plus an 'NADH' kinetic
    series — be loaded one read at a time via ``load(measurement=...)``.

    Returns [] when the sheet has no such read-mode metadata (e.g. a bare
    single-table export), in which case the whole sheet is parsed as before.
    """
    starts = []
    for i in range(len(raw)):
        cell0 = raw.iat[i, 0] if raw.shape[1] else None
        if not isinstance(cell0, str) or cell0.strip().lower() != 'mode':
            continue
        mode_val = None
        for j in range(1, raw.shape[1]):
            v = raw.iat[i, j]
            if isinstance(v, str) and v.strip():
                mode_val = v.strip().lower()
                break
        # 'Mode | Kinetic' and other non-read rows don't start a measurement
        if mode_val is None or not any(k in mode_val for k, _ in _MODE_KEYWORDS):
            continue
        name = None
        for k in range(i + 1, min(i + 8, len(raw))):
            c0 = raw.iat[k, 0]
            if isinstance(c0, str) and c0.strip().lower() == 'name':
                for j in range(1, raw.shape[1]):
                    v = raw.iat[k, j]
                    if isinstance(v, str) and v.strip():
                        name = v.strip()
                        break
                break
        starts.append({'name': name, 'start_row': i})

    for idx, b in enumerate(starts):
        b['end_row'] = (
            starts[idx + 1]['start_row'] if idx + 1 < len(starts) else len(raw)
        )
    return starts


def _measurement_names(measurements):
    """Display names for measurement blocks (fallback to positional labels)."""
    return [
        m['name'] or f'measurement {i + 1}'
        for i, m in enumerate(measurements)
    ]


def _select_measurement(measurements, measurement):
    """Pick one measurement block by name/index; return (selected, all_names).

    A string matches a measurement Name case-insensitively (exact wins over
    substring); an int indexes the measurements in sheet order.
    """
    names = _measurement_names(measurements)
    if isinstance(measurement, int) and not isinstance(measurement, bool):
        try:
            return measurements[measurement], names
        except IndexError:
            raise ValueError(
                f"measurement index {measurement} out of range; "
                f"{len(measurements)} measurements: {names}"
            )
    low = str(measurement).strip().lower()
    exact = [m for m, n in zip(measurements, names) if n.lower() == low]
    if exact:
        return exact[0], names
    subs = [(m, n) for m, n in zip(measurements, names) if low in n.lower()]
    if len(subs) == 1:
        return subs[0][0], names
    if not subs:
        raise ValueError(
            f"measurement {measurement!r} not found; available: {names}"
        )
    raise ValueError(
        f"measurement {measurement!r} is ambiguous; matches {[n for _, n in subs]}"
    )


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


_SUBREAD_RE = re.compile(r'^\d+;\d+$')


def _find_multi_read_header(raw):
    """Index of a 'Multiple Reads per Well' header row, or None.

    This Tecan layout is transposed relative to a normal endpoint: a header
    row whose column-0 label is 'Well' lists the well IDs across the remaining
    columns, and the rows below it are per-well statistics ('Mean', 'StDev')
    followed by the individual sub-read positions ('1;2', '2;1', …). It's
    recognized by that 'Well' header sitting directly above a 'Mean' row, which
    distinguishes it from the generic well-ID-header endpoint layout.
    """
    for i in range(len(raw)):
        c0 = raw.iat[i, 0] if raw.shape[1] else None
        if not isinstance(c0, str) or c0.strip().lower() != 'well':
            continue
        well_count = sum(
            1 for j in range(1, raw.shape[1])
            if isinstance(raw.iat[i, j], str) and WELL_RE.match(raw.iat[i, j].strip())
        )
        if well_count < 1:
            continue
        for r in range(i + 1, min(i + 4, len(raw))):
            cr = raw.iat[r, 0]
            if isinstance(cr, str) and cr.strip().lower() == 'mean':
                return i
    return None


def _detect_format(raw):
    """Return 'simple_kinetic', 'wavelength_scan', 'kinetic_scan', 'multi_read', or 'endpoint'."""
    if _find_simple_kinetic_header(raw) is not None:
        return 'simple_kinetic'
    if _find_wavelength_scan_header(raw) is not None:
        return 'wavelength_scan'
    if _find_kinetic_scan_starts(raw):
        return 'kinetic_scan'
    # multi_read before endpoint: its 'Well'-headered grid also matches the
    # generic endpoint row-header detector, but it's a distinct layout.
    if _find_multi_read_header(raw) is not None:
        return 'multi_read'
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
    base_cols = ['Time [s]', 'Absorbance', 'Absorbance_raw', 'Absorbance_std',
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


def _parse_simple_kinetic(raw, header_row, conditions, condition_tags, end_row=None):
    """Long-format DataFrame from one simple-kinetic table (single wavelength).

    `end_row` bounds the table body so stacked tables (e.g. the raw, test,
    reference, and pathlength-corrected blocks of a path-corrected read) don't
    bleed into each other.
    """
    header = raw.iloc[header_row].astype('object').tolist()
    body = raw.iloc[header_row + 1:end_row].copy()
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


def _parse_multi_read(raw, header_row, conditions, condition_tags, reads='mean',
                      keep_saturated=False):
    """Long-format DataFrame from a 'Multiple Reads per Well' sheet.

    The block below the 'Well' header row holds per-well 'Mean' and 'StDev'
    rows plus one row per sub-read position ('1;2', '2;1', …) — the pattern
    (2×2, 3×3, cross, circle…) just changes which grid coordinates appear, so
    every '<int>;<int>' row is collected as a sub-read. Only wells with at
    least one non-null sub-read are kept (unmeasured wells carry a 0 mean).

    reads='mean' (default) returns one row per well with the Tecan-computed
    Mean as the signal and StDev in an 'Absorbance_std' column; reads='all'
    returns one row per sub-read with a 'Read' column naming its position.

    Over-range / invalid readings (the Tecan 'OVER', 'UNDER', 'Invalid'
    sentinels written when a well saturates the detector) are dropped to NaN
    and the affected wells are warned about, so a saturated well doesn't
    silently vanish.

    keep_saturated adds a boolean 'Saturated' column flagging the over-range
    wells; combined with the protection in `_subset_wells`, those wells are
    retained (with a NaN signal) rather than dropped as empty.
    """
    if reads not in ('mean', 'all'):
        raise ValueError(f"reads={reads!r}; expected 'mean' or 'all'")

    def _num(v):
        """Numeric value, or NaN for blanks and Tecan over/under/invalid flags."""
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    header = raw.iloc[header_row].astype('object').tolist()
    well_cols = {
        j: header[j].strip()
        for j in range(1, len(header))
        if isinstance(header[j], str) and WELL_RE.match(header[j].strip())
    }
    if not well_cols:
        raise ValueError("multi-read header row had no well columns")

    mean_row = stdev_row = None
    subread_rows = []  # (position label, row index)
    for r in range(header_row + 1, len(raw)):
        lab = raw.iat[r, 0]
        if not isinstance(lab, str) or not lab.strip():
            break  # blank label → end of this read's block
        low = lab.strip().lower()
        if low == 'mean':
            mean_row = r
        elif low == 'stdev':
            stdev_row = r
        elif _SUBREAD_RE.match(lab.strip()):
            subread_rows.append((lab.strip(), r))
        else:
            break  # 'End Time' or another section

    # A well counts as measured if it has any non-blank sub-read cell (a numeric
    # value or an over-range flag); fully-blank columns are unmeasured wells.
    measured = {
        j: well for j, well in well_cols.items()
        if any(pd.notna(raw.iat[r, j]) for _, r in subread_rows)
    }

    # Warn about over-range / invalid wells (sub-read or Mean flagged non-numeric).
    saturated = sorted(
        well for j, well in measured.items()
        if any(
            isinstance(raw.iat[r, j], str) and raw.iat[r, j].strip()
            for _, r in subread_rows
        )
        or (mean_row is not None
            and isinstance(raw.iat[mean_row, j], str)
            and raw.iat[mean_row, j].strip())
    )
    if saturated:
        warnings.warn(
            f"over-range/invalid readings in {len(saturated)} well(s) "
            f"(those readings set to NaN; a well is dropped only if its mean "
            f"is invalid): {saturated}",
            stacklevel=3,
        )

    if reads == 'all':
        rows = [
            {'Well': well, 'Read': lab, 'Absorbance': _num(raw.iat[r, j])}
            for j, well in measured.items()
            for lab, r in subread_rows
            if not np.isnan(_num(raw.iat[r, j]))  # drop blanks and over-range
        ]
    else:
        rows = [
            {
                'Well': well,
                'Absorbance': raw.iat[mean_row, j] if mean_row is not None else np.nan,
                'Absorbance_std': raw.iat[stdev_row, j] if stdev_row is not None else np.nan,
            }
            for j, well in measured.items()
        ]

    if keep_saturated:
        sat_set = set(saturated)
        if reads == 'all':
            # Every saturated sub-read was filtered out above, so a fully
            # over-range well would have no rows. Add a NaN placeholder so the
            # well survives and can be flagged.
            present = {r['Well'] for r in rows}
            for j, well in measured.items():
                if well in sat_set and well not in present:
                    rows.append({'Well': well, 'Read': None, 'Absorbance': np.nan})

    df = pd.DataFrame(rows)
    if keep_saturated:
        df['Saturated'] = df['Well'].isin(sat_set)
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
    filename=None,
    conditions=None,
    condition_tags=DEFAULT_CONDITION_TAGS,
    sheet_name=None,
    format='auto',
    wavelength=None,
    tolerance=None,
    wells=None,
    drop_empty_wells=True,
    table=None,
    measurement=None,
    reads='mean',
    keep_saturated=False,
):
    """Load a Tecan Spark plate-reader Excel export.

    Auto-detects the data layout (covering simple kinetic, kinetic scan,
    wavelength scan, endpoint, and multi-read formats) and the position of the
    data block within the sheet, so it tolerates the variable-length Tecan
    metadata header without manual `skiprows`.

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
    format : 'auto' | 'simple_kinetic' | 'kinetic_scan' | 'wavelength_scan' | 'endpoint' | 'multi_read'
        Override format detection.
    wavelength : float | None
        Used for kinetic-scan and wavelength-scan files. If set, the spectrum
        is collapsed to a single wavelength via extract_wavelength.
    tolerance : float | None
        Passed to extract_wavelength when `wavelength` is given.
    wells : str | sequence of str | None
        Subset to a section of the plate. Accepts a rectangular range
        ('G1-G2', 'B2:D5'), a list of well IDs (['G1', 'G2']), or 'auto' /
        'plate_area' to use the Tecan 'Part of Plate' / 'Plate area' metadata
        region. None (default) keeps every well that survives
        `drop_empty_wells`.
    drop_empty_wells : bool
        When True (default), drop wells whose signal column is entirely empty.
        Path-corrected / partial-plate exports list all 96 well columns in the
        header but only populate the measured ones, so this keeps the result
        to the section of interest.
    table : str | int | None
        Which kinetic table to read when a sheet stacks several (a path-
        corrected read writes one table per quantity: the raw measurement, the
        pathlength test/reference wavelengths, the 'Pathlength corrected …
        [OD/cm]' result, and the pathlength itself). A string matches the table
        title case-insensitively (e.g. 'corrected'); an int indexes them in
        sheet order. None (default) uses the sole table, or — when several are
        present — the pathlength-corrected one if found, else the first. The
        chosen title and the full list are recorded in df.attrs['table'] and
        df.attrs['available_tables'].
    measurement : str | int | None
        Which read to load when a sheet stacks several Tecan measurement
        actions — e.g. a script that records an 'EGFP' endpoint read and an
        'NADH' kinetic series in the same 'Result sheet'. A string matches the
        measurement Name case-insensitively ('egfp', 'NADH'); an int indexes
        them in sheet order. The sheet is scoped to that read's block before
        format detection, so each read parses as its own layout (endpoint vs
        kinetic). None (default) parses the whole sheet — which, for a
        multi-read sheet, resolves to whichever data block is detected first.
        The chosen read and the full list are recorded in
        df.attrs['measurement'] and df.attrs['available_measurements'].
    reads : 'mean' | 'all'
        Only used for 'Multiple Reads per Well' sheets (format='multi_read').
        'mean' (default) returns one row per well using the Tecan-computed Mean
        as the signal, with the per-well StDev in a '<signal>_std' column.
        'all' returns one row per individual sub-read with a 'Read' column
        naming its grid position ('1;2', '2;1', …) — works for any read
        pattern (2×2, 4×4, cross, circle), which just changes the positions.
    keep_saturated : bool
        Multi-read sheets only. When True, over-range / invalid wells (Tecan
        'OVER' / 'UNDER' / 'Invalid') are retained with a NaN signal and a
        boolean 'Saturated' column marking them, instead of being dropped as
        empty wells. Default False (drop them, after the usual warning). The
        NaN signal still excludes them from fits/standard curves — they're just
        visible and flagged in the frame.

    For absorbance reads that don't already carry a wavelength column (simple
    kinetic / endpoint), the probe wavelength is recovered from the Tecan
    'Measurement wavelength [nm]' metadata and added as a constant
    'Wavelength (nm)' column (matched to the selected table when a sheet holds
    several measurements).
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

    measurements = _find_measurements(raw)
    available_measurements = (
        _measurement_names(measurements) if len(measurements) > 1 else None
    )
    selected_measurement = None
    if measurement is not None:
        if not measurements:
            raise ValueError(
                f"measurement={measurement!r} requested but no read-mode "
                f"measurement blocks were found in sheet {sheet_name!r}"
            )
        chosen_m, available_measurements = _select_measurement(
            measurements, measurement
        )
        selected_measurement = chosen_m['name']
        raw = raw.iloc[chosen_m['start_row']:chosen_m['end_row']].reset_index(drop=True)

    if format == 'auto':
        format = _detect_format(raw)

    selected_table = None
    available_tables = None
    if format == 'simple_kinetic':
        tables = _simple_kinetic_tables(raw)
        if not tables:
            raise ValueError(
                "format='simple_kinetic' but no [Time [s], A1, ...] header row "
                f"was found in sheet {sheet_name!r}"
            )
        chosen, available_tables = _select_table(tables, table)
        selected_table = chosen['title']
        df = _parse_simple_kinetic(
            raw, chosen['header_row'], conditions, condition_tags,
            end_row=chosen['end_row'],
        )
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
    elif format == 'multi_read':
        header_row = _find_multi_read_header(raw)
        if header_row is None:
            raise ValueError(
                "format='multi_read' but no 'Well'-headered Multiple Reads "
                f"per Well block was found in sheet {sheet_name!r}"
            )
        df = _parse_multi_read(
            raw, header_row, conditions, condition_tags, reads=reads,
            keep_saturated=keep_saturated,
        )
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
            "'wavelength_scan', 'kinetic_scan', 'multi_read', or 'endpoint'"
        )

    df = _subset_wells(df, wells, raw, drop_empty_wells)

    if available_tables is not None:
        df.attrs['table'] = selected_table
        df.attrs['available_tables'] = available_tables

    if available_measurements is not None:
        df.attrs['available_measurements'] = available_measurements
    if selected_measurement is not None:
        df.attrs['measurement'] = selected_measurement

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
            if 'Absorbance_std' in df.columns:
                rename['Absorbance_std'] = f'{signal_col}_std'
            if rename:
                df = df.rename(columns=rename)
                df.attrs['signal_kind'] = mode

    # Single-wavelength reads (simple kinetic / endpoint) don't carry a
    # wavelength column the way scans do; recover the probe wavelength from the
    # absorbance metadata so it travels with the data. It's a constant per-
    # measurement tag, so it gets soft brackets — 'Wavelength (nm)' — to sit
    # alongside the other metadata columns (S (µM), E (nM)), unlike the
    # square-bracket scan axis 'Wavelength [nm]'.
    has_wavelength = any(c in df.columns for c in ('Wavelength (nm)', 'Wavelength [nm]'))
    if (mode is None or mode == 'absorbance') and not has_wavelength:
        wv = _extract_absorbance_wavelength(raw, selected_table)
        if wv is not None and 'Well' in df.columns:
            df.insert(df.columns.get_loc('Well') + 1, 'Wavelength (nm)', wv)

    n_wells = df['Well'].nunique() if 'Well' in df.columns else 0
    table_note = ''
    if available_tables is not None and len(available_tables) > 1:
        table_note = f", table={selected_table!r} of {available_tables}"
    meas_note = ''
    if available_measurements is not None and len(available_measurements) > 1:
        if selected_measurement is not None:
            meas_note = (
                f", measurement={selected_measurement!r} of {available_measurements}"
            )
        else:
            meas_note = (
                f", measurements available: {available_measurements} "
                "(pass measurement= to pick one)"
            )
    print(
        f"loaded {os.path.basename(filename)!r} "
        f"(sheet={sheet_name!r}, format={format}, "
        f"mode={mode or 'unknown'}, "
        f"{n_wells} wells, {len(df)} rows{table_note}{meas_note})"
    )
    return df


def load_folder(
    folder='.',
    conditions=None,
    pattern='*.xlsx',
    source_col='Notebook',
    sort=True,
    **load_kwargs,
):
    """Load every plate-reader file in a folder and stack them into one frame.

    Globs `folder` for files matching `pattern`, runs `load()` on each with the
    shared `conditions` / `**load_kwargs`, tags every row with its source file,
    and concatenates the results into a single long-format DataFrame. Use this
    when one experiment is split across several exports in a directory (e.g. one
    notebook per run, same plate layout repeated).

    Parameters
    ----------
    folder : str
        Directory to scan. Defaults to the current directory.
    conditions : dict[str, list] | None
        Well->metadata mapping applied to *every* file (same plate layout across
        runs). Passed straight to `load()`.
    pattern : str
        Glob pattern for the files to load (default '*.xlsx'). Excel lock files
        ('~$...') are skipped automatically.
    source_col : str
        Name of the column added to identify each row's source file. Its value
        is the filename without extension (e.g. 'run1.xlsx' -> 'run1').
    sort : bool
        Sort the matched files by name before loading (default True), so the
        source column has a stable, predictable order.
    **load_kwargs
        Forwarded to `load()` (e.g. condition_tags, sheet_name, format, wells,
        table, wavelength). The same options apply to every file.

    Returns
    -------
    pandas.DataFrame
        All files concatenated, with `source_col` inserted as the first column.
        `df.attrs['notebooks']` lists the loaded source names; `signal_kind` is
        carried over when every file agrees on it.
    """
    paths = [
        f for f in glob.glob(os.path.join(folder, pattern))
        if not os.path.basename(f).startswith('~$')
    ]
    if sort:
        paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(
            f"no files matching {pattern!r} found in {os.path.abspath(folder)!r}"
        )

    frames = []
    sources = []
    for path in paths:
        source = os.path.splitext(os.path.basename(path))[0]
        df = load(path, conditions=conditions, **load_kwargs)
        df.insert(0, source_col, source)
        frames.append(df)
        sources.append(source)

    combined = pd.concat(frames, ignore_index=True)

    combined.attrs['notebooks'] = sources
    combined.attrs['notebook_col'] = source_col
    signal_kinds = {f.attrs.get('signal_kind') for f in frames}
    signal_kinds.discard(None)
    if len(signal_kinds) == 1:
        combined.attrs['signal_kind'] = next(iter(signal_kinds))

    print(
        f"loaded {len(frames)} notebooks from {os.path.abspath(folder)!r} "
        f"({len(combined)} rows total): {sources}"
    )
    return combined


def extract_wavelength(scan_df, wavelength, tolerance=None):
    """Reduce a kinetic-scan DataFrame to a single-wavelength kinetic trace.

    Picks the closest available wavelength (warns if not exact). The result
    has the same schema as load() output and works directly
    with compute_initial_rates / plot_progress_curves.

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


def _extract_absorbance_wavelength(raw, table_title=None):
    """Measurement wavelength (nm) from the Tecan metadata, for absorbance reads.

    Each measurement action records its name ('Name | BzP') and probe wavelength
    ('Measurement wavelength [nm] | 284'). Returns the wavelength whose
    measurement name appears in `table_title` when several are present, the sole
    wavelength when there's just one (or all agree), or None when it can't be
    pinned to a single value.
    """
    pairs = []  # (measurement name, wavelength)
    current_name = None
    for i in range(min(len(raw), 200)):
        cell0 = raw.iat[i, 0] if raw.shape[1] else None
        if not isinstance(cell0, str):
            continue
        label = cell0.strip().lower()
        if label == 'name':
            for j in range(1, raw.shape[1]):
                v = raw.iat[i, j]
                if isinstance(v, str) and v.strip():
                    current_name = v.strip()
                    break
        elif label == 'measurement wavelength [nm]':
            for j in range(1, raw.shape[1]):
                v = raw.iat[i, j]
                if pd.isna(v):
                    continue
                try:
                    pairs.append((current_name, float(v)))
                    break
                except (TypeError, ValueError):
                    continue

    if not pairs:
        return None
    if table_title:
        tl = str(table_title).lower()
        named = [wv for name, wv in pairs if name and name.lower() in tl]
        if len(named) == 1 or (named and len(set(named)) == 1):
            return named[0]
    wvs = [wv for _, wv in pairs]
    return wvs[0] if len(set(wvs)) == 1 else None


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
