# plater

A Python package for analyzing plate-reader assay data.

## Overview

Lightweight toolkit for working with plate-reader output: parses Excel exports into long-format DataFrames, handles common analysis workflows (replicate pooling, model fitting), and produces publication-style plots.

Supports enzyme-kinetics workflows (initial rates, Michaelis–Menten fits), substrate/product standard curves, BCA protein quantitation, and absorbance / fluorescence / luminescence reads. Other assay types are planned (denaturant melts, ligand-binding titrations).

## Package layout

Everything is re-exported at the top level, so `import plater as pl` then `pl.<function>` works regardless of which module a function lives in. The split is just for navigation:

| Module | What's in it |
| --- | --- |
| `io` | Loading & format parsing (`load`, `extract_wavelength`, …) |
| `rates` | Initial-rate computation (`compute_initial_rates`) |
| `kinetics` | Michaelis–Menten model & fitting |
| `standards` | Substrate/product standard curves |
| `assays` | Colorimetric assays (BCA) |
| `plotting` | All `plot_*` routines |

(`_style` and `_common` hold shared internals.)

## Installation

### Install from source
```bash
git clone https://github.com/micah-olivas/plater.git
cd plater
pip install -e .

# Optional — enables on-line rate labels in plot_progress_curves
pip install -e ".[plot]"
```

### Requirements
- Python ≥ 3.10
- Core dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`
- Optional: `adjustText` (for `annotate_rates='lines'`)

## Usage

### Loading data

`load()` parses a plate-reader Excel export (currently Tecan Spark format) and auto-detects the data layout — *simple kinetic* (single wavelength × time) and *kinetic scan* (full spectrum × time), plus *wavelength scan* (single timepoint × spectrum), *endpoint* (single read per well), and *multi-read* (multiple reads per well). Pass a `conditions` dict mapping each well to its metadata, or omit it for a first-pass inspection of an unfamiliar file.

```python
import plater as pl

conditions = {
    'A1': [1, 'pNPA', 1250, 100],   # Replicate, Substrate, S (µM), E (nM)
    'A2': [1, 'pNPA',  625, 100],
    'B1': [1, 'pNPA', 1250,   0],   # no-enzyme control
    # ...
}

df = pl.load('myfile.xlsx', conditions=conditions)
```

Partial-plate / path-corrected exports list all 96 well columns in the header but only populate the region that was measured. `load()` drops the empty wells by default (`drop_empty_wells=True`), so you get just the section of interest. To subset explicitly, pass `wells=` — a rectangular range, a list, or `'auto'` to read the Tecan *Part of Plate* / *Plate area* metadata:

```python
df = pl.load('path_corrected.xlsx', wells='auto')      # use the measured 'Plate area' (e.g. G1-G2)
df = pl.load('path_corrected.xlsx', wells='B2-D5')     # rectangle: rows B–D × cols 2–5
df = pl.load('path_corrected.xlsx', wells=['G1', 'G2'])

pl.expand_well_range('B2-D5')   # -> ['B2', 'B3', ..., 'D5'] — the range helper on its own
```

**Stacked tables.** A path-corrected read writes *several* kinetic tables into one sheet — the raw measurement, the pathlength test/reference wavelengths, the `Pathlength corrected … [OD/cm]` result, and the pathlength itself. `load()` reads one table at a time (so they aren't melted together) and, by default, picks the pathlength-corrected one. Choose another with `table=` (title substring or index), and inspect what's available via `df.attrs`:

```python
df = pl.load('path_corrected.xlsx')                 # -> 'Pathlength corrected BzP [OD/cm]' by default
df = pl.load('path_corrected.xlsx', table='BzP')    # the raw 284 nm measurement
df = pl.load('path_corrected.xlsx', table=0)        # by sheet order

df.attrs['table']             # 'Pathlength corrected BzP [OD/cm]'
df.attrs['available_tables']  # ['BzP', 'BzP Pathlength test wavelength', ..., 'BzP Pathlength [cm]']
```

### Loading a folder of notebooks

When one experiment is spread across several exports in a directory — one
notebook per run, same plate layout repeated — `load_folder()` loads them all
and stacks them into one long-format DataFrame. The shared `conditions` (and any
other `load()` options) apply to every file, and a `Notebook` column (the
filename without extension) marks each row's source:

```python
df = pl.load_folder('runs/', conditions=conditions)   # every *.xlsx in runs/
df['Notebook'].unique()                                # ['run1', 'run2', ...]
df.attrs['notebooks']                                  # same list, load order

# customize the scan / source column
df = pl.load_folder('runs/', conditions=conditions,
                    pattern='*spark*.xlsx', source_col='Run')
```

Files are loaded in sorted order; Excel lock files (`~$...`) are skipped. The
result drops straight into the kinetics workflow below — group by `Notebook`
alongside the usual condition columns. Because each notebook keeps its own
trace identity (well IDs reused across files don't merge), you can separate them
when plotting — `pl.plot_progress_curves(kin, split_by='Notebook', ...)` for one
panel per notebook, or `color_by='Notebook'` to overlay them on a single axis.

### Kinetics workflow

```python
# Linear fit over [0, t_end] per (well × condition)
rates = pl.compute_initial_rates(df, t_end=75)

# Michaelis–Menten fit per substrate
mm = pl.fit_michaelis_menten(
    rates,
    exclude=[{'Substrate': 'pNPA', 'S (µM)': 1250}],
)

# Plots
pl.plot_progress_curves(df, rates_df=rates, t_end_fit=75)
pl.plot_initial_rates(rates, mm_params_df=mm)
```

### Kinetic scans

For full-spectrum-vs-time data, load and pick a probe wavelength:

```python
scan = pl.load('scan.xlsx', conditions=conditions)
pl.plot_spectra(scan, n_timepoints=8)        # pick a probe wavelength
df = pl.extract_wavelength(scan, 405)        # collapse to single λ — drops into the kinetics workflow above
```

## Input format

A `conditions` dict maps wells of interest to a list of metadata values. The list shape must match `condition_tags` (default `('Replicate', 'Substrate', 'S (µM)', 'E (nM)')`):

```python
conditions = {
    'A1': [1, 'pNPA', 1250, 100],
    'A2': [1, 'pNPA',  625, 100],
    'B1': [1, 'pNPA', 1250,   0],
}
```

Override `condition_tags` for other assay types — e.g. a denaturant melt:

```python
df = pl.load(
    'melt.xlsx',
    conditions={'A1': [1, 'WT', 0.0], 'A2': [1, 'WT', 0.5], ...},
    condition_tags=('Replicate', 'Variant', '[GuHCl] (M)'),
)
```

Wells absent from `conditions` are dropped from the returned DataFrame.

## Available functions

### Loading
- `load()` — auto-detect layout and parse to long-format DataFrame; subset with `wells=` / `drop_empty_wells`
- `load_folder()` — load every file in a folder with shared `conditions` and concatenate, tagged by a `Notebook` source column
- `extract_wavelength()` — collapse a kinetic scan to a single probe wavelength
- `expand_well_range()` — expand a range/list spec (`'B2-D5'`) into a list of well IDs

### Analysis
- `compute_initial_rates()` — linear fit over `[0, t_end]` per group, with auto sign detection and an exclusion list
- `fit_michaelis_menten()` — per-substrate MM fits with parameter errors

### Standard curves
- `compute_standard_curve()` — aggregate signal vs. `[S]` into a standard curve (mean ± SEM per concentration)
- `fit_standard_curve()` — linear fit; slope is the extinction coefficient
- `fit_single_point_standard()` — single-calibrator standard curve through a given intercept
- `apply_standard_curve()` — convert signal / rates to product concentration via a fit
- `adjust_stock_concentration()` — infer true stock concentration from a slope ratio against a reference

### Colorimetric assays (BCA)
- `fit_bca_standard()` — 4PL fit to a BCA BSA standard curve
- `back_calculate_bca()` — back-calculate sample stock concentrations from a BCA 4PL fit
- `pick_bca_dilution()` — pick the most reliable dilution per sample and check cross-dilution agreement

### Plotting
- `plot_standard_curves()` — overlay one or more standard curves, with optional per-dataset linear / exponential / 4PL fit
- `plot_progress_curves()` — A vs. t per well or per condition (replicate pooling), with optional inset of the fit window and overlaid linear fits
- `plot_initial_rates()` — rate vs. `[S]` with optional MM fit overlay
- `plot_rates_categorical()` — strip plot of rates by category (e.g. variant comparison at a single `[S]`)
- `plot_spectra()` — A vs. wavelength colored by time, one panel per well
- `plot_bca_standard()` — plot a BCA 4PL fit with standards (and optional sample overlay)
