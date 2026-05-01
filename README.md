# plater

A Python package for analyzing plate-reader assay data.

## Overview

Lightweight toolkit for working with plate-reader output: parses Excel exports into long-format DataFrames, handles common analysis workflows (control subtraction, replicate pooling, model fitting), and produces publication-style plots.

Currently supports enzyme kinetics workflows (initial-rate determination, Michaelis-Menten fits). Other assay types planned (denaturant melts, ligand-binding titrations, etc.).

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

`load()` parses a plate-reader Excel export (currently Tecan Spark format) and auto-detects the data layout — *simple kinetic* (single wavelength × time) vs. *kinetic scan* (full spectrum × time). Pass a `conditions` dict mapping each well to its metadata, or omit it for a first-pass inspection of an unfamiliar file.

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

### Kinetics workflow

```python
# Subtract no-enzyme control drift while preserving the t=0 baseline
df_corr = pl.subtract_paired_control(df, keep_controls=True)

# Linear fit over [0, t_end] per (well × condition)
rates = pl.compute_initial_rates(df_corr, t_end=75)

# Michaelis–Menten fit per substrate
mm = pl.fit_michaelis_menten(
    rates,
    exclude=[{'Substrate': 'pNPA', 'S (µM)': 1250}],
)

# Plots
pl.plot_progress_curves(df_corr, rates_df=rates, t_end_fit=75)
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
- `load()` — auto-detect layout and parse to long-format DataFrame
- `extract_wavelength()` — collapse a kinetic scan to a single probe wavelength

### Analysis
- `compute_initial_rates()` — linear fit over `[0, t_end]` per group, with auto sign detection and an exclusion list
- `subtract_paired_control()` — drift-correct samples against paired controls, preserving the absolute absorbance scale
- `fit_michaelis_menten()` — per-substrate MM fits with parameter errors

### Plotting
- `plot_progress_curves()` — A vs. t per well or per condition (replicate pooling), with optional inset of the fit window and overlaid linear fits
- `plot_initial_rates()` — rate vs. `[S]` with optional MM fit overlay
- `plot_rates_categorical()` — strip plot of rates by category (e.g. variant comparison at a single `[S]`)
- `plot_spectra()` — A vs. wavelength colored by time, one panel per well
