"""Plate reader kinetics analysis: initial rates, standard curves, MM fits.

Quick usage:
    import plater as pl

    df = pl.load('myfile.xlsx', conditions={...})
    rates = pl.compute_initial_rates(df, t_end=75)
    pl.plot_progress_curves(df, rates_df=rates, t_end_fit=75)

    mm = pl.fit_michaelis_menten(rates, exclude=[{'Substrate': 'BzP', 'S (µM)': 1250}])
    pl.plot_initial_rates(rates, mm_params_df=mm, exclude=[...])
"""

from . import _style as style  # `pl.style.POINT_EDGE_COLOR = ...` re-themes all plots

from .io import load, load_folder, extract_wavelength, expand_well_range
from .rates import compute_initial_rates
from .kinetics import michaelis_menten, fit_michaelis_menten
from .standards import compute_standard_curve, fit_standard_curve, fit_single_point_standard, apply_standard_curve, adjust_stock_concentration
from .assays import fit_bca_standard, back_calculate_bca, pick_bca_dilution, plot_bca_standard
from .plotting import plot_standard_curves, plot_progress_curves, plot_initial_rates, plot_rates_categorical, plot_spectra

# tunable defaults (override on the package, e.g. pl.DEFAULT_DPI = 200)
from ._style import DEFAULT_FIGSIZE, DEFAULT_FIGSIZE_WIDE, DEFAULT_DPI, POINT_EDGE_COLOR, POINT_EDGE_WIDTH, POINT_FACE_LIGHTEN
from ._common import DATA_COLUMNS, SIGNAL_COL_BY_KIND, SIGNAL_RATE_UNIT_BY_KIND
from .io import DEFAULT_CONDITION_TAGS

__all__ = [
    'load',
    'load_folder',
    'extract_wavelength',
    'expand_well_range',
    'compute_initial_rates',
    'michaelis_menten',
    'fit_michaelis_menten',
    'compute_standard_curve',
    'fit_standard_curve',
    'fit_single_point_standard',
    'apply_standard_curve',
    'adjust_stock_concentration',
    'fit_bca_standard',
    'back_calculate_bca',
    'pick_bca_dilution',
    'plot_bca_standard',
    'plot_standard_curves',
    'plot_progress_curves',
    'plot_initial_rates',
    'plot_rates_categorical',
    'plot_spectra',
    'style',
    'DEFAULT_FIGSIZE',
    'DEFAULT_FIGSIZE_WIDE',
    'DEFAULT_DPI',
    'POINT_EDGE_COLOR',
    'POINT_EDGE_WIDTH',
    'POINT_FACE_LIGHTEN',
    'DATA_COLUMNS',
    'SIGNAL_COL_BY_KIND',
    'SIGNAL_RATE_UNIT_BY_KIND',
    'DEFAULT_CONDITION_TAGS',
]
