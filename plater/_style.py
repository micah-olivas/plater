"""plater._style: shared matplotlib theme + figure-background helpers (internal)."""

import numpy as np
import matplotlib as mpl


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
    # white backgrounds by default; pass transparent=True to a plotting function
    # for a transparent figure + axes. 'auto' makes savefig follow the figure's
    # facecolor, so a transparent figure also saves transparent.
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'auto',
})


def _apply_background(fig, axes=None, transparent=False):
    """Set the figure (and axes) background to white or transparent.

    White by default; `transparent=True` makes both the figure patch and every
    axes patch transparent (on screen and, via savefig.facecolor='auto', on
    save). `axes` may be a single Axes, an iterable of Axes, or None.
    """
    face = 'none' if transparent else 'white'
    fig.patch.set_facecolor(face)
    if axes is not None:
        for ax in np.atleast_1d(axes).ravel():
            if ax is not None:
                ax.set_facecolor(face)
