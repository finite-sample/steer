"""Shared visual grammar for publication figures in steer."""

import matplotlib as mpl

INK = "#222222"
MUTED = "#666666"
GRID = "#E5E5E5"
ACCENT = "#1F5A8A"
TEAL = "#1B7A6E"
WARN = "#B83A2D"
AMBER = "#C97A16"


def apply_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Charter", "Georgia", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "pdf.fonttype": 42,
        }
    )
