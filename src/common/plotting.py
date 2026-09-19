"""Shared plot style and helpers, so every figure in the repository matches.

Fonts, edges, line widths and the save settings live in PLOT_STYLE as matplotlib
rcParams, so plot functions only set what is specific to them. Call
apply_plot_style() once before drawing. Each analysis script does this in main(), and
anything else reusing their plot functions should do the same.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common.constants import STAGES
from models.model_type import ModelType

EDGE_COLOUR = "black"
LINE_WIDTH = 0.8
THICK_LINE_WIDTH = 1.5  # medians and the lines joining paired points
FALLBACK_COLOUR = "#bfbfbf"  # a model family this release does not define

PLOT_STYLE = {
    "font.size": 7.5,  # annotations, and any text not sized below
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "figure.titlesize": 10,
    "figure.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    # Bars, histograms, violins and legend patches all get a thin black edge
    "patch.force_edgecolor": True,
    "patch.edgecolor": EDGE_COLOUR,
    "patch.linewidth": LINE_WIDTH,
    "lines.linewidth": LINE_WIDTH,
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


def apply_plot_style():
    """Set PLOT_STYLE for every figure drawn after this call."""
    mpl.rcParams.update(PLOT_STYLE)


def model_colour(model_type):
    """Colour for a model_type value, grey for a family this release does not define."""
    try:
        return ModelType(model_type).colour
    except ValueError:
        return FALLBACK_COLOUR


def finish(fig, save_path=None):
    """Save the figure and close it, or show it when there is no path."""
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
        return
    plt.show()


def plot_confusion_matrix(counts, *, title=None, save_path=None):
    """Row-normalised confusion matrix, each cell annotated with its share and count.

    Takes a frame of epoch counts, true stage down the rows and predicted stage along
    the columns, so the diagonal is recall.
    """
    counts = counts.reindex(index=STAGES, columns=STAGES, fill_value=0)
    totals = counts.sum(axis=1).to_numpy()[:, None]
    shares = np.divide(
        counts.to_numpy(dtype=float),
        totals,
        out=np.zeros(counts.shape),
        where=totals > 0,  # a stage the reference never scored
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(STAGES)):
        for j in range(len(STAGES)):
            ax.annotate(
                f"{shares[i, j]:.2f}\n{int(counts.iat[i, j]):,}",
                (j, i),
                ha="center",
                va="center",
                color="white" if shares[i, j] > 0.5 else "black",
            )

    ax.set_xticks(range(len(STAGES)), STAGES)
    ax.set_yticks(range(len(STAGES)), STAGES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046)

    finish(fig, save_path)
