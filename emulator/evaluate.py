"""Error statistics in dex and diagnostic plots for emulators.

Every error here is a difference of log10 values, i.e. in dex. An error of
0.01 dex is a 2.3% error in the mixing ratio; 0.1 dex is 26%.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
"""list of str: categorical colours, used in this fixed order."""

CONTEXT_GREY = "#b8b6ae"
"""str: colour for lines that give context but are not the focus."""


def _check_same_shape(
    log10_true,
    log10_pred,
):
    """Raise an error unless the true and predicted arrays match.

    Parameters
    ----------
    log10_true : ndarray, shape (n_models, n_outputs)
        True log10 values.
    log10_pred : ndarray, shape (n_models, n_outputs)
        Predicted log10 values.

    Raises
    ------
    ValueError
        If the shapes differ or the arrays are not 2-D.
    """
    # 1. Both 2-D and the same shape
    if np.ndim(log10_true) != 2 or np.shape(log10_true) != np.shape(log10_pred):
        raise ValueError(
            f"true and predicted arrays must be 2-D with the same shape, got "
            f"{np.shape(log10_true)} and {np.shape(log10_pred)}"
        )


def error_statistics(
    log10_true,
    log10_pred,
    output_names,
    threshold=None,
):
    """Summarize the absolute error of every output, in dex.

    Parameters
    ----------
    log10_true : ndarray, shape (n_models, n_outputs)
        True log10 values (from the chemistry code).
    log10_pred : ndarray, shape (n_models, n_outputs)
        Emulator predictions.
    output_names : list of str
        Name of each output column.
    threshold : float, optional
        If given, a model counts for an output only where the true or the
        predicted value is above ``threshold`` (e.g. -12: abundances a
        spectrum could notice). Taking the larger of the two catches both
        missed molecules and invented ones. Default None: every model counts.

    Returns
    -------
    stats : dict
        ``"name"`` (list of str) and, as ndarrays of shape (n_outputs,):
        ``"n"`` (models counted), ``"median"``, ``"p95"``, ``"p99"`` and
        ``"max"`` of |log10_pred - log10_true|.
    """
    # 1. Check the inputs and compute |error| in dex
    _check_same_shape(log10_true, log10_pred)
    abs_error = np.abs(log10_pred - log10_true)

    # 2. One summary per output column
    stats = {
        "name": list(output_names),
        "n": [],
        "median": [],
        "p95": [],
        "p99": [],
        "max": [],
    }
    for column in range(abs_error.shape[1]):
        errors = abs_error[:, column]

        # 2a. Optionally keep only models where this output matters
        if threshold is not None:
            larger = np.maximum(log10_true[:, column], log10_pred[:, column])
            errors = errors[larger > threshold]

        # 2b. Percentiles (NaN when no model is left)
        stats["n"].append(errors.size)
        if errors.size == 0:
            errors = np.array([np.nan])
        stats["median"].append(np.median(errors))
        stats["p95"].append(np.percentile(errors, 95))
        stats["p99"].append(np.percentile(errors, 99))
        stats["max"].append(np.max(errors))

    # 3. Lists to arrays
    for key in ("n", "median", "p95", "p99", "max"):
        stats[key] = np.array(stats[key])

    return stats


def summary_errors(
    log10_true,
    log10_pred,
    threshold=-12.0,
):
    """Two numbers for comparing emulators: typical error and tail error.

    Parameters
    ----------
    log10_true : ndarray, shape (n_models, n_outputs)
        True log10 values.
    log10_pred : ndarray, shape (n_models, n_outputs)
        Emulator predictions.
    threshold : float or None, default -12.0
        Passed to ``error_statistics``: count a model for an output only
        where the true or predicted value exceeds it. None counts all.

    Returns
    -------
    summary : dict of str to float
        ``"median"`` : median |error| [dex], averaged over the outputs.
        ``"p99"`` : 99th percentile of |error| [dex], averaged over the
        outputs.
    """
    # 1. Per-output statistics
    names = [str(column) for column in range(np.shape(log10_true)[1])]
    stats = error_statistics(log10_true, log10_pred, names, threshold)

    # 2. Average over the outputs
    summary = {
        "median": float(np.nanmean(stats["median"])),
        "p99": float(np.nanmean(stats["p99"])),
    }

    return summary


def format_error_table(
    stats,
    title="",
):
    """Format the output of ``error_statistics`` as a text table.

    Parameters
    ----------
    stats : dict
        Output of ``error_statistics``.
    title : str, default ""
        Line printed above the table.

    Returns
    -------
    table : str
        One row per output: number of models, median, 95th and 99th
        percentile and maximum of |error| in dex.
    """
    # 1. Header
    lines = [title] if title else []
    lines.append(
        f"{'output':8s} {'n':>7s} {'median':>8s} {'95%':>8s} {'99%':>8s} {'max':>8s}  [dex]"
    )

    # 2. One line per output
    for row, name in enumerate(stats["name"]):
        lines.append(
            f"{name:8s} {stats['n'][row]:7d} {stats['median'][row]:8.4f} "
            f"{stats['p95'][row]:8.4f} {stats['p99'][row]:8.4f} {stats['max'][row]:8.3f}"
        )
    table = "\n".join(lines)

    return table


def plot_history(history):
    """Plot training and validation loss, and the learning rate, per epoch.

    Parameters
    ----------
    history : dict
        Must have ``"train_loss"`` and ``"val_loss"``; ``"learning_rate"``
        is plotted in a second panel if present.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    # 1. Panels: loss, plus the learning rate if it was recorded
    has_lr = "learning_rate" in history
    fig, axes = plt.subplots(
        1, 2 if has_lr else 1, figsize=(10 if has_lr else 5.5, 3.6)
    )
    axes = np.atleast_1d(axes)
    epochs = np.arange(len(history["train_loss"]))

    # 2. Loss curves
    axes[0].plot(
        epochs, history["train_loss"], color=PALETTE[0], lw=2, label="training"
    )
    axes[0].plot(
        epochs, history["val_loss"], color=PALETTE[1], lw=2, label="validation"
    )
    axes[0].set(
        xlabel="epoch", ylabel="mean squared error (standardized)", yscale="log"
    )
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.3)

    # 3. Learning rate
    if has_lr:
        axes[1].plot(epochs, history["learning_rate"], color=PALETTE[0], lw=2)
        axes[1].set(xlabel="epoch", ylabel="learning rate", yscale="log")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()

    return fig


def plot_error_histograms(
    log10_true,
    log10_pred,
    output_names,
    threshold=None,
    n_columns=5,
):
    """Histogram of the signed error (prediction minus truth) of every output.

    Parameters
    ----------
    log10_true : ndarray, shape (n_models, n_outputs)
        True log10 values.
    log10_pred : ndarray, shape (n_models, n_outputs)
        Emulator predictions.
    output_names : list of str
        Name of each output column.
    threshold : float, optional
        Count a model only where the true or predicted value exceeds this
        (see ``error_statistics``). Default None: all models.
    n_columns : int, default 5
        Panels per row.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One panel per output, with a log-scaled count axis so the tails
        (the rare large errors) stay visible.
    """
    # 1. Check the inputs and lay out the panels
    _check_same_shape(log10_true, log10_pred)
    n_outputs = len(output_names)
    n_rows = int(np.ceil(n_outputs / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.6 * n_columns, 2.1 * n_rows))
    axes = axes.ravel()

    # 2. One histogram per output
    for column, name in enumerate(output_names):
        error = log10_pred[:, column] - log10_true[:, column]
        if threshold is not None:
            larger = np.maximum(log10_true[:, column], log10_pred[:, column])
            error = error[larger > threshold]
        p99 = np.percentile(np.abs(error), 99) if error.size else 0.0
        limit = max(3.0 * p99, 1e-3)
        axes[column].hist(np.clip(error, -limit, limit), bins=60, color=PALETTE[0])
        axes[column].set_yscale("log")
        axes[column].set_title(f"{name}  (99%: {p99:.3f} dex)", fontsize=9)
        axes[column].tick_params(labelsize=7)

    # 3. Hide unused panels and label the axes once
    for ax in axes[n_outputs:]:
        ax.set_visible(False)
    fig.supxlabel(
        "prediction - truth [dex]  (values beyond the axis are piled at the edges)"
    )
    fig.supylabel("number of models")
    fig.tight_layout()

    return fig


def plot_residuals_vs_true(
    log10_true,
    log10_pred,
    output_names,
    max_points=20000,
    n_columns=5,
    seed=0,
):
    """Scatter of the error against the true value for every output.

    Parameters
    ----------
    log10_true : ndarray, shape (n_models, n_outputs)
        True log10 values.
    log10_pred : ndarray, shape (n_models, n_outputs)
        Emulator predictions.
    output_names : list of str
        Name of each output column.
    max_points : int, default 20000
        Plot at most this many randomly chosen models (keeps the file small).
    n_columns : int, default 5
        Panels per row.
    seed : int, default 0
        Seed for choosing the plotted models.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One panel per output: x = true log10 value, y = prediction - truth.
        Shows at which abundances the emulator struggles (often near the
        floor, where a species switches on or off).
    """
    # 1. Check the inputs, pick the models to plot, lay out the panels
    _check_same_shape(log10_true, log10_pred)
    rng = np.random.default_rng(seed)
    rows = rng.permutation(log10_true.shape[0])[:max_points]
    n_outputs = len(output_names)
    n_rows = int(np.ceil(n_outputs / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.6 * n_columns, 2.1 * n_rows))
    axes = axes.ravel()

    # 2. One scatter per output
    for column, name in enumerate(output_names):
        truth = log10_true[rows, column]
        error = log10_pred[rows, column] - truth
        axes[column].scatter(
            truth, error, s=1, alpha=0.25, color=PALETTE[0], rasterized=True
        )
        axes[column].axhline(0.0, color="black", lw=0.8)
        axes[column].set_title(name, fontsize=9)
        axes[column].tick_params(labelsize=7)

    # 3. Hide unused panels and label the axes once
    for ax in axes[n_outputs:]:
        ax.set_visible(False)
    fig.supxlabel("true log10 value")
    fig.supylabel("prediction - truth [dex]")
    fig.tight_layout()

    return fig


def plot_error_vs_input(
    inputs,
    abs_error,
    input_labels,
    output_names,
    n_bins=25,
    percentile=95,
    n_highlight=4,
):
    """How the error of each output changes along each input.

    Parameters
    ----------
    inputs : ndarray, shape (n_models, n_inputs)
        Inputs of the evaluated models (any units; used for the x axes).
    abs_error : ndarray, shape (n_models, n_outputs)
        |prediction - truth| in dex.
    input_labels : list of str
        Axis label for each input.
    output_names : list of str
        Name of each output column.
    n_bins : int, default 25
        Bins along each input.
    percentile : float, default 95
        Percentile of |error| shown in each bin.
    n_highlight : int, default 4
        The outputs with the largest overall error are drawn in colour, the
        rest in grey.

    Returns
    -------
    fig : matplotlib.figure.Figure
        One panel per input. Each line is one output.
    """
    # 1. Choose the outputs to highlight (largest overall error)
    overall = np.percentile(abs_error, percentile, axis=0)
    highlight = list(np.argsort(overall)[::-1][:n_highlight])

    # 2. Lay out the panels
    n_inputs = inputs.shape[1]
    n_columns = 3
    n_rows = int(np.ceil(n_inputs / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns, figsize=(4.2 * n_columns, 3.2 * n_rows), sharey=True
    )
    axes = axes.ravel()

    # 3. Binned percentile of |error| along every input, for every output
    for column in range(n_inputs):
        ax = axes[column]
        for output in range(abs_error.shape[1]):
            value, edges, _ = binned_statistic(
                inputs[:, column],
                abs_error[:, output],
                statistic=lambda e: np.percentile(e, percentile) if e.size else np.nan,
                bins=n_bins,
            )
            centres = 0.5 * (edges[:-1] + edges[1:])
            value = np.where(value > 0.0, value, np.nan)  # no zeros on a log axis
            if output in highlight:
                colour = PALETTE[highlight.index(output)]
                ax.plot(
                    centres,
                    value,
                    color=colour,
                    lw=2,
                    label=output_names[output],
                    zorder=3,
                )
            else:
                ax.plot(centres, value, color=CONTEXT_GREY, lw=1, zorder=2)
        ax.set(xlabel=input_labels[column], yscale="log")
        ax.grid(alpha=0.3)

    # 4. Labels and legend
    for ax in axes[n_inputs:]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, title="largest errors", title_fontsize=8)
    fig.supylabel(f"{percentile}th percentile of |error| [dex]")
    fig.tight_layout()

    return fig


def plot_error_map(
    x,
    y,
    abs_error,
    x_label,
    y_label,
    title="",
    percentile=95,
    n_bins=30,
):
    """Map of an error percentile over two inputs (e.g. temperature and pressure).

    Parameters
    ----------
    x : ndarray, shape (n_models,)
        First input (horizontal axis).
    y : ndarray, shape (n_models,)
        Second input (vertical axis).
    abs_error : ndarray, shape (n_models,)
        |error| in dex for one output (or the largest over all outputs).
    x_label : str
        Horizontal axis label.
    y_label : str
        Vertical axis label.
    title : str, default ""
        Panel title.
    percentile : float, default 95
        Percentile of |error| shown in each cell.
    n_bins : int, default 30
        Cells along each axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Darker cells have larger errors (log colour scale).
    """
    # 1. Percentile of |error| in every cell
    value, x_edges, y_edges, _ = binned_statistic_2d(
        x,
        y,
        abs_error,
        statistic=lambda e: np.percentile(e, percentile) if e.size else np.nan,
        bins=n_bins,
    )

    # 2. Draw the map
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    floor = max(np.nanmin(value), 1e-5)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        value.T,
        cmap="Blues",
        norm=plt.matplotlib.colors.LogNorm(vmin=floor, vmax=np.nanmax(value)),
    )
    fig.colorbar(mesh, ax=ax, label=f"{percentile}th percentile of |error| [dex]")
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    fig.tight_layout()

    return fig
