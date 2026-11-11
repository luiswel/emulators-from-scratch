"""Load, transform and split the emulator training grid.

The network never sees physical units. Each input is mapped onto [-1, 1]
and each output (a log10 mixing ratio, or log10 of the mean molecular
weight) is standardized to mean 0 and standard deviation 1 over the
training set. The numbers that define these maps (the "transforms") belong
to the emulator as much as the network weights do: they are fitted on the
training set only and saved next to the weights.
"""

import json

import h5py
import numpy as np


def _check_two_dimensional(
    array,
    name,
):
    """Raise an error unless ``array`` is a finite 2-D array.

    Parameters
    ----------
    array : ndarray
        Array to check.
    name : str
        Name used in the error message.

    Raises
    ------
    ValueError
        If ``array`` is not 2-D or contains NaN or infinite values.
    """
    # 1. Shape check
    if np.ndim(array) != 2:
        raise ValueError(
            f"{name} must be 2-D (n_models, n_columns), got {np.shape(array)}"
        )

    # 2. Finite-value check
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains NaN or infinite values; clean the grid first"
        )


def load_grid(path):
    """Read the team's FastChem grid (HDF5 file written by ``chemem``).

    Parameters
    ----------
    path : str or pathlib.Path
        The HDF5 file, e.g. ``training_250k.h5``.

    Returns
    -------
    grid : dict
        ``"inputs"`` : ndarray, shape (n_models, 6)
            The six features: log10 T [K], log10 P [bar], [M/H] [dex],
            log10 C/O, log10 S/O, log10 N/O.
        ``"input_names"`` : list of str
            Names of the six features, in column order.
        ``"outputs"`` : ndarray, shape (n_models, 19)
            log10 volume mixing ratios of 18 species (floored at -30), then
            log10 of the mean molecular weight [u] in the last column.
        ``"output_names"`` : list of str
            Species labels (``"H2O"``, ``"CO"``, ...) and ``"mu"``.
        ``"split"`` : ndarray of int, shape (n_models,)
            0 = train, 1 = validation, 2 = test, as assigned by the grid
            generator (so everyone in the team uses the same test set).
        ``"valid"`` : ndarray of bool, shape (n_models,)
            True where FastChem converged and conserved every element.
        ``"config"`` : dict
            The generator's settings (ranges, sampling, FastChem options).

    Notes
    -----
    Only this function knows the file layout. If the team changes the
    format, change this function and nothing else.
    """
    # 1. Read the arrays we need
    with h5py.File(path, "r") as file:
        inputs = file["features"][:]
        log10_vmr = file["log10_vmr"][:].astype(np.float64)
        log10_mu = file["log10_mean_molecular_weight"][:].astype(np.float64)
        input_names = [name.decode() for name in file["feature_names"][:]]
        species = [name.decode() for name in file["species_labels"][:]]
        split = file["split"][:].astype(int)
        valid = file["valid"][:].astype(bool)
        config = json.loads(file.attrs["config_json"])

    # 2. Put the mean molecular weight after the 18 species
    outputs = np.column_stack([log10_vmr, log10_mu])
    output_names = species + ["mu"]

    grid = {
        "inputs": inputs,
        "input_names": input_names,
        "outputs": outputs,
        "output_names": output_names,
        "split": split,
        "valid": valid,
        "config": config,
    }

    return grid


def split_from_labels(split):
    """Turn a column of split labels into row numbers for each set.

    Parameters
    ----------
    split : ndarray of int, shape (n_models,)
        0 = train, 1 = validation, 2 = test.

    Returns
    -------
    indices : dict of str to ndarray of int
        Row numbers for ``"train"``, ``"val"`` and ``"test"``.

    Notes
    -----
    The training set fits the weights. The validation set is used to make
    choices (network size, learning rate, when to stop). The test set is
    looked at once, at the end, to report how good the final emulator is.
    """
    # 1. Rows carrying each label
    indices = {
        "train": np.where(split == 0)[0],
        "val": np.where(split == 1)[0],
        "test": np.where(split == 2)[0],
    }

    return indices


def split_indices(
    n_models,
    fractions=(0.8, 0.1, 0.1),
    seed=42,
):
    """Randomly assign every model to the training, validation or test set.

    Use this for a grid that does not come with its own split.

    Parameters
    ----------
    n_models : int
        Number of models in the grid.
    fractions : tuple of float, default (0.8, 0.1, 0.1)
        Fractions for (train, validation, test). Must sum to 1.
    seed : int, default 42
        Seed for the random shuffle, so the split is reproducible.

    Returns
    -------
    indices : dict of str to ndarray of int
        Row numbers for ``"train"``, ``"val"`` and ``"test"``. Every row
        appears in exactly one set.
    """
    # 1. Check the fractions
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {fractions}")

    # 2. Shuffle the row numbers with a fixed seed
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_models)

    # 3. Cut the shuffled rows into three blocks
    n_train = int(round(fractions[0] * n_models))
    n_val = int(round(fractions[1] * n_models))
    indices = {
        "train": np.sort(shuffled[:n_train]),
        "val": np.sort(shuffled[n_train : n_train + n_val]),
        "test": np.sort(shuffled[n_train + n_val :]),
    }

    return indices


def fit_input_transform(
    inputs,
    log10_columns=None,
):
    """Find the numbers that map each input column onto [-1, 1].

    Parameters
    ----------
    inputs : ndarray, shape (n_models, n_inputs)
        Training-set inputs.
    log10_columns : sequence of bool, length n_inputs, optional
        True for columns to replace by their log10 before scaling (for
        example a pressure stored in bar). Default None: no column is
        logged, which is right for the team grid because its features are
        already log10 T, log10 P, and so on.

    Returns
    -------
    transform : dict
        ``"log10_columns"`` : ndarray of bool, shape (n_inputs,)
            Which columns get a log10.
        ``"lower"`` : ndarray, shape (n_inputs,)
            Minimum of each column (after the log10, where applied).
        ``"upper"`` : ndarray, shape (n_inputs,)
            Maximum of each column (after the log10, where applied).
    """
    # 1. Check the arguments
    _check_two_dimensional(inputs, "inputs")
    if log10_columns is None:
        log10_columns = np.zeros(inputs.shape[1], dtype=bool)
    log10_columns = np.asarray(log10_columns, dtype=bool)
    if log10_columns.shape != (inputs.shape[1],):
        raise ValueError("log10_columns needs one entry per input column")

    # 2. Take log10 of the chosen columns
    values = np.array(inputs, dtype=float)
    values[:, log10_columns] = np.log10(values[:, log10_columns])

    # 3. Record the range of every column
    transform = {
        "log10_columns": log10_columns,
        "lower": values.min(axis=0),
        "upper": values.max(axis=0),
    }

    return transform


def transform_inputs(
    inputs,
    transform,
):
    """Map inputs onto [-1, 1] with a fitted input transform.

    Parameters
    ----------
    inputs : ndarray, shape (n_models, n_inputs)
        Inputs in the same units as the training inputs.
    transform : dict
        Output of ``fit_input_transform``.

    Returns
    -------
    scaled : ndarray, shape (n_models, n_inputs)
        Inputs in [-1, 1] for points inside the training range. Values
        outside [-1, 1] mean the point lies outside the training grid,
        where the emulator would be extrapolating.
    """
    # 1. Take log10 of the same columns as during fitting
    values = np.array(inputs, dtype=float)
    columns = transform["log10_columns"]
    values[:, columns] = np.log10(values[:, columns])

    # 2. Linear map: lower -> -1, upper -> +1
    width = transform["upper"] - transform["lower"]
    scaled = 2.0 * (values - transform["lower"]) / width - 1.0

    return scaled


def fit_output_transform(
    log10_outputs,
    log10_floor=-30.0,
):
    """Find the numbers that standardize each output column.

    Parameters
    ----------
    log10_outputs : ndarray, shape (n_models, n_outputs)
        Training-set outputs as log10 values.
    log10_floor : float, default -30.0
        Values below this are raised to it before anything else. The grid
        is already floored at -30; a higher floor (say -20) tells the
        network not to care about abundances far below what a spectrum
        can detect.

    Returns
    -------
    transform : dict
        ``"log10_floor"`` : float
            The floor.
        ``"mean"`` : ndarray, shape (n_outputs,)
            Mean of each (floored) output over the training set.
        ``"std"`` : ndarray, shape (n_outputs,)
            Standard deviation of each (floored) output. Columns that never
            change (std = 0) get std = 1 to avoid dividing by zero.
    """
    # 1. Check the input
    _check_two_dimensional(log10_outputs, "log10_outputs")

    # 2. Apply the floor
    floored = np.maximum(log10_outputs, log10_floor)

    # 3. Mean and spread of each column
    mean = floored.mean(axis=0)
    std = floored.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)

    transform = {"log10_floor": float(log10_floor), "mean": mean, "std": std}

    return transform


def transform_outputs(
    log10_outputs,
    transform,
):
    """Turn log10 outputs into the standardized values the network learns.

    Parameters
    ----------
    log10_outputs : ndarray, shape (n_models, n_outputs)
        Outputs as log10 values.
    transform : dict
        Output of ``fit_output_transform``.

    Returns
    -------
    standardized : ndarray, shape (n_models, n_outputs)
        (max(log10_outputs, floor) - mean) / std for every column.
    """
    # 1. Floor, then standardize each column
    floored = np.maximum(log10_outputs, transform["log10_floor"])
    standardized = (floored - transform["mean"]) / transform["std"]

    return standardized


def inverse_transform_outputs(
    standardized,
    transform,
):
    """Turn network outputs back into log10 values.

    Parameters
    ----------
    standardized : ndarray, shape (n_models, n_outputs)
        Network outputs.
    transform : dict
        Output of ``fit_output_transform``.

    Returns
    -------
    log10_outputs : ndarray, shape (n_models, n_outputs)
        Predicted log10 values, never below the floor (the network can
        overshoot below it; those values are raised back to the floor).
    """
    # 1. Undo the standardization
    log10_outputs = np.asarray(standardized) * transform["std"] + transform["mean"]

    # 2. Nothing below the floor
    log10_outputs = np.maximum(log10_outputs, transform["log10_floor"])

    return log10_outputs
