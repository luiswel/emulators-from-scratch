"""Load a trained emulator and predict abundances with NumPy only.

A trained emulator is a short list of matrices plus the numbers that define
the input and output transforms. Evaluating it is a few matrix products, so
it runs anywhere NumPy runs, including inside a retrieval code that does
not use JAX.

Example
-------
>>> from emulator.trained import TrainedEmulator
>>> emu = TrainedEmulator("runs/emulator.npz")
>>> log10_values = emu.predict_log10(features)  # features: (n_points, 6)

``features`` has the same six columns as the grid: log10 T [K],
log10 P [bar], [M/H], log10 C/O, log10 S/O, log10 N/O.
"""

import numpy as np

from emulator import data

NUMPY_ACTIVATIONS = {
    "tanh": np.tanh,
    "silu": lambda z: z / (1.0 + np.exp(-z)),
}
"""dict of str to callable: NumPy versions of ``jax_mlp.ACTIVATIONS``."""

RANGE_TOLERANCE = 1e-3
"""float: margin allowed beyond [-1, 1] in scaled inputs (0.05% of each input's
range). The training set's extremes sit a hair inside the grid's edges, so the
validation and test sets contain a few models up to ~1e-5 beyond them."""


class TrainedEmulator:
    """A trained emulator read from a ``.npz`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        File written by ``jax_mlp.save_emulator``.

    Attributes
    ----------
    params : list of dict
        ``"W"`` (ndarray, shape (n_in, n_out)) and ``"b"`` (shape (n_out,))
        per layer, as float64.
    activation : str
        Hidden-layer activation (``"tanh"`` or ``"silu"``).
    input_names : list of str
        Input names in column order.
    output_names : list of str
        Output (species) names in column order.
    input_transform : dict
        ``"log10_columns"``, ``"lower"``, ``"upper"`` (see
        ``data.fit_input_transform``).
    output_transform : dict
        ``"log10_floor"``, ``"mean"``, ``"std"`` (see
        ``data.fit_output_transform``).
    """

    def __init__(
        self,
        path,
    ):
        """Read the weights, transforms and names from ``path``."""
        # 1. Read every array in the file
        with np.load(path, allow_pickle=False) as file:
            stored = {key: file[key] for key in file.files}

        # 2. Weights and biases
        n_layers = int(stored["n_layers"])
        self.params = [
            {
                "W": stored[f"W_{index}"].astype(np.float64),
                "b": stored[f"b_{index}"].astype(np.float64),
            }
            for index in range(n_layers)
        ]
        self.activation = str(stored["activation"])

        # 3. Names and transforms
        self.input_names = [str(name) for name in stored["input_names"]]
        self.output_names = [str(name) for name in stored["output_names"]]
        self.input_transform = {
            "log10_columns": stored["input_log10_columns"],
            "lower": stored["input_lower"],
            "upper": stored["input_upper"],
        }
        self.output_transform = {
            "log10_floor": float(stored["output_log10_floor"]),
            "mean": stored["output_mean"],
            "std": stored["output_std"],
        }

    def _scale_and_check(
        self,
        inputs,
        allow_extrapolation,
    ):
        """Scale physical inputs and refuse points outside the training range.

        Parameters
        ----------
        inputs : array_like, shape (n_points, n_inputs)
            Inputs in the units of the training grid, columns ordered as
            ``input_names``.
        allow_extrapolation : bool
            If False, raise an error for any point outside the training range.

        Returns
        -------
        scaled : ndarray, shape (n_points, n_inputs)
            Inputs mapped onto [-1, 1].

        Raises
        ------
        ValueError
            If ``inputs`` has the wrong shape, or a point lies outside the
            training range and ``allow_extrapolation`` is False.
        """
        # 1. Shape check
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        if inputs.shape[1] != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} input columns "
                f"{self.input_names}, got shape {inputs.shape}"
            )

        # 2. Scale onto [-1, 1]
        scaled = data.transform_inputs(inputs, self.input_transform)

        # 3. Range check: |scaled| > 1 means outside the training grid
        outside = np.abs(scaled) > 1.0 + RANGE_TOLERANCE
        if not allow_extrapolation and np.any(outside):
            bad_columns = [
                self.input_names[i] for i in np.where(outside.any(axis=0))[0]
            ]
            raise ValueError(
                f"{int(outside.any(axis=1).sum())} point(s) outside the training "
                f"range in {bad_columns}; the emulator would be extrapolating"
            )

        return scaled

    def predict_log10(
        self,
        inputs,
        allow_extrapolation=False,
    ):
        """Predict log10 outputs.

        Parameters
        ----------
        inputs : array_like, shape (n_points, n_inputs)
            Inputs in the units of the training grid, columns ordered as
            ``input_names``.
        allow_extrapolation : bool, default False
            Allow points outside the training range (not recommended).

        Returns
        -------
        log10_values : ndarray, shape (n_points, n_outputs)
            Predicted log10 outputs (mixing ratios, and the mean molecular
            weight in u for the team grid), columns ordered as
            ``output_names``.
        """
        # 1. Scale (and check) the inputs
        a = self._scale_and_check(inputs, allow_extrapolation)

        # 2. Hidden layers, then the linear output layer
        activation = NUMPY_ACTIVATIONS[self.activation]
        for layer in self.params[:-1]:
            a = activation(a @ layer["W"] + layer["b"])
        standardized = a @ self.params[-1]["W"] + self.params[-1]["b"]

        # 3. Undo the output transform
        log10_values = data.inverse_transform_outputs(
            standardized, self.output_transform
        )

        return log10_values

    def predict(
        self,
        inputs,
        allow_extrapolation=False,
    ):
        """Predict linear outputs (mixing ratios, mean molecular weight).

        Parameters
        ----------
        inputs : array_like, shape (n_points, n_inputs)
            Inputs in the units of the training grid, columns ordered as
            ``input_names``.
        allow_extrapolation : bool, default False
            Allow points outside the training range (not recommended).

        Returns
        -------
        values : ndarray, shape (n_points, n_outputs)
            10 to the power of ``predict_log10``: mixing ratios, and the
            mean molecular weight in u for the team grid.
        """
        # 1. Predict in log space, then exponentiate
        values = 10.0 ** self.predict_log10(inputs, allow_extrapolation)

        return values
