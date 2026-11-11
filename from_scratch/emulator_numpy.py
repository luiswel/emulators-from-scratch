"""A complete chemistry emulator in one file, using NumPy and nothing else.

This file repeats, in order and without any machine-learning library, every
step the lessons build up:

1. read the grid
2. split it into training, validation and test sets
3. transform inputs and outputs
4. draw the starting weights
5. the forward pass
6. the loss
7. backpropagation
8. Adam
9. the training loop
10. prediction with the trained weights
11. the test

Run it from anywhere (the grid must be in the repository folder):

    python from_scratch/emulator_numpy.py

With the default settings (two hidden layers of 128 units, 40 epochs) it
trains in about a minute on a laptop CPU.
"""

import os
import time

import h5py
import numpy as np

GRID_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "training_250k.h5"
)
"""str: the training grid, in the repository folder (log10 features in, log10 abundances out)."""

LAYER_SIZES = [6, 128, 128, 19]
"""list of int: units per layer, inputs first and outputs last."""

LOG10_FLOOR = -15.0
"""float: abundances below 10**LOG10_FLOOR are raised to it."""

N_EPOCHS = 40
"""int: passes through the training set."""

BATCH_SIZE = 512
"""int: models per gradient step."""

LR_START = 1e-3
"""float: Adam learning rate at the first epoch."""

LR_END = 1e-5
"""float: Adam learning rate at the last epoch."""


def read_grid(path):
    """Read inputs, outputs and the split labels from the grid file.

    Parameters
    ----------
    path : str
        HDF5 grid file.

    Returns
    -------
    inputs : ndarray, shape (n_models, 6)
        log10 T, log10 P, [M/H], log10 C/O, log10 S/O, log10 N/O.
    outputs : ndarray, shape (n_models, 19)
        log10 mixing ratios of 18 species, then log10 mean molecular weight.
    split : ndarray of int, shape (n_models,)
        0 = train, 1 = validation, 2 = test.
    """
    # 1. Read the arrays stored in the file
    with h5py.File(path, "r") as file:
        inputs = file["features"][:]
        log10_vmr = file["log10_vmr"][:].astype(np.float64)
        log10_mu = file["log10_mean_molecular_weight"][:].astype(np.float64)
        split = file["split"][:]

    # 2. Put the mean molecular weight after the 18 species
    outputs = np.column_stack([log10_vmr, log10_mu])

    return inputs, outputs, split


def init_params(
    layer_sizes,
    rng,
):
    """Draw weights from Normal(0, 1/n_in) and set biases to zero.

    Parameters
    ----------
    layer_sizes : list of int
        Units per layer.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    params : list of list of ndarray
        ``[W, b]`` per layer, with W of shape (n_in, n_out), b of shape
        (n_out,).
    """
    # 1. One weight matrix and one bias vector per layer
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))
        params.append([weights, np.zeros(n_out)])

    return params


def forward(
    params,
    x,
):
    """Forward pass: a = tanh(a W + b) for hidden layers, linear output.

    Parameters
    ----------
    params : list of list of ndarray
        ``[W, b]`` per layer.
    x : ndarray, shape (n_batch, n_inputs)
        Scaled inputs.

    Returns
    -------
    y_pred : ndarray, shape (n_batch, n_outputs)
        Network outputs.
    activations : list of ndarray
        The input of every layer (``activations[0] = x``), kept for the
        backward pass.
    """
    # 1. Hidden layers
    activations = [x]
    a = x
    for weights, biases in params[:-1]:
        a = np.tanh(a @ weights + biases)
        activations.append(a)

    # 2. Linear output layer
    weights, biases = params[-1]
    y_pred = a @ weights + biases

    return y_pred, activations


def backward(
    params,
    activations,
    y_pred,
    y_true,
):
    """Gradients of the mean squared error by backpropagation.

    Parameters
    ----------
    params : list of list of ndarray
        ``[W, b]`` per layer.
    activations : list of ndarray
        Second output of ``forward``.
    y_pred : ndarray, shape (n_batch, n_outputs)
        First output of ``forward``.
    y_true : ndarray, shape (n_batch, n_outputs)
        Targets.

    Returns
    -------
    grads : list of list of ndarray
        ``[dL/dW, dL/db]`` per layer, same shapes as ``params``.

    Notes
    -----
    ``delta`` is dL/dz for the current layer, z = a W + b:
    dL/dW = a.T @ delta, dL/db = delta summed over the batch,
    delta_previous = (delta @ W.T) * (1 - a**2).
    """
    # 1. Derivative of the loss with respect to the output
    delta = 2.0 * (y_pred - y_true) / y_true.size

    # 2. Back through the layers
    grads = [None] * len(params)
    for index in reversed(range(len(params))):
        a_in = activations[index]
        grads[index] = [a_in.T @ delta, delta.sum(axis=0)]
        if index > 0:
            delta = (delta @ params[index][0].T) * (1.0 - a_in**2)

    return grads


def adam_step(
    params,
    grads,
    state,
    learning_rate,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """Update every weight in place with one Adam step.

    Parameters
    ----------
    params : list of list of ndarray
        ``[W, b]`` per layer, updated in place.
    grads : list of list of ndarray
        Gradients, same structure.
    state : dict
        ``"m"`` and ``"v"`` (same structure as ``params``) and ``"t"``
        (int), updated in place.
    learning_rate : float
        Step size.
    beta1 : float, default 0.9
        Decay of the running mean of the gradient.
    beta2 : float, default 0.999
        Decay of the running mean of the squared gradient.
    epsilon : float, default 1e-8
        Avoids dividing by zero.
    """
    # 1. Count the step (the bias corrections need it)
    state["t"] += 1
    correction1 = 1.0 - beta1 ** state["t"]
    correction2 = 1.0 - beta2 ** state["t"]

    # 2. Running averages, then the update, for every array
    for layer, layer_grads, layer_m, layer_v in zip(
        params, grads, state["m"], state["v"]
    ):
        for value, grad, m, v in zip(layer, layer_grads, layer_m, layer_v):
            m *= beta1
            m += (1.0 - beta1) * grad
            v *= beta2
            v += (1.0 - beta2) * grad**2
            value -= (
                learning_rate * (m / correction1) / (np.sqrt(v / correction2) + epsilon)
            )


def cosine_learning_rate(
    epoch,
    n_epochs,
    lr_start,
    lr_end,
):
    """Learning rate falling from lr_start to lr_end along half a cosine.

    Parameters
    ----------
    epoch : int
        Current epoch, from 0.
    n_epochs : int
        Total epochs.
    lr_start : float
        Rate at the first epoch.
    lr_end : float
        Rate at the last epoch.

    Returns
    -------
    learning_rate : float
        Rate for this epoch.
    """
    # 1. Fraction of the run completed, then the cosine
    progress = epoch / max(n_epochs - 1, 1)
    learning_rate = lr_end + 0.5 * (lr_start - lr_end) * (
        1.0 + np.cos(np.pi * progress)
    )

    return learning_rate


def mean_squared_error(
    params,
    x,
    y,
):
    """Mean squared error of the network on a set of models.

    Parameters
    ----------
    params : list of list of ndarray
        ``[W, b]`` per layer.
    x : ndarray, shape (n_models, n_inputs)
        Scaled inputs.
    y : ndarray, shape (n_models, n_outputs)
        Targets.

    Returns
    -------
    loss : float
        mean((forward(x) - y)**2).
    """
    # 1. Predict and average the squared residuals
    loss = float(np.mean((forward(params, x)[0] - y) ** 2))

    return loss


def main():
    """Read, transform, train, predict and test, printing the results."""
    # 1. Read the grid and split it with the labels stored in the file
    inputs, outputs, split = read_grid(GRID_FILE)
    train, val, test = (np.flatnonzero(split == label) for label in (0, 1, 2))

    # 2. Input transform: map each column of the training inputs onto [-1, 1]
    lower = inputs[train].min(axis=0)
    upper = inputs[train].max(axis=0)
    x = 2.0 * (inputs - lower) / (upper - lower) - 1.0

    # 3. Output transform: floor, then standardize with the training mean and spread
    floored = np.maximum(outputs, LOG10_FLOOR)
    mean = floored[train].mean(axis=0)
    std = floored[train].std(axis=0)
    y = (floored - mean) / std

    # 4. Starting weights and Adam's state (zeros shaped like every array)
    rng = np.random.default_rng(0)
    params = init_params(LAYER_SIZES, rng)
    state = {
        "m": [[np.zeros_like(value) for value in layer] for layer in params],
        "v": [[np.zeros_like(value) for value in layer] for layer in params],
        "t": 0,
    }

    # 5. Training loop: shuffle, then one Adam step per mini-batch
    best_loss, best_params = np.inf, None
    start_time = time.perf_counter()
    for epoch in range(N_EPOCHS):
        learning_rate = cosine_learning_rate(epoch, N_EPOCHS, LR_START, LR_END)
        order = rng.permutation(train)
        for begin in range(0, len(order), BATCH_SIZE):
            batch = order[begin : begin + BATCH_SIZE]
            y_pred, activations = forward(params, x[batch])
            grads = backward(params, activations, y_pred, y[batch])
            adam_step(params, grads, state, learning_rate)

        # 5a. Keep the weights with the lowest validation loss
        val_loss = mean_squared_error(params, x[val], y[val])
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = [[value.copy() for value in layer] for layer in params]
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
            print(
                f"epoch {epoch:3d}  lr {learning_rate:.1e}  validation loss {val_loss:.2e}  "
                f"({time.perf_counter() - start_time:.0f} s)"
            )

    # 6. Predict the test set and undo the output transform
    y_pred, _ = forward(best_params, x[test])
    log10_pred = np.maximum(y_pred * std + mean, LOG10_FLOOR)
    log10_true = floored[test]

    # 7. Errors in dex where the true or predicted abundance exceeds 1e-12
    abs_error = np.abs(log10_pred - log10_true)
    matters = np.maximum(log10_pred, log10_true) > -12.0
    medians = [
        np.median(abs_error[matters[:, k], k]) for k in range(abs_error.shape[1])
    ]
    p99s = [
        np.percentile(abs_error[matters[:, k], k], 99)
        for k in range(abs_error.shape[1])
    ]
    print(
        f"test set: median error {np.mean(medians):.4f} dex, 99th percentile {np.mean(p99s):.3f} dex"
    )


if __name__ == "__main__":
    main()
