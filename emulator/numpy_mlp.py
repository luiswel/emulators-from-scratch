"""A neural network in plain NumPy, trained by hand.

Nothing is hidden here. The forward pass is a loop of matrix products, the
backward pass is the chain rule written out layer by layer, and training is
plain mini-batch stochastic gradient descent. Lesson 3 uses this module.
Lesson 4 swaps the hand-written gradient for JAX's automatic one and checks
that the two agree.

Shapes
------
Data are stored with one model per row, so a layer computes
``z = a @ W + b`` with ``a`` of shape (n_batch, n_in), ``W`` of shape
(n_in, n_out) and ``b`` of shape (n_out,).

Network
-------
For layer sizes [n_0, n_1, ..., n_L]:

    a_0 = x
    a_l = tanh(a_{l-1} @ W_l + b_l)        for l = 1 ... L-1 (hidden layers)
    y   = a_{L-1} @ W_L + b_L               (linear output layer)
"""

import numpy as np


def _check_layer_sizes(layer_sizes):
    """Raise an error unless ``layer_sizes`` describes a valid network.

    Parameters
    ----------
    layer_sizes : sequence of int
        Number of units per layer, inputs first and outputs last.

    Raises
    ------
    ValueError
        If there are fewer than two layers or any size is below 1.
    """
    # 1. At least an input and an output layer
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes needs at least [n_inputs, n_outputs]")

    # 2. Every layer has at least one unit
    if min(layer_sizes) < 1:
        raise ValueError(f"every layer needs at least one unit, got {layer_sizes}")


def init_params(
    layer_sizes,
    seed=0,
):
    """Draw random starting weights for a fully connected network.

    Parameters
    ----------
    layer_sizes : sequence of int
        Units per layer, e.g. [6, 64, 64, 19] for 6 inputs, two hidden layers
        of 64 units and 19 outputs.
    seed : int, default 0
        Random seed. The same seed always gives the same weights.

    Returns
    -------
    params : list of dict
        One dict per layer with ``"W"`` (ndarray, shape (n_in, n_out)) and
        ``"b"`` (ndarray, shape (n_out,)).

    Notes
    -----
    Weights are drawn from Normal(0, 1/n_in) (LeCun initialisation). With
    this spread, ``a @ W`` has roughly the same size as ``a`` whatever the
    layer width, so the signal neither explodes nor dies out as it passes
    through the layers. Biases start at zero.
    """
    # 1. Check the layer sizes
    _check_layer_sizes(layer_sizes)

    # 2. Draw each layer's weights and biases
    rng = np.random.default_rng(seed)
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        weights = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))
        biases = np.zeros(n_out)
        params.append({"W": weights, "b": biases})

    return params


def count_params(params):
    """Count the trainable numbers in a network.

    Parameters
    ----------
    params : list of dict
        Network weights (see ``init_params``).

    Returns
    -------
    n_params : int
        Total number of weights and biases.
    """
    # 1. Sum the sizes of all arrays
    n_params = int(sum(layer["W"].size + layer["b"].size for layer in params))

    return n_params


def forward(
    params,
    x,
):
    """Run inputs through the network.

    Parameters
    ----------
    params : list of dict
        Network weights (see ``init_params``).
    x : ndarray, shape (n_batch, n_inputs)
        Scaled inputs.

    Returns
    -------
    y_pred : ndarray, shape (n_batch, n_outputs)
        Network outputs (standardized log10 mixing ratios).
    activations : list of ndarray
        The input to every layer: ``activations[0] = x`` and
        ``activations[l]`` is the output of hidden layer l, shape
        (n_batch, n_l). The backward pass needs these.
    """
    # 1. Hidden layers: linear map, then tanh
    activations = [x]
    a = x
    for layer in params[:-1]:
        a = np.tanh(a @ layer["W"] + layer["b"])
        activations.append(a)

    # 2. Output layer: linear map only, so outputs can take any value
    y_pred = a @ params[-1]["W"] + params[-1]["b"]

    return y_pred, activations


def mean_squared_error(
    y_pred,
    y_true,
):
    """Mean of the squared differences over every model and every output.

    Parameters
    ----------
    y_pred : ndarray, shape (n_batch, n_outputs)
        Network outputs.
    y_true : ndarray, shape (n_batch, n_outputs)
        Target outputs.

    Returns
    -------
    loss : float
        mean((y_pred - y_true)**2).
    """
    # 1. Average squared residual
    loss = float(np.mean((y_pred - y_true) ** 2))

    return loss


def backward(
    params,
    activations,
    y_pred,
    y_true,
):
    """Gradient of the mean squared error with respect to every parameter.

    Parameters
    ----------
    params : list of dict
        Network weights used in the forward pass.
    activations : list of ndarray
        Second output of ``forward`` (the input to every layer).
    y_pred : ndarray, shape (n_batch, n_outputs)
        First output of ``forward``.
    y_true : ndarray, shape (n_batch, n_outputs)
        Target outputs.

    Returns
    -------
    grads : list of dict
        Same structure as ``params``: ``grads[l]["W"]`` is dLoss/dW_l and
        ``grads[l]["b"]`` is dLoss/db_l.

    Notes
    -----
    ``delta`` (shape (n_batch, n_units)) holds dLoss/dz for the current
    layer, where z = a @ W + b is the layer's value before the tanh. The
    chain rule gives, going from the last layer to the first:

    - output layer: delta = 2 (y_pred - y_true) / (n_batch * n_outputs)
    - weights:      dLoss/dW = a_in.T @ delta
    - biases:       dLoss/db = delta summed over the batch
    - previous z:   delta_prev = (delta @ W.T) * (1 - a_in**2)

    The last line uses d tanh(z)/dz = 1 - tanh(z)**2 = 1 - a_in**2.
    """
    # 1. Derivative of the loss with respect to the network output
    delta = 2.0 * (y_pred - y_true) / y_true.size

    # 2. Walk back through the layers, last to first
    grads = [None] * len(params)
    for index in reversed(range(len(params))):
        a_in = activations[index]

        # 2a. Gradients of this layer's weights and biases
        grads[index] = {"W": a_in.T @ delta, "b": delta.sum(axis=0)}

        # 2b. Send delta to the previous layer (through W and the tanh)
        if index > 0:
            delta = (delta @ params[index]["W"].T) * (1.0 - a_in**2)

    return grads


def numerical_gradient(
    params,
    x,
    y_true,
    step=1e-6,
):
    """Gradient by central finite differences, to check ``backward``.

    Parameters
    ----------
    params : list of dict
        Network weights. They are nudged one at a time and restored.
    x : ndarray, shape (n_batch, n_inputs)
        Scaled inputs.
    y_true : ndarray, shape (n_batch, n_outputs)
        Target outputs.
    step : float, default 1e-6
        Size of the nudge applied to each parameter.

    Returns
    -------
    grads : list of dict
        Same structure as ``params``, each entry approximated by
        [Loss(theta + step) - Loss(theta - step)] / (2 step).

    Notes
    -----
    This needs two forward passes per parameter, so it is only usable on a
    tiny network. It is the standard test that a hand-written backward pass
    is right: the two gradients should agree to about 1e-7 (relative).
    """
    # 1. Loop over every parameter array and every entry in it
    grads = []
    for layer in params:
        layer_grads = {}
        for key in ("W", "b"):
            values = layer[key]
            gradient = np.zeros_like(values)
            for position in np.ndindex(values.shape):
                original = values[position]

                # 1a. Loss with the entry nudged up, then down
                values[position] = original + step
                loss_up = mean_squared_error(forward(params, x)[0], y_true)
                values[position] = original - step
                loss_down = mean_squared_error(forward(params, x)[0], y_true)

                # 1b. Restore the entry and store the slope
                values[position] = original
                gradient[position] = (loss_up - loss_down) / (2.0 * step)

            layer_grads[key] = gradient
        grads.append(layer_grads)

    return grads


def train_sgd(
    params,
    x_train,
    y_train,
    x_val,
    y_val,
    learning_rate=0.1,
    batch_size=256,
    n_epochs=50,
    seed=0,
):
    """Train with plain mini-batch stochastic gradient descent.

    Parameters
    ----------
    params : list of dict
        Starting weights (not modified; a copy is trained).
    x_train : ndarray, shape (n_train, n_inputs)
        Scaled training inputs.
    y_train : ndarray, shape (n_train, n_outputs)
        Transformed training outputs.
    x_val : ndarray, shape (n_val, n_inputs)
        Scaled validation inputs.
    y_val : ndarray, shape (n_val, n_outputs)
        Transformed validation outputs.
    learning_rate : float, default 0.1
        Step size eta in theta <- theta - eta * gradient.
    batch_size : int, default 256
        Models per gradient step.
    n_epochs : int, default 50
        Passes through the training set.
    seed : int, default 0
        Seed for shuffling the training set each epoch.

    Returns
    -------
    params : list of dict
        Trained weights.
    history : dict of str to list of float
        ``"train_loss"`` and ``"val_loss"``, one value per epoch, computed
        on the full sets at the end of each epoch.
    """
    # 1. Work on a copy so the caller's starting weights stay untouched
    params = [{key: value.copy() for key, value in layer.items()} for layer in params]
    rng = np.random.default_rng(seed)
    n_train = x_train.shape[0]
    history = {"train_loss": [], "val_loss": []}

    # 2. Loop over epochs
    for epoch in range(n_epochs):

        # 2a. New random order of the training models every epoch
        order = rng.permutation(n_train)

        # 2b. One gradient step per mini-batch
        for start in range(0, n_train, batch_size):
            batch = order[start : start + batch_size]
            y_pred, activations = forward(params, x_train[batch])
            grads = backward(params, activations, y_pred, y_train[batch])
            for layer, layer_grads in zip(params, grads):
                layer["W"] -= learning_rate * layer_grads["W"]
                layer["b"] -= learning_rate * layer_grads["b"]

        # 2c. Record the loss on the full training and validation sets
        history["train_loss"].append(
            mean_squared_error(forward(params, x_train)[0], y_train)
        )
        history["val_loss"].append(mean_squared_error(forward(params, x_val)[0], y_val))

    return params, history
