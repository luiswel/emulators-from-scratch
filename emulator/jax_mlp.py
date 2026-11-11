"""The emulator network in JAX: automatic gradients, Adam and training.

JAX is NumPy with two additions that matter here:

- ``jax.grad`` turns a loss function into its gradient function, so the
  backward pass of ``numpy_mlp.backward`` is written for us;
- ``jax.jit`` compiles a function the first time it runs, so later calls
  are fast.

The network is the same as in ``numpy_mlp``: a list of dicts with ``"W"``
and ``"b"``, hidden layers with a smooth activation and a linear output.
Weights are float32, JAX's default.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from emulator import numpy_mlp

ACTIVATIONS = {"tanh": jnp.tanh, "silu": jax.nn.silu}
"""dict of str to callable: activation functions available to ``forward``."""


def init_params(
    layer_sizes,
    seed=0,
):
    """Random starting weights, identical to ``numpy_mlp.init_params``.

    Parameters
    ----------
    layer_sizes : sequence of int
        Units per layer, inputs first and outputs last.
    seed : int, default 0
        Random seed.

    Returns
    -------
    params : list of dict
        ``"W"`` (jax array, shape (n_in, n_out)) and ``"b"`` (shape (n_out,))
        per layer, float32.
    """
    # 1. Draw the weights with NumPy (same numbers as the NumPy lesson)
    numpy_params = numpy_mlp.init_params(layer_sizes, seed)

    # 2. Convert every array to a float32 JAX array
    params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, jnp.float32), numpy_params
    )

    return params


def forward(
    params,
    x,
    activation="tanh",
):
    """Run inputs through the network.

    Parameters
    ----------
    params : list of dict
        Network weights.
    x : array, shape (n_batch, n_inputs)
        Scaled inputs.
    activation : {"tanh", "silu"}, default "tanh"
        Nonlinearity of the hidden layers.

    Returns
    -------
    y_pred : jax array, shape (n_batch, n_outputs)
        Network outputs (standardized log10 mixing ratios).
    """
    # 1. Hidden layers: linear map, then the activation
    act = ACTIVATIONS[activation]
    a = x
    for layer in params[:-1]:
        a = act(a @ layer["W"] + layer["b"])

    # 2. Linear output layer
    y_pred = a @ params[-1]["W"] + params[-1]["b"]

    return y_pred


def loss_fn(
    params,
    x,
    y_true,
    activation="tanh",
):
    """Mean squared error of the network on a batch.

    Parameters
    ----------
    params : list of dict
        Network weights. ``jax.grad`` differentiates with respect to this
        first argument.
    x : array, shape (n_batch, n_inputs)
        Scaled inputs.
    y_true : array, shape (n_batch, n_outputs)
        Target outputs.
    activation : {"tanh", "silu"}, default "tanh"
        Nonlinearity of the hidden layers.

    Returns
    -------
    loss : jax scalar
        mean((y_pred - y_true)**2).
    """
    # 1. Predict, then average the squared residuals
    y_pred = forward(params, x, activation)
    loss = jnp.mean((y_pred - y_true) ** 2)

    return loss


def adam_init(params):
    """Starting state of the Adam optimizer.

    Parameters
    ----------
    params : list of dict
        Network weights (only their shapes are used).

    Returns
    -------
    state : dict
        ``"m"`` : running mean of the gradients (zeros, same structure as
        ``params``); ``"v"`` : running mean of the squared gradients (zeros);
        ``"step"`` : number of updates done so far (0).
    """
    # 1. Zeros shaped like every parameter array
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    state = {"m": zeros, "v": zeros, "step": jnp.array(0, dtype=jnp.int32)}

    return state


def adam_update(
    params,
    grads,
    state,
    learning_rate,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """One Adam step (Kingma & Ba 2015).

    Parameters
    ----------
    params : list of dict
        Current weights.
    grads : list of dict
        Gradient of the loss, same structure as ``params``.
    state : dict
        Optimizer state from ``adam_init`` or the previous call.
    learning_rate : float
        Largest step any single weight takes (roughly).
    beta1 : float, default 0.9
        Memory of the gradient average (about 1 / (1 - beta1) = 10 steps).
    beta2 : float, default 0.999
        Memory of the squared-gradient average (about 1000 steps).
    epsilon : float, default 1e-8
        Avoids dividing by zero.

    Returns
    -------
    new_params : list of dict
        Updated weights.
    new_state : dict
        Updated optimizer state.

    Notes
    -----
    For every weight separately:

        m = beta1 m + (1 - beta1) g          (smoothed gradient: direction)
        v = beta2 v + (1 - beta2) g**2       (smoothed squared gradient: scale)
        m_hat = m / (1 - beta1**t)           (undo the bias from starting at 0)
        v_hat = v / (1 - beta2**t)
        w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

    Dividing by sqrt(v_hat) gives every weight a step of about
    ``learning_rate``, whether its gradient is large or tiny. Plain gradient
    descent instead uses one step size for all weights.
    """
    # 1. Update the running averages
    step = state["step"] + 1
    m = jax.tree_util.tree_map(
        lambda m_, g: beta1 * m_ + (1 - beta1) * g, state["m"], grads
    )
    v = jax.tree_util.tree_map(
        lambda v_, g: beta2 * v_ + (1 - beta2) * g**2, state["v"], grads
    )

    # 2. Bias corrections for the first few steps
    m_correction = 1 - beta1**step
    v_correction = 1 - beta2**step

    # 3. Move every weight
    new_params = jax.tree_util.tree_map(
        lambda w, m_, v_: w
        - learning_rate * (m_ / m_correction) / (jnp.sqrt(v_ / v_correction) + epsilon),
        params,
        m,
        v,
    )
    new_state = {"m": m, "v": v, "step": step}

    return new_params, new_state


def cosine_learning_rate(
    epoch,
    n_epochs,
    lr_start=1e-3,
    lr_end=1e-5,
):
    """Learning rate that decays from ``lr_start`` to ``lr_end`` along a cosine.

    Parameters
    ----------
    epoch : int
        Current epoch, starting at 0.
    n_epochs : int
        Total number of epochs.
    lr_start : float, default 1e-3
        Learning rate at the first epoch.
    lr_end : float, default 1e-5
        Learning rate at the last epoch.

    Returns
    -------
    learning_rate : float
        Large steps early to make fast progress, small steps late to settle
        into the minimum.
    """
    # 1. Fraction of training done, from 0 to 1
    progress = epoch / max(n_epochs - 1, 1)

    # 2. Half a cosine period between the two rates
    learning_rate = lr_end + 0.5 * (lr_start - lr_end) * (
        1.0 + np.cos(np.pi * progress)
    )

    return float(learning_rate)


def make_train_step(
    activation="tanh",
    optimizer="adam",
):
    """Build the compiled function that performs one training step.

    Parameters
    ----------
    activation : {"tanh", "silu"}, default "tanh"
        Nonlinearity of the hidden layers.
    optimizer : {"adam", "sgd"}, default "adam"
        ``"adam"`` uses ``adam_update``; ``"sgd"`` is the plain gradient
        step of Lesson 3, w <- w - learning_rate * gradient.

    Returns
    -------
    train_step : callable
        ``train_step(params, state, x_batch, y_batch, learning_rate)``
        returns ``(new_params, new_state, loss)``. It is compiled by
        ``jax.jit`` on its first call.
    """
    # 1. Check the optimizer name
    if optimizer not in ("adam", "sgd"):
        raise ValueError(f"optimizer must be 'adam' or 'sgd', got {optimizer!r}")

    # 2. Loss with the activation fixed, so JAX only sees arrays
    loss_with_activation = partial(loss_fn, activation=activation)

    # 3. Gradient (and value) of the loss with respect to params
    value_and_grad = jax.value_and_grad(loss_with_activation)

    # 4. One step: gradient, then the optimizer's update
    def step(
        params,
        state,
        x_batch,
        y_batch,
        learning_rate,
    ):
        """Single optimizer step on one batch.

        Parameters
        ----------
        params : list of dict
            Current weights.
        state : dict
            Optimizer state (Adam's running averages; unused by SGD).
        x_batch : jax array, shape (n_batch, n_inputs)
            Scaled inputs of the batch.
        y_batch : jax array, shape (n_batch, n_outputs)
            Transformed outputs of the batch.
        learning_rate : float
            Step size for this step.

        Returns
        -------
        new_params : list of dict
            Updated weights.
        new_state : dict
            Updated optimizer state.
        loss : jax scalar
            Batch loss before the update.
        """
        loss, grads = value_and_grad(params, x_batch, y_batch)
        if optimizer == "adam":
            new_params, new_state = adam_update(params, grads, state, learning_rate)
        else:
            new_params = jax.tree_util.tree_map(
                lambda w, g: w - learning_rate * g, params, grads
            )
            new_state = state
        return new_params, new_state, loss

    train_step = jax.jit(step)

    return train_step


def train(
    params,
    x_train,
    y_train,
    x_val,
    y_val,
    n_epochs=200,
    batch_size=512,
    lr_start=1e-3,
    lr_end=1e-5,
    activation="tanh",
    optimizer="adam",
    seed=0,
    print_every=10,
):
    """Train with Adam and a cosine schedule; keep the best validation weights.

    Parameters
    ----------
    params : list of dict
        Starting weights (from ``init_params``).
    x_train : ndarray, shape (n_train, n_inputs)
        Scaled training inputs.
    y_train : ndarray, shape (n_train, n_outputs)
        Transformed training outputs.
    x_val : ndarray, shape (n_val, n_inputs)
        Scaled validation inputs.
    y_val : ndarray, shape (n_val, n_outputs)
        Transformed validation outputs.
    n_epochs : int, default 200
        Passes through the training set.
    batch_size : int, default 512
        Models per gradient step.
    lr_start : float, default 1e-3
        Learning rate at the first epoch.
    lr_end : float, default 1e-5
        Learning rate at the last epoch.
    activation : {"tanh", "silu"}, default "tanh"
        Nonlinearity of the hidden layers.
    optimizer : {"adam", "sgd"}, default "adam"
        Update rule (see ``make_train_step``). SGD needs a much larger
        learning rate than Adam (about 0.3 against 1e-3 here).
    seed : int, default 0
        Seed for shuffling the training set.
    print_every : int, default 10
        Print a progress line every this many epochs (0 for silence).

    Returns
    -------
    best_params : list of dict
        Weights from the epoch with the lowest validation loss.
    history : dict of str to list
        Per epoch: ``"train_loss"`` (mean batch loss during the epoch),
        ``"val_loss"``, ``"learning_rate"`` and ``"seconds"``.
    """
    # 1. Move the data to JAX once (float32)
    x_train = jnp.asarray(x_train, jnp.float32)
    y_train = jnp.asarray(y_train, jnp.float32)
    x_val = jnp.asarray(x_val, jnp.float32)
    y_val = jnp.asarray(y_val, jnp.float32)
    n_train = x_train.shape[0]
    n_batches = n_train // batch_size

    # 2. Optimizer state, compiled step and compiled validation loss
    state = adam_init(params)
    train_step = make_train_step(activation, optimizer)
    val_loss_fn = jax.jit(partial(loss_fn, activation=activation))
    rng = np.random.default_rng(seed)

    # 3. Bookkeeping for the best weights
    best_params = params
    best_val_loss = np.inf
    history = {"train_loss": [], "val_loss": [], "learning_rate": [], "seconds": []}

    # 4. Loop over epochs
    for epoch in range(n_epochs):
        start_time = time.perf_counter()
        learning_rate = cosine_learning_rate(epoch, n_epochs, lr_start, lr_end)

        # 4a. Shuffle, then one Adam step per mini-batch
        order = jnp.asarray(rng.permutation(n_train)[: n_batches * batch_size])
        batches = order.reshape(n_batches, batch_size)
        batch_losses = []
        for batch in batches:
            params, state, loss = train_step(
                params, state, x_train[batch], y_train[batch], learning_rate
            )
            batch_losses.append(loss)

        # 4b. Validation loss; keep these weights if they are the best so far
        val_loss = float(val_loss_fn(params, x_val, y_val))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

        # 4c. Record and report
        history["train_loss"].append(float(jnp.mean(jnp.stack(batch_losses))))
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(learning_rate)
        history["seconds"].append(time.perf_counter() - start_time)
        if print_every and (epoch % print_every == 0 or epoch == n_epochs - 1):
            print(
                f"epoch {epoch:4d}  lr {learning_rate:.1e}  "
                f"train {history['train_loss'][-1]:.2e}  val {val_loss:.2e}  "
                f"({history['seconds'][-1]:.1f} s)"
            )

    return best_params, history


def save_emulator(
    path,
    params,
    input_transform,
    output_transform,
    input_names,
    output_names,
    activation,
):
    """Write everything needed to use the emulator into one ``.npz`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file name (``.npz``).
    params : list of dict
        Trained weights.
    input_transform : dict
        Output of ``data.fit_input_transform``.
    output_transform : dict
        Output of ``data.fit_output_transform``.
    input_names : list of str
        Names of the inputs, in column order.
    output_names : list of str
        Names of the outputs (species), in column order.
    activation : str
        Activation used in training.

    Notes
    -----
    The file holds plain NumPy arrays, so ``trained.TrainedEmulator`` can
    load and evaluate it without JAX (in any Python environment).
    """
    # 1. Weights and biases, layer by layer
    arrays = {"n_layers": np.array(len(params))}
    for index, layer in enumerate(params):
        arrays[f"W_{index}"] = np.asarray(layer["W"])
        arrays[f"b_{index}"] = np.asarray(layer["b"])

    # 2. The transforms and the names
    arrays["input_log10_columns"] = input_transform["log10_columns"]
    arrays["input_lower"] = input_transform["lower"]
    arrays["input_upper"] = input_transform["upper"]
    arrays["output_log10_floor"] = np.array(output_transform["log10_floor"])
    arrays["output_mean"] = output_transform["mean"]
    arrays["output_std"] = output_transform["std"]
    arrays["input_names"] = np.array(input_names)
    arrays["output_names"] = np.array(output_names)
    arrays["activation"] = np.array(activation)

    # 3. Write the file
    np.savez(path, **arrays)
