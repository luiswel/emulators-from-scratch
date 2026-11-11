"""Simple emulators that a neural network has to beat.

All functions work on transformed data: inputs scaled to [-1, 1] and outputs
as standardized log10 mixing ratios (see ``data.py``).

1. The mean: predict the training-set average for every model. Any useful
   emulator must do much better than this.
2. Polynomials: a smooth global fit, solved exactly by linear least squares.
3. Nearest neighbours: average the closest training models. This is close
   in spirit to interpolating in a precomputed table.
"""

from itertools import combinations_with_replacement

import numpy as np
from scipy.spatial import cKDTree


def predict_mean(
    y_train,
    n_query,
):
    """Predict the training-set mean of every output for every query point.

    Parameters
    ----------
    y_train : ndarray, shape (n_train, n_outputs)
        Training outputs.
    n_query : int
        Number of points to predict.

    Returns
    -------
    y_pred : ndarray, shape (n_query, n_outputs)
        The same row (the mean) repeated ``n_query`` times.
    """
    # 1. Average over the training set, then repeat the row
    mean = y_train.mean(axis=0)
    y_pred = np.tile(mean, (n_query, 1))

    return y_pred


def polynomial_features(
    x,
    degree,
):
    """Build every product of input columns up to a total degree.

    Parameters
    ----------
    x : ndarray, shape (n_points, n_inputs)
        Scaled inputs.
    degree : int
        Highest total degree. ``degree=2`` with inputs (a, b) gives
        [1, a, b, a*a, a*b, b*b].

    Returns
    -------
    features : ndarray, shape (n_points, n_features)
        One column per monomial. With n inputs there are
        (n + degree)! / (n! degree!) columns: 28 for 6 inputs at degree 2,
        210 at degree 4, 924 at degree 6.
    """
    # 1. List the monomials as tuples of column numbers, e.g. (0, 0, 3) = x0*x0*x3
    n_inputs = x.shape[1]
    monomials = [()]
    for order in range(1, degree + 1):
        monomials += list(combinations_with_replacement(range(n_inputs), order))

    # 2. Multiply the chosen columns together for every monomial
    features = np.ones((x.shape[0], len(monomials)))
    for column, monomial in enumerate(monomials):
        for index in monomial:
            features[:, column] *= x[:, index]

    return features


def fit_polynomial(
    x_train,
    y_train,
    degree,
):
    """Fit a polynomial of the inputs to every output by least squares.

    Parameters
    ----------
    x_train : ndarray, shape (n_train, n_inputs)
        Scaled training inputs.
    y_train : ndarray, shape (n_train, n_outputs)
        Transformed training outputs.
    degree : int
        Highest total degree of the polynomial.

    Returns
    -------
    coefficients : ndarray, shape (n_features, n_outputs)
        Weight of each monomial for each output.

    Notes
    -----
    The model is linear in its coefficients, so the best fit has a closed
    form (the normal equations); ``numpy.linalg.lstsq`` solves it stably.
    A neural network is not linear in its weights, which is why it needs an
    iterative optimizer instead.
    """
    # 1. Build the design matrix
    features = polynomial_features(x_train, degree)

    # 2. Solve min ||features @ coefficients - y_train||^2
    coefficients = np.linalg.lstsq(features, y_train, rcond=None)[0]

    return coefficients


def predict_polynomial(
    x,
    coefficients,
    degree,
):
    """Evaluate a fitted polynomial.

    Parameters
    ----------
    x : ndarray, shape (n_points, n_inputs)
        Scaled inputs.
    coefficients : ndarray, shape (n_features, n_outputs)
        Output of ``fit_polynomial``.
    degree : int
        The degree used in ``fit_polynomial``.

    Returns
    -------
    y_pred : ndarray, shape (n_points, n_outputs)
        Predicted transformed outputs.
    """
    # 1. Same monomials as in the fit, times the coefficients
    y_pred = polynomial_features(x, degree) @ coefficients

    return y_pred


def predict_nearest_neighbours(
    x_train,
    y_train,
    x_query,
    k=8,
):
    """Average the outputs of the k closest training models.

    Parameters
    ----------
    x_train : ndarray, shape (n_train, n_inputs)
        Scaled training inputs.
    y_train : ndarray, shape (n_train, n_outputs)
        Transformed training outputs.
    x_query : ndarray, shape (n_query, n_inputs)
        Scaled inputs to predict.
    k : int, default 8
        Number of neighbours. With k = 1 this is a nearest-neighbour lookup.

    Returns
    -------
    y_pred : ndarray, shape (n_query, n_outputs)
        Inverse-distance-weighted mean of the neighbours' outputs.

    Notes
    -----
    Distances are measured in the scaled input space, so one unit in
    temperature counts the same as one unit in any other input.
    """
    # 1. Index the training points so neighbour searches are fast
    tree = cKDTree(x_train)

    # 2. Find the k nearest training points of every query point
    distances, neighbours = tree.query(x_query, k=k)
    if k == 1:
        distances = distances[:, None]
        neighbours = neighbours[:, None]

    # 3. Weight each neighbour by 1 / distance (closer counts more)
    weights = 1.0 / np.maximum(distances, 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    # 4. Weighted average of the neighbours' outputs
    y_pred = np.einsum("qk,qko->qo", weights, y_train[neighbours])

    return y_pred
