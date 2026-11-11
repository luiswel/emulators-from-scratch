"""Teaching package for the EXOMINTS 2026 chemistry emulator.

The package holds the reusable code. The notebooks in ``lessons/`` explain
it one step at a time and call these functions.

Modules
-------
practice_grid
    Make a stand-in FastChem grid with the same structure as the team grid.
data
    Load a grid, remove failed models, transform inputs and outputs, split.
baselines
    Simple emulators to beat: the mean, polynomials, nearest neighbours.
numpy_mlp
    A neural network in plain NumPy, with backpropagation written by hand.
jax_mlp
    The same network in JAX, trained with Adam; saving to disk.
evaluate
    Error statistics in dex and diagnostic plots.
trained
    ``TrainedEmulator``: load a saved emulator and predict abundances with
    NumPy only (no JAX needed to use it).
"""
