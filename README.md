# Emulators From Scratch

A course in building a neural-network emulator of FastChem equilibrium chemistry by hand, written at EXOMINTS 2026 by the emulators team. Every step is shown as equations, the NumPy code that implements them, and a numerical check: reading the grid, the transforms, least squares, the forward pass, backpropagation, Adam, the training loop, prediction from the saved weights and the emulator's derivatives. The final emulator reproduces FastChem with condensation to about 0.02 dex, about 280 times faster.

**Read the course online:** https://luiswel.github.io/emulators-from-scratch/

## Start here

```bash
git clone https://github.com/luiswel/emulators-from-scratch.git
```

```bash
cd emulators-from-scratch && conda env create -f environment.yml
```

```bash
conda activate emulator
```

The notebooks train on `training_250k.h5`, a grid of 250,000 FastChem models made by the EXOMINTS emulators team. The grid (464 MB) is not in this repository: team members can get it from the team and put it in this folder, next to this README. To use your own grid instead, change `data.load_grid` in `emulator/data.py`, which is the only function that knows the file layout. On Linux, delete the `libblas` line from `environment.yml` before creating the environment.

```bash
cd lessons && jupyter lab
```

Work through the notebooks in order, because each builds on the one before. Each runs top to bottom on a laptop, and the outputs are already saved in them, so you can read a lesson on GitHub before running anything.

| File | What it builds by hand | Run time |
|---|---|---|
| `lessons/01_the_data` | Reading the HDF5 grid, the split as indexing, the input transform (also as a matrix product), the output transform | 10 s |
| `lessons/02_baselines` | Scoring in dex, linear regression through the normal equations, polynomials through the SVD (and why conditioning needs scaled inputs), nearest neighbours with distances as one matrix product | 30 s |
| `lessons/03_numpy_network` | What hidden units do, LeCun initialization and why, the forward pass, backpropagation line by line, a finite-difference gradient check, the SGD training loop | 35 s |
| `lessons/04_jax_emulator` | What `jax.grad` computes (vector–Jacobian products), Adam built from its two running averages, the cosine schedule, the JAX training loop, short runs to choose settings, the final emulator | 7 min |
| `lessons/05_test_and_use` | Prediction from the saved arrays, the cost of a prediction, the test, where the emulator fails, speed against FastChem, atmosphere profiles, the Jacobian as a chain of matrix products | 10 s |
| `from_scratch/emulator_numpy.py` | All of the above in one file of NumPy, with no machine-learning library | 20 s |

Run times are for an M1 Max laptop, CPU only.

```bash
python from_scratch/emulator_numpy.py
```

## Results

Errors are in dex, averaged over the 19 outputs and counted where the true or predicted mixing ratio exceeds 1e-12 (Lesson 2 explains why). The first four rows use the validation set, the last two the test set.

| Emulator | Size | Training | Median error | 99th percentile |
|---|---|---|---|---|
| Training-set mean | 19 numbers | none | 9.5 | 15.4 |
| Polynomial, degree 5 | 462 terms per output | one SVD, 6 s | 0.72 | 3.0 |
| Nearest neighbours, k = 8 | the whole training set | none | 0.29 | 3.0 |
| NumPy network, 2 × 64 tanh, plain SGD (Lesson 3) | 5,843 weights | 20 epochs, 5 s | 0.29 | 1.9 |
| One-file NumPy emulator, 3 × 256 tanh, Adam (`from_scratch`) | 138,259 weights | 300 epochs, 11 min | 0.022 | 0.24 |
| JAX network, 3 × 256 tanh, Adam, floor 1e-15 (Lesson 4) | 138,259 weights | 300 epochs, 4 min | 0.022 | 0.24 |

Per species, the final emulator's median test error ranges from 0.0002 dex (He) to 0.049 dex (H2S), and its 99th percentile stays below 0.4 dex for every output except C2H2 (0.6 dex) and H2S (1.0 dex). The mean molecular weight is good to 0.0005 dex (0.1%). The trained emulator is a 559 kB file, against 464 MB for the grid.

For a 100-layer atmosphere on an M1 Max laptop, FastChem 4.0.4 built with -O3 took 0.115 s with condensation and 0.012 s without, and the emulator took 0.42 ms: about 280 and 30 times faster.

## What is where

- `emulator/`: the reusable code, which the lessons check their hand-written versions against.
  - `data.py`: load the grid, split it, fit and apply the input and output transforms.
  - `baselines.py`: mean, polynomial and nearest-neighbour emulators.
  - `numpy_mlp.py`: the network in NumPy, with the backward pass written by hand and a finite-difference check.
  - `jax_mlp.py`: the network in JAX, Adam, the cosine schedule, the training loop, `save_emulator`.
  - `evaluate.py`: errors in dex and the diagnostic plots.
  - `trained.py`: `TrainedEmulator`, which loads a saved emulator and predicts with NumPy only.
  - `fastchem_check.py`: `GridFastChem`, FastChem set up exactly as the grid was made (the tables are stored in the grid file), to test the emulator on new inputs.
- `lessons/`: the five notebooks, with their outputs.
- `from_scratch/emulator_numpy.py`: the whole emulator in one NumPy file.
- `runs/`: what the notebooks write. `emulator.npz` is the trained emulator from Lesson 4 and `baseline_scores.json` holds the Lesson 2 scores, both included so that later lessons run on their own.
- `docs/`: the course website (GitHub Pages).
- `environment.yml`: the conda environment.

The grid holds 250,000 models made with FastChem 4.0.4 by the team's `chemem` generator, with condensation on and rainout off. The models form a Latin hypercube in six log features (T 57 to 4000 K, P 1e-6 to 1e3 bar, [M/H] -1 to 2, C/O 0.1 to 1.5, S/O and N/O 0.001 to 0.5), their outputs are floored at 1e-30, and the file includes a train/validation/test split.

## Using the trained emulator

```python
import sys
sys.path.insert(0, "path/to/emulators-from-scratch")
from emulator.trained import TrainedEmulator

emulator = TrainedEmulator("path/to/emulators-from-scratch/runs/emulator.npz")
log10_values = emulator.predict_log10(features)
```

`features` has shape (n_layers, 6) with the grid's columns: log10 T [K], log10 P [bar], [M/H], log10 C/O, log10 S/O, log10 N/O. The 19 output columns are log10 mixing ratios of H, H2, H-, e-, He, H2O, CO, CO2, CH4, C2H2, Fe, TiO, SiO, HCN, NH3, N2, H2S, OH, then log10 of the mean molecular weight in u. Points outside the training range raise an error rather than returning an extrapolated number. `TrainedEmulator` needs only NumPy and h5py, so it also runs in older Python environments (3.8 or later). Lesson 5 shows the eight lines of NumPy it runs.

## FastChem on a Mac

Only Lesson 5 and `emulator/fastchem_check.py` use FastChem, to compare the emulator with fresh FastChem runs. The pyfastchem 4.0.4 wheel that `pip install pyfastchem` gives on Apple Silicon is compiled without optimization (-O0): FastChem's wheel workflow sets `CXXFLAGS`, and setuptools then uses those flags instead of Python's own `-O3`. It runs 10 to 40 times slower than an optimized build while giving identical results. To build FastChem 4.0.4 with `-O3` inside the `emulator` environment, linked to the environment's own OpenMP library:

```bash
conda activate emulator && python -m pip uninstall -y pyfastchem
```

```bash
git clone --branch v4.0.4 https://github.com/NewStrangeWorlds/FastChem.git && cd FastChem
```

```bash
env PATH="$CONDA_PREFIX/bin:/usr/bin:/bin:/usr/sbin:/sbin" CC=/usr/bin/clang CXX=/usr/bin/clang++ \
    CPPFLAGS="-Xclang -fopenmp -I$CONDA_PREFIX/include" CFLAGS="-O3 -DNDEBUG" CXXFLAGS="-O3 -DNDEBUG" \
    LDFLAGS="$CONDA_PREFIX/lib/libomp.dylib -Wl,-rpath,$CONDA_PREFIX/lib" \
    python -m pip install . --no-build-isolation --no-deps
```

`-O3` asks the compiler to optimize (inline small functions, keep values in registers, vectorize loops) and `-DNDEBUG` removes debugging checks. The `PATH` setting keeps Homebrew's older OpenMP library out of the build. One model on one core then takes under 1 ms gas-only above 300 K and 5 to 80 ms with condensation, depending on temperature. On Linux the wheel is probably optimized already (its build does not set `CXXFLAGS`), but that is unchecked.

## The ideas on one page

- An emulator is regression: pick a family of functions, a measure of misfit and a way to find the best member of the family, then check the result on models it never saw.
- Look at the data before fitting anything, because plateaus, switches, cliffs and floors decide how hard the problem is.
- Before fitting, take logs of anything that spans orders of magnitude, scale the inputs to [-1, 1] and standardize the outputs. Fit these transforms on the training set only and save them with the weights.
- Measure errors in dex, and only where they matter (here, where the true or predicted mixing ratio exceeds 1e-12).
- Models that are linear in their coefficients (polynomials) are solved in one step by least squares, best through the SVD. A network is not linear in its weights, so it is trained by gradient descent.
- A network layer is a matrix product and an element-wise tanh. Backpropagation is the chain rule written as matrix products, ∂L/∂W = AᵀΔ and Δ_prev = (ΔWᵀ) ⊙ (1 − A²), and it should be checked against finite differences.
- Train with Adam (a per-weight step size) and a decaying learning rate, and keep the weights with the best validation loss.
- Change one thing at a time with short runs, then train long once. Test the physically motivated ideas too, since 1/T as an input sounded right and made the emulator worse.
- Look at the test set once, at the end, and never use the emulator outside the range of its training data.

## Notes on the EXOMINTS grid

These come from runs on the grid's own validation set, with errors measured as in the results table.

- Train on a floor of 1e-15 rather than the grid's 1e-30. In 40-epoch runs of the 3 × 256 network the median error fell from 0.083 to 0.056 dex, and in 60-epoch runs the worst case fell from 14 to 8.7 dex. A floor of 1e-20 sits in between (300 epochs, median 0.0245 against 0.0222 dex, worst case 12.9 against 7.5 dex). Floors above 1e-15 are untested.
- Keep log10 T as the temperature input. Replacing it with 1000/T roughly doubled the median error (0.157 against 0.083 dex in 40-epoch runs), because the grid is uniform in log T and 1/T squeezes everything hotter than about 800 K into a tenth of the input range. Feeding both gave the same error as log T alone.
- tanh beat SiLU in the one comparison we ran (60 epochs with 1/T inputs, median 0.093 against 0.148 dex).
- Training for 300 epochs halved the median error of a 60-epoch run (0.022 against 0.045 dex), and the loss was still falling at the end. A wider network (512 units per layer) is the next thing to try.
- The remaining large errors sit at condensation edges. H2S is the hardest output, because below about 700 K the sulfur condenses into a succession of sulfides (mainly FeS, with FeS2, Ni3S2 and MnS) and each one adds a kink to the gas-phase H2S. All ten worst test predictions are H2S or NH3 below 225 K, with errors of 3.6 to 5.8 dex.
- With FastChem built with -O3, one model on one core takes under 1 ms gas-only above 300 K (18 ms below), and 5 ms above 1000 K, 19 ms between 300 and 1000 K and 79 ms below 300 K with condensation. The pip wheel on a Mac is 10 to 40 times slower with identical outputs, so the grid values are unaffected.
- The grid is valid only above about 57 K, because every FastChem attempt below that failed, and the emulator refuses inputs outside its training range.
- In the grid file, `species_names` calls both H and H2 "Hydrogen", so the code uses `species_labels`.
- Every number here uses the split stored in the grid file, so it can be compared directly with anyone else's results on this grid.

## Coding conventions and Black

The code follows a shared set of conventions: NumPy-style docstrings everywhere, numbered steps inside each function, one parameter per line with a trailing comma, input checks in private helpers, and named return values.

**Black** is a program that reformats Python files to a single standard layout (spacing, line breaks, quotes). It never changes what the code does, so everyone's code looks the same and differences between versions show only real changes. The trailing-comma rule exists because Black keeps a signature with a trailing comma split one parameter per line. To format everything (notebooks included), run this in the repository folder with the environment active:

```bash
python -m black .
```

## Credits

The training grid was computed with FastChem (Stock et al. 2018, 2022; Kitzmann et al. 2024) by the EXOMINTS 2026 emulators team. The emulator uses NumPy, SciPy, Matplotlib, h5py and JAX, and trains with Adam (Kingma & Ba 2015).

## License

MIT, see [LICENSE](LICENSE).
