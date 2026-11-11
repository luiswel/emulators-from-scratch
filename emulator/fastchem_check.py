"""Run FastChem exactly as the team grid did, to test the emulator.

The grid file stores the three FastChem input tables (element abundances,
gas-phase and condensate equilibrium constants) and the generator's
settings, so any model can be recomputed: a full atmosphere profile, or
models the emulator has never seen.

How the grid builds element abundances from the six features
------------------------------------------------------------
- H and He stay solar.
- Every metal except C, N, O and S is scaled by 10**[M/H].
- The total number of C + N + O + S atoms per H is the solar value times
  10**[M/H], shared out with the ratios C/O, S/O and N/O:
  O = total / (1 + C/O + S/O + N/O), C = O * C/O, S = O * S/O, N = O * N/O.
"""

import json
import os

import h5py
import numpy as np
import pyfastchem

BOLTZMANN_CGS = 1.380649e-16
"""float: Boltzmann constant in erg / K."""


class GridFastChem:
    """FastChem set up with the tables and settings of the team grid.

    Parameters
    ----------
    grid_path : str or pathlib.Path
        The grid HDF5 file (it contains the tables).
    tables_dir : str or pathlib.Path
        Folder where the tables are written as text files for FastChem.

    Attributes
    ----------
    fastchem : pyfastchem.FastChem
        The FastChem object (gas phase + condensates).
    config : dict
        The grid generator's settings.
    solar : ndarray, shape (n_elements,)
        Solar element abundances (n_X / n_H) in FastChem's element order.
    element_index : dict of str to int
        Position of each element symbol in FastChem's element list.
    species_index : list of int
        Position of each output species in FastChem's gas species list.
    """

    def __init__(
        self,
        grid_path,
        tables_dir,
    ):
        """Write the tables, start FastChem and apply the grid's settings."""
        # 1. Read the tables, the settings and the species list from the grid
        with h5py.File(grid_path, "r") as file:
            tables = {
                key: file[f"input_tables/{key}"][()]
                for key in ("abundances", "gas", "condensates")
            }
            self.config = json.loads(file.attrs["config_json"])
            species_symbols = [name.decode() for name in file["species_symbols"][:]]

        # 2. Write each table to a text file FastChem can read
        os.makedirs(tables_dir, exist_ok=True)
        paths = {}
        for key, content in tables.items():
            paths[key] = os.path.join(tables_dir, f"{key}.dat")
            with open(paths[key], "wb") as handle:
                handle.write(content)

        # 3. Start FastChem (verbose level 0 = quiet) with the grid's solver settings
        self.fastchem = pyfastchem.FastChem(
            paths["abundances"], paths["gas"], paths["condensates"], 0
        )
        for name, value in self.config["solver_parameters"].items():
            self.fastchem.setParameter(name, value)

        # 4. Look up where every element and output species sits in FastChem's lists
        self.solar = np.array(self.fastchem.getElementAbundances())
        self.element_index = {
            self.fastchem.getElementSymbol(i): i
            for i in range(self.fastchem.getElementNumber())
        }
        self.species_index = [
            self.fastchem.getGasSpeciesIndex(symbol) for symbol in species_symbols
        ]
        if pyfastchem.FASTCHEM_UNKNOWN_SPECIES in self.species_index:
            raise ValueError("a grid species is missing from FastChem's gas table")

    def element_abundances(
        self,
        features,
    ):
        """Element abundances (n_X / n_H) for one composition.

        Parameters
        ----------
        features : ndarray, shape (6,)
            log10 T, log10 P, [M/H], log10 C/O, log10 S/O, log10 N/O. Only
            the last four are used.

        Returns
        -------
        abundances : ndarray, shape (n_elements,)
            Abundances in FastChem's element order (see module docstring).
        """
        # 1. Scale every metal with [M/H]; H and He stay solar
        scale = 10.0 ** features[2]
        abundances = self.solar * scale
        for element in ("H", "He", "e-"):
            abundances[self.element_index[element]] = self.solar[
                self.element_index[element]
            ]

        # 2. Share the scaled C + N + O + S budget out with the three ratios
        c_o, s_o, n_o = 10.0 ** features[3], 10.0 ** features[4], 10.0 ** features[5]
        budget = scale * sum(
            self.solar[self.element_index[e]] for e in ("C", "N", "O", "S")
        )
        oxygen = budget / (1.0 + c_o + s_o + n_o)
        abundances[self.element_index["O"]] = oxygen
        abundances[self.element_index["C"]] = oxygen * c_o
        abundances[self.element_index["S"]] = oxygen * s_o
        abundances[self.element_index["N"]] = oxygen * n_o

        return abundances

    def compute(
        self,
        features,
        condensation=None,
    ):
        """Run FastChem and return outputs in the same format as the grid.

        Parameters
        ----------
        features : ndarray, shape (n_points, 6)
            log10 T [K], log10 P [bar], [M/H], log10 C/O, log10 S/O,
            log10 N/O. Rows that share the last four columns (one
            composition, e.g. an atmosphere profile) go to FastChem in a
            single call.
        condensation : bool, optional
            Equilibrium condensation on or off. Default None: as in the grid
            (on). Gas-only chemistry is much faster but is not what the
            emulator was trained on.

        Returns
        -------
        log10_outputs : ndarray, shape (n_points, 19)
            log10 mixing ratios of the 18 species (floored like the grid),
            then log10 of the mean molecular weight [u].
        converged : ndarray of bool, shape (n_points,)
            True where FastChem converged and conserved every element.
        """
        # 1. Set up the outputs and FastChem's input/output containers
        features = np.atleast_2d(features)
        log10_floor = np.log10(self.config["vmr_floor"])
        log10_outputs = np.full(
            (features.shape[0], len(self.species_index) + 1), np.nan
        )
        converged = np.zeros(features.shape[0], dtype=bool)
        compositions, which = np.unique(features[:, 2:], axis=0, return_inverse=True)
        which = np.ravel(which)

        # 2. One FastChem call per composition
        for number, composition in enumerate(compositions):
            rows = np.where(which == number)[0]
            temperature = 10.0 ** features[rows, 0]
            pressure = 10.0 ** features[rows, 1]

            # 2a. Element abundances, temperature, pressure, condensation settings
            self.fastchem.setElementAbundances(
                self.element_abundances(features[rows[0]])
            )
            input_data = pyfastchem.FastChemInput()
            output_data = pyfastchem.FastChemOutput()
            input_data.temperature = temperature
            input_data.pressure = pressure
            if condensation is None:
                condensation = bool(self.config["condensation"])
            input_data.equilibrium_condensation = bool(condensation)
            input_data.rainout_condensation = bool(self.config["rainout"])
            self.fastchem.calcDensities(input_data, output_data)

            # 2b. Mixing ratio = number density / total gas number density (P / k_B T)
            densities = np.array(output_data.number_densities)[:, self.species_index]
            gas_density = pressure * 1e6 / (BOLTZMANN_CGS * temperature)
            vmr = densities / gas_density[:, None]
            log10_outputs[rows, :-1] = np.maximum(
                np.log10(np.maximum(vmr, 1e-300)), log10_floor
            )
            log10_outputs[rows, -1] = np.log10(
                np.array(output_data.mean_molecular_weight)
            )

            # 2c. Convergence and element conservation
            flags = np.array(output_data.fastchem_flag) == pyfastchem.FASTCHEM_SUCCESS
            conserved = np.all(np.array(output_data.element_conserved) == 1, axis=1)
            converged[rows] = flags & conserved

        return log10_outputs, converged
