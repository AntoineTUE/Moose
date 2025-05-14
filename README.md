![Logo](./docs/assets/moose.svg)
# Moose: Molecular optical emission spectroscopy for Python

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10454339.svg)](https://doi.org/10.5281/zenodo.10454339)
![GitHub License](https://img.shields.io/github/license/AntoineTUE/Moose)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/AntoineTUE/Moose/build.yml?label=PyPI%20build)
![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/AntoineTUE/Moose/documentation.yml?label=Documentation%20build)
![PyPI python versions](https://img.shields.io/pypi/pyversions/moose-spectra.svg)
![PyPI Downloads](https://img.shields.io/pypi/dm/moose-spectra)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/zenodo/10.5281/zenodo.10454339/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

Welcome to `Moose`, a python package for simulating optical emission spectra for diatomic molecules.

`Moose` was born out of a need to fit rotational and vibrational temperatures for a large set of data.

For this it uses *line-by-line databases*, assuming a Boltzmann distribution for rotational and/or vibrational temperatures.

On top of that `Moose` is intended to be *minimal*: it provides you some basic tools to do just that, simulate some spectra.

It is up to you to read and sanitize experimental data, that you would like to fit based on these simulations, for instance.

In addition, `Moose` is quite *adaptable*: you can use it as a foundation for more elaborate models and analysis.

Put differently, `Moose` is aimed at helping you: you can integrate and adapt it to your workflow, rather than the reverse.

To get a better grasp of how it works under the hood, see the online documentation for [code reference](https://antoinetue.github.io/Moose/reference/Moose/Simulation/) and [examples](https://antoinetue.github.io/Moose/examples/).

## Supported bands

Since `Moose` reuses the databases from [MassiveOES](https://bitbucket.org/OES_muni/massiveoes), it supports the same bands for fitting.
Details can be found in [data_sources.txt](./Moose/data/data_sources.txt), and the [page with information on citing](https://antoinetue.github.io/Moose/citing/).

| **Molecule** | **Band** |
| --- | --- |
| OH | A-X |
| N2+ | B-X |
| NH | A-X |
| NO | B-X |
| N2 | C-B |
| C2 | Swan |
| CN | B-X |

## Dependencies

Dependencies are mainly specified in the [pyproject.toml](./pyproject.toml), the [requirements.txt](./requirements.txt) file is there solely to aid in deploying to Binder.
See below for a table of these dependencies, which will be installed along with `Moose` if they are not present yet.
Version numbers are just indications of the package versions that were used to develop the code, your mileage may vary using either older versions.
Part of the automated test suite checks for compatibility of `numpy` and `pandas` version `<= 2.0` and `>= 2.0`, to ensure these older versions remain supported.

Furthermore, even though the project does not import by default `lmfit` and contains no functions that specifically rely on it, the code contains convenience functions that were specifically written for usage with the `lmfit.Model` and `lmfit.Parameters` classes.
If `lmfit` is installed in the active environment, some additional convenience function will be imported.

These dependencies will be installed when you install `Moose` using `pip` or `uv`.

| **Package**  | **Version**  |
|---|---|
| Numpy  | >= 1.24.2  |
| Pandas | >= 1.5.3 |
| scipy | >= 1.11.1 |
| *lmfit* (optional) | >= *1.2.0* |

## Installation

To simply install `Moose` itself, running the following suffices:

```bash
pip install moose-spectra
```

Installing `Moose` with additional optional dependencies is described on the [documentation page on getting started](https://antoinetue.github.io/Moose/get_started/).

You can also install the lastest development version of `Moose`, which may not be released to PyPI yet, using this git repository directly:

```bash
pip install git+https://github.com/AntoineTUE/Moose.git
```

## Basic usage

A basic example demonstrating the usage is as follows.
It assumes that there is a `pandas` DataFrame (called `data`) containing several spectra normalized on the interval (0,1), with the first column being a shared wavelength axis.
It is important that the wavelength range over which we query the database (`wl_interval`) plus the padding range `wl_pad`, is larger than the experimental range plus the possible shift (`mu`) between simulation and experiment.

Extending the fitting over multiple cores/processes can be done by using for instance the excellent [Dask](https://dask.org/) library, via i.e. `client.map`.
More elaborate examples are available via the [online documentation](https://antoinetue.github.io/Moose/examples/).

```python
import Moose, lmfit

wl_interval = (320,345) # Wavelength interval over which to simulate the spectrum
db = Moose.query_DB('N2CB.db', wl_interval)


# Create the lmfit.Model and lmfit.Parameters object needed for the fit.
model = lmfit.Model(Moose.model_for_fit, sim_db=db)
params = lmfit.create_params(**Moose.default_params)
# Or use some default suggested parameters for thermal fitting, i.e. T_rot==T_vib
# params = lmfit.create_params(**Moose.thermal_default_params)

# Perform the fit
fits = []
for col in data.columns[1:]:
    fits.append(model.Fit(data[col].values, x=data['Wavelength'].values, params=params))

```

## Copyright notice from MassiveOES

The original software is massiveOES developed by the Masaryk University available from: <https://bitbucket.org/OES_muni/massiveoes>.

### Publications

VORÁČ, Jan; SYNEK, Petr; PROCHÁZKA, Vojtěch; HODER, Tomáš. State-by-state emission spectra fitting for non-equilibrium plasmas: OH spectra of surface barrier discharge at argon/water interface. Journal of Physics D: Applied Physics. 2017, 50(29), 294002. DOI: <https://doi.org/10.1088/1361-6463/aa7570>.

VORÁČ, Jan; SYNEK, Petr; POTOČŇÁKOVÁ, Lucia; HNILICA, Jaroslav; KUDRLE, Vít. Batch processing of overlapping molecular spectra as a tool for spatio-temporal diagnostics of power modulated microwave plasma jet. Plasma Sources Science and Technology 26.2 (2017), 025010. DOI: <https://doi.org/10.1088/1361-6595/aa51f0>.

#### Original sources for databases

| Database | Source | DOI |
| --- | --- | --- |
| OHAX |J. Luque and D.R. Crosley, J. Chem. Phys., 109, 439 (1998) | <https://dx.doi.org/10.1088/0022-3727/39/17/015> |
| OHAX | L.R. Williams and D.R. Crosley, J. Chem. Phys, 104, 6507 (1996) | <https://doi.org/10.1063/1.471371> |
| N$_2^+$BX |  J. Luque and D.R. Crosley, “LIFBASE: Database and Spectral Simulation Program (Version 1.5) ”, SRI International Report MP 99-009 (1999). |  |
| NHAX | Western C 2016 PGOPHER—a program for simulating rotational structure, Version 9.0.100, University of Bristol |  |
| NHAX |  Ram R and Bernath P 2010 J. Mol. Spectrosc. 260 115–9 | <https://doi.org/10.1016/j.jms.2010.01.006> |
| NHAX | Lents J 1973 J. Quant. Spectrosc. Radiat. Transfer 13 297–310 | <https://doi.org/10.1016/0022-4073(73)90061-7> |
| NHAX  | Seong J, Park J K and Sun H 1994 Chem. Phys. Lett. 228 443–50 | <https://doi.org/10.1016/j.jms.2010.01.006> |
| NOBX | J. Luque and D.R. Crosley, “LIFBASE: Database and Spectral Simulation Program (Version 1.5) ”, SRI International Report MP 99-009 (1999). |  |
| N$_2$CB | Nassar H, Pellerin S, Musiol K, Martinie O, Pellerin N and Cormier J 2004. Phys. D: Appl. Phys. 37 1904 | <https://doi.org/10.1088/0022-3727/37/14/005> |
| N$_2$CB | Laux C O and Kruger C H 1992 J. Quant. Spectrosc. Radiat. Transfer 48 9–24 | <https://doi.org/10.1016/0022-4073(92)90003-M> |
| N$_2$CB | Faure G and Shko’nik S 1998 J. Phys. D: Appl. Phys. 31 1212 | <https://doi.org/10.1088/0022-3727/31/10/013> |
| C$_2$ Swan | James S.A. Brooke and Peter F. Bernath and Timothy W. Schmidt and George B. Bacskay; Journal of Quantitative Spectroscopy and Radiative Transfer; 2013 | <https://doi.org/10.1016/j.jqsrt.2013.02.025> |
| C$_2$ Swan | Carbone, Emile and D'Isa, Federico and Hecimovic, Ante and Fantz, Ursel; PSST; 2020; | <https://doi.org/10.1088/1361-6595/ab74b4> |

## License information

This project is licensed under the MIT license.
See [LICENSE](./LICENSE).
