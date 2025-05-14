![Logo](assets/moose.svg)
# Moose: Molecular Optical Emission Simulation for python

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

Additionally, it has only a few, common dependencies (`numpy`, `scipy`, `pandas`) by default.

To get a better grasp of how it works under the hood, see the [code reference](reference/Moose)

## MassiveOES
The code that turned into `Moose`, was intended as a wrapper for [MassiveOES](https://bitbucket.org/OES_muni/massiveoes), to parallelize it with [Dask](https://dask.org).

However, the basic desired functionality was easier to implement separately in a stateless manner, rather than probably reworking the entire library.

The goal is not to provide the same functionality as `MassiveOES`, but instead provide a basis to build your own code on top of.

`Moose` still uses the same format as [MassiveOES](https://bitbucket.org/OES_muni/massiveoes) for the line-by-line databases, since it makes them *interoperable*.

You may be better served using `MassiveOES` or other programs, because of the following:

*   `Moose` does not give you tools to read, correct, or plot spectra
    *   There are better, more feature complete libraries for that
*   `Moose` lacks out-of-the-box fitting of multiple excited species not in equilibrium
    *   See the [examples](./examples) pages to see how you could use `Moose` for fitting
    *   Or simply build upon `Moose` for your own custom fitting routines
*  The provided fitting examples simply recalculate the simulated spectrum for each fit iteration, which sacrifices some speed. 
    *   The benefit is that it can be run in [parallel with ease](./examples/dask).
    *   If you don't have a huge set of very similar data or complex fits, this should not be an issue

 
