![Logo](assets/moose.svg)
# Moose: Molecular Optical Emission Simulation for python
Welcome to `Moose`, a python package for simulating optical emission spectra for diatomic molecules.

`Moose` was born out of a need to fit rotational and vibrational temperatures for a large set of data.

For this it uses *line-by-line databases*, assuming a Boltzmann distribution for rotational and/or vibrational temperatures.

On top of that `Moose` is intended to be *minimal*: it provides you some basic tools to do just that, simulate some spectra.

It is up to you to read and sanitize experimental data, that you would like to fit based on these simulations, for instance.

Put differently, `Moose` is aimed at helping you: you can integrate and adapt it to your workflow, rather than the reverse.

To get a better grasp of how it works under the hood, see the [code reference](reference/Moose)

## MassiveOES
The code that turned into `Moose`, was intended as a wrapper for [MassiveOES](https://bitbucket.org/OES_muni/massiveoes), to parallelize it with [Dask](https://dask.org).

However, the basic desired functionality was easier to implement separately in a stateless manner, rather than probably reworking the entire library.

The goal is not to provide the same functionality as `MassiveOES`, but instead provide a basis to build your own code on top of.

`Moose` still uses the same format as [MassiveOES](https://bitbucket.org/OES_muni/massiveoes) for the line-by-line databases, since it makes them *interoperable*.

You may be better served using `MassiveOES` or other programs, because of the following:

*   `Moose` lacks out-of-the-box fitting of multiple excited species not in equilibrium
*   `Moose` simply recalculates the simulated spectrum for each fit iteration, which sacrifices some speed. 
    *   The benefit is that it can be run in parallel with ease.
*   `Moose` does not give you tools to read, correct, or plot spectra
    *   There are better, more feature complete libraries for that
 