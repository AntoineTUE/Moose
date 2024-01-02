# Examples of using `Moose`
`Moose` is meant to be simple, leaving it up to you how you'd like to integrate in your workflow.

To give you some inspiration on getting started, we have a number of examples.

These examples all demonstrate, how you could use `Moose` to fit emission spectra and (interactively) plot the results with different backends. 

!!! note "Examples have additional dependencies"
    Note however that these examples may rely on additional dependencies, that need to be installed in your environment if you want to execute them yourself.
    
    See the [getting started page](../get_started) on how to install these alongside `Moose`.


## Interactive plotting examples
Interactive viewing and fitting of data is a common scenario where `Moose` may be helpful.

These examples demonstrate how you could use different plotting backends to achieve this.

* [Interactive Bokeh](./bokeh)
* [Interactive Plotly](./plotly)
* [Interactive Matplotlib](./matplotlib)
* [Different fitting backends](./Moose_fitting_backends)
* [Parallel fitting with Dask](./dask)
* [Spectra with multiple species](./Multiple_species)

!!! note "Interactive features only work locally"
    While some degree of interaction is possible from your browser, these example notebooks must be run (locally) in a python environment to do everything.
    You can find these example notebooks and the used data in the `docs/examples` folder of the `git` repository.

Most of these examples rely on [`lmfit`](https://lmfit.github.io/lmfit-py/) to perform the fitting of the spectra.

The fitting is started via the `cb_fit` function, which is a callback that is invoked when pressing the `Fit` button in the simple UI that is created for most of the examples.