# Installing `Moose`

`Moose` can be installed with `pip` to get the latest release, note that on [PyPI](https://pypi.org/project/moose-spectra/) it is known as `moose-spectra`:

``` bash
pip install moose-spectra
```

You can also install the lastest development version of `Moose`, which may not be released to PyPI yet, using this git repository directly:

```bash
pip install git+https://github.com/AntoineTUE/Moose.git
```

You can also install several optional dependencies alongside `Moose` for fitting spectra, interactive plotting, or locally updating these documentation pages.

To do that run one of the following commands:

``` bash
# installs lmfit dependency
pip install moose-spectra[fit]

# install doc page dependencies
pip install moose-spectra[docs]

# install dependencies for running the examples
pip install moose-spectra[examples]

# install the all the above dependencies
pip install moose-spectra[all] 
```

# How to use `Moose`
A basic example demonstrating how to simulate and fit a spectrum is as follows.

1. Query a database for the emission system of interest, which is done with `query_DB`. This function will look for databases in the location where `Moose` has been installed, but it can also be provided with a path to your own compatible database. See the [code reference](../reference/Moose) for more information.

2.  Create a sample spectrum and add noise to it. To do this, first call the `model_for_fit` function, which is a high level function to calculate a spectrum, given line positions, temperatures, broadening parameters, and so forth.

3. Next prepare to fit this simulated spectrum. Create an `lmfit.Model`, by wrapping the `model_for_fit` function and passing in the database as well to the wrapper. 

4. As initial guess and bounds for the fit parameters, create an `lmfit.Parameters` object. For simplicity you can set these with some default values using `Moose.default_params`.



```python
import Moose
import numpy as np
import lmfit

rng = np.random.default_rng(100)

db = Moose.query_DB('N2CB', wl=(320,390)) # Restrict to wavelength between 320 and 390 nm

x = np.linspace(320,390,2000)

simulated = Moose.model_for_fit(x,sigma=0.01,gamma=0.01,mu=0.5,T_rot=1000, T_vib=5000, sim_db=db)

y = rng.normal(simulated, 0.01)# add noise

model = lmfit.Model(Moose.model_for_fit, sim_db=db, independent_vars=["x"])
params = lmfit.create_params(**Moose.default_params)

result = model.fit(y, x=x, params=params)

result.plot(title='N2CB fit', datafmt='x')

```

For more examples see the [examples section](../examples)
