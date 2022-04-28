# Moose: a Massive OES wrapper

A Library for fitting optical emission spectra, assuming a Boltzmann distribution for rotational and vibrational temperatures.
For the simulations it uses the databases compiled by the [MassiveOES](https://bitbucket.org/OES_muni/massiveoes) project, with the corresponding references mentioned in [data_sources.txt](./Moose/data/data_sources.txt). 
Though applying similar principles as MassiveOES, this project was written for multi-process batch fitting of spectra, whilst providing a simple API that does not assume much about the users workflow. 
This also avoids unnessecary dependencies (esp. on specific outdated versions), that might cause conflicts in environments.
As a consequence, the code should be easily integrated into existing workflows, at the cost of not providing conveniece functions.
As it started out as a thin wrapper for MassiveOES, it's name was inspired on that project.
However, the basic desired functionality was easier to implement separately in a stateless manner, rather than basically reworking the entire library.

Also do note that it ultimately provides less rigorous treatment options for spectra than what MassiveOES includes out of the box, such as non-linear baseline correction or quadratic wavelength axis correction.
Such functionality can be implemented by extending the provided methods and model in combination with [lmfit](https://lmfit.github.io/lmfit-py/) or even using it's [Composite Model](https://lmfit.github.io/lmfit-py/model.html#composite-models-adding-or-multiplying-models) feature.

## Supported bands
Since `Moose` reuses the databases from [MassiveOES](https://bitbucket.org/OES_muni/massiveoes), it supports the same bands for fitting.
Details can be found in [data_sources.txt](./Moose/data/data_sources.txt).

| **Molecule** | **Band** |
| --- | --- |
| OH | A-X |
| N2+ | B-X |
| NH | A-X |
| NO | B-X |
| N2 | C-B |
| C2 | Swan |


## Dependencies
The following packages are dependencies to be able to use `Moose`. 
Version numbers are just indications of the package versions that were used to develop the code, your mileage may vary using either older or newer versions.
Furthermore, even though the project does not import `lmfit` and contains no functions that specifically rely on it, the code contains convenience functions that were specifically written for usage with the `lmfit.Model` and `lmfit.Parameters` classes. 

| **Package**  | **Version**  |
|---|---|
| Numpy  | 1.21.5  |
| Pandas | 1.4.0 |
| scipy | 1.7.3 |
| *lmfit* | *1.0.3* |




## Installation
Install it using either `pip` or `conda` by running them from the project directory.
Installing in editable/development mode an be achieved with eiter:

`pip install -e ./`

or 

`conda develop .`

## Basic usage
A basic example demonstrating the usage is as follows.
It assumes that there is a `pandas` DataFrame (called `data`) containing several spectra normalized on the interval (0,1), with the first column being a shared wavelength axis.
It is important that the wavelength range over which we query the database (`wl_interval`), is larger than the experimental range plus the possible shift (`mu`) between simulation and experiment.

Extending the fitting over multiple cores/processes can be done by using for instance the excellent [Dask](https://dask.org/) library, via i.e. `client.map`.

```python
import Moose, lmfit

wl_interval = (320,345) # Wavelength interval over which to simulate the spectrum
db = Moose.query_DB('N2CB.db', wl_interval)


# Create the lmfit.Model and lmfit.Parameters object needed for the fit.
model = lmfit.Model(Moose.model_for_fit, sim_db=db)
params = model.make_params()
Moose.set_params(params,{'sigma':{'value':0.15, 'min':0.1, 'max':0.4}, 'gamma': {'value':0.15,'min':0.1,'max': 0.4}, 'T_rot': {'value': 1000, 'min': 300, 'max': 10000}, 'T_vib': {'value': 1000, 'min': 300, 'max': 10000},'mu': {'value': 0, 'min':-2, 'max':2}, 'A': {'value': 1, 'min': 0.2, 'max': 2}, 'b': {'value':0, 'min': -0.05,'max': 0.05}}, print=True)

# Perform the fit
fits = []
for col in data.columns[1:]:
    fits.append(model.Fit(data[col].values, x=data['Wavelength'].values, params=params))

```


## Copyright notice from MassiveOES

The original software is massiveOES developed by the Masaryk University available from: https://bitbucket.org/OES_muni/massiveoes.

### Publications

VORÁČ, Jan; SYNEK, Petr; PROCHÁZKA, Vojtěch; HODER, Tomáš. State-by-state emission spectra fitting for non-equilibrium plasmas: OH spectra of surface barrier discharge at argon/water interface. Journal of Physics D: Applied Physics. 2017, 50(29), 294002. DOI: https://doi.org/10.1088/1361-6463/aa7570.


VORÁČ, Jan; SYNEK, Petr; POTOČŇÁKOVÁ, Lucia; HNILICA, Jaroslav; KUDRLE, Vít. Batch processing of overlapping molecular spectra as a tool for spatio-temporal diagnostics of power modulated microwave plasma jet. Plasma Sources Science and Technology 26.2 (2017), 025010. DOI: https://doi.org/10.1088/1361-6595/aa51f0.


