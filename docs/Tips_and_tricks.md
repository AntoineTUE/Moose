# Tips and tricks

## Limit amount of line-by-line data for faster fitting

The default for the [`Moose.query_DB`][Moose.query_DB] function is to return all line-by-line data that is contained in the database.

However, if you are only interested in fitting a limited subset of this dataset, filtering this data will reduce the amount of lines used in the [`Moose.create_stick_spectrum`][Moose.create_stick_spectrum] function, which can greatly speed up function evaluation.

When using [`Moose.query_DB`][Moose.query_DB] you can specify a limit to the wavelength range (or max $J$ or $v$ for that matter), to limit the data returned from the query.

Alternatively, you can also query the full dataset, and filter it down by slicing the returned Dataframe, or using e.g. [`groupby`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html).

```python
db = Moose.query_DB("CNBX")
# limit to only transitions with $\Delta v = 0$
db_v0 = db[db.Dv == 0]
```

## Batch fitting a sequence of data

A common scenario you may have is that you have a sequence of spectra recorded to measure the evolution of $T_{\mathrm{rot}}$ during discharge ignition.

Assuming that subsequent spectra should be more-or-less similar, you can exploit this to gain good initial estimates for each fit in the sequence, by using the fitted parameters of the preceding fit.

!!!note
    You must still make sure that a Boltzmann distribution (and thus a temperature) is a valid assumption for the rotational and/or vibrational population distributions. A fit may look good, but be physically incorrect.

```python
import lmfit
import Moose

db = Moose.query_DB("N2CB")

data = [...]

params = lmfit.create_params(**Moose.default_params)
model = lmfit.Model(Moose.model_for_fit, sim_db=db, normalize=True)
results = []

params_fit = params.copy()  # make a copy to a variable that is updated in the loop
for x, y in data:
    result = model.fit(y, x=x, params=params_fit)
    params_fit = result.params.copy()
    results.append(result)
```
