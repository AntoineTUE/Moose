"""A suite of test that actually fit data, to check if these remain consistent."""

import pytest
import lmfit
import Moose
from Moose.lmfit import multi_species_objective

from numpy.testing import assert_allclose


@pytest.mark.fit
def test_fit_SPS_experimental(SPS_experimental_spectrum):
    data = SPS_experimental_spectrum
    db_n2 = Moose.query_DB("N2CB", (300, 400))
    model = lmfit.Model(Moose.model_for_fit, normalize=True, sim_db=db_n2)
    params = lmfit.create_params(**Moose.default_params)
    result = model.fit(data["Norm"].to_numpy(), x=data.wl.to_numpy(), params=params)

    assert_allclose(
        list(result.params.valuesdict().values()),
        [3.846e-2, 4.869e-2, 2.746e-1, 1.1803e3, 9.99999e3, 1.00787e0, 2.5823e-2, 100, 10],
        rtol=5e-3,
    )


@pytest.mark.fit
def test_fit_SPS_experimental_minimizer(SPS_experimental_spectrum):
    data = SPS_experimental_spectrum
    db_n2 = Moose.query_DB("N2CB", (300, 400))
    params = lmfit.create_params(**Moose.default_params)
    result = lmfit.minimize(
        multi_species_objective,
        params,
        kws={"x": data.wl.to_numpy(), "y": data["Norm"].to_numpy(), "N2": db_n2, "normalize": True},
    )

    assert_allclose(
        list(result.params.valuesdict().values()),
        [3.846e-2, 4.869e-2, 2.746e-1, 1.1803e3, 9.99999e3, 1.00787e0, 2.5823e-2, 100, 10],
        rtol=5e-3,
    )


@pytest.mark.fit
def test_fit_SPS_synthetic(SPS_synthetic_spectrum):
    data = SPS_synthetic_spectrum
    db_n2 = Moose.query_DB("N2CB", (300, 400))
    model = lmfit.Model(Moose.model_for_fit, normalize=True, sim_db=db_n2)
    params = lmfit.create_params(**Moose.default_params)
    params.add("A", 1, False)
    params.add("b", 0, False)
    result = model.fit(data["Norm"].to_numpy(), x=data.wl.to_numpy(), params=params)

    assert_allclose(
        list(result.params.valuesdict().values()),
        [4.151e-02, 1.0280e-04, -1.344e-01, 1.190e03, 3.027e03, 1, 0, 100, 10],
        rtol=5e-3,
    )


@pytest.mark.fit
def test_fit_SPS_synthetic_minimizer(SPS_synthetic_spectrum):
    data = SPS_synthetic_spectrum
    db_n2 = Moose.query_DB("N2CB", (300, 400))
    params = lmfit.create_params(**Moose.default_params)
    params.add("A", 1, False)
    params.add("b", 0, False)
    result = lmfit.minimize(
        multi_species_objective,
        params,
        kws={"x": data.wl.to_numpy(), "y": data["Norm"].to_numpy(), "N2": db_n2, "normalize": True},
    )

    assert_allclose(
        list(result.params.valuesdict().values()),
        [4.151e-02, 1.0280e-04, -1.344e-01, 1.190e03, 3.027e03, 1, 0, 100, 10],
        rtol=5e-3,
    )


@pytest.mark.fit
def test_fit_SPS_experimental_minimizer_benchmark(SPS_experimental_spectrum, benchmark):
    """Benchmark of fit performance"""
    data = SPS_experimental_spectrum
    db_n2 = Moose.query_DB("N2CB", (300, 400))
    params = lmfit.create_params(**Moose.default_params)
    result = benchmark(
        lmfit.minimize,
        multi_species_objective,
        params,
        kws={"x": data.wl.to_numpy(), "y": data["Norm"].to_numpy(), "N2": db_n2, "normalize": True},
    )

    assert_allclose(
        list(result.params.valuesdict().values()),
        [3.846e-2, 4.869e-2, 2.746e-1, 1.1803e3, 9.99999e3, 1.00787e0, 2.5823e-2, 100, 10],
        rtol=5e-3,
    )


@pytest.mark.fit
def test_fit_SPS_synthetic_minimizer_benchmark(SPS_synthetic_spectrum, benchmark):
    data = SPS_synthetic_spectrum
    db_n2 = Moose.query_DB("N2CB", (300, 400))
    params = lmfit.create_params(**Moose.default_params)
    params.add("A", 1, False)
    params.add("b", 0, False)
    result = benchmark(
        lmfit.minimize,
        multi_species_objective,
        params,
        kws={"x": data.wl.to_numpy(), "y": data["Norm"].to_numpy(), "N2": db_n2, "normalize": True},
    )

    assert_allclose(
        list(result.params.valuesdict().values()),
        [4.151e-02, 1.0280e-04, -1.344e-01, 1.190e03, 3.027e03, 1, 0, 100, 10],
        rtol=5e-3,
    )
