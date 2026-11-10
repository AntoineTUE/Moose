import sqlite3 as sql
import pandas as pd

import pytest


def pytest_sessionstart(session):
    import numpy as np
    import pandas as pd

    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")


def pytest_addoption(parser):
    group = parser.getgroup("Custom test markers")
    group.addoption(
        "--lmfit",
        action="store_true",
        dest="lmfit",
        default=False,
        help="enable tests for integration testing with lmfit as fitting backend",
    )
    group.addoption(
        "--fit",
        action="store_true",
        dest="fit",
        default=False,
        help="enable tests that fit spectra for validation",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "lmfit: enable tests with lmfit")
    config.addinivalue_line("markers", "fit: enable tests that actually fit data with lmfit")
    parts = []
    if not getattr(config.option, "lmfit", False):
        parts.append("not lmfit")
    if not getattr(config.option, "fit", False):
        parts.append("not fit")
    if parts and not config.option.markexpr:
        config.option.markexpr = " and ".join(parts)
    elif parts:
        config.option.markexpr = f"({config.option.markexpr}) and " + " and ".join(parts)


@pytest.fixture(scope="module")
def temp_db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("test-db").joinpath("test.db")
    with sql.connect(db_path) as conn:
        cursor = conn.cursor()
        # mimic the database structure
        cursor.execute(
            "CREATE TABLE lines (id INTEGER PRIMARY KEY, A REAL, B REAL, upper_state INTEGER, lower_state INTEGER, air_wavelength REAL, vacuum_wavelength REAL, wavenumber REAL, branch TEXT)"
        )
        cursor.execute(
            "CREATE TABLE upper_states (id INTEGER PRIMARY KEY, E_J REAL, J INTEGER, component INTEGER, E_V FLOAT, v INTEGER)"
        )
        cursor.execute(
            "CREATE TABLE lower_states (id INTEGER PRIMARY KEY, E_J REAL, J INTEGER, component INTEGER, E_V FLOAT, v INTEGER)"
        )
        cursor.execute(
            "INSERT INTO lines (A, B, upper_state, lower_state, air_wavelength, vacuum_wavelength, wavenumber) VALUES (1.0, 2.0, 1, 2, 500.0, 600.0, 700.0)"
        )
        cursor.execute("INSERT INTO upper_states (id, E_J,J,E_v,v) VALUES (1, 10,1,100,0)")
        cursor.execute("INSERT INTO lower_states (id, E_J,J,E_v,v) VALUES (2,10,1,100,0)")
        conn.commit()
    return db_path


@pytest.fixture
def SPS_experimental_spectrum(request):
    marker = request.node.get_closest_marker("fit")
    active = True if marker is None else marker.kwargs.get("active", True) if marker.kwargs else True
    if not active:
        return None
    data = pd.read_csv("./tests/test_files/N2SPS_experimental_sample.txt", sep="\t", header=1, names=["wl", "I"])
    data["Norm"] = (data.I - data.I.min()) / (data.I.max() - data.I.min())
    return data


@pytest.fixture
def SPS_synthetic_spectrum(request):
    marker = request.node.get_closest_marker("fit")
    active = True if marker is None else marker.kwargs.get("active", True) if marker.kwargs else True
    if not active:
        return None
    data = pd.read_csv(
        "./tests/test_files/calc_N2C_spectrum_Trot1200_Tvib3000_slit0.1.txt", sep="\t", header=0, names=["wl", "I"]
    )
    data["Norm"] = (data.I - data.I.min()) / (data.I.max() - data.I.min())
    return data
