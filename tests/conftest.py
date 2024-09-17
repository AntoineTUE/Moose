import pytest
import sqlite3 as sql
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--lmfit",
        action="store_true",
        dest="lmfit",
        default=False,
        help="enable tests for integration testing with lmfit as fitting backend",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "lmfit: enable tests with lmfit")
    if not config.option.lmfit:
        config.option.markexpr = "not lmfit"


@pytest.fixture(scope="module")
def temp_db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("test-db").joinpath("test.db")
    with sql.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE lines (id INTEGER PRIMARY KEY, A REAL, B REAL, upper_state INTEGER, lower_state INTEGER, air_wavelength REAL, vacuum_wavelength REAL, wavenumber REAL, E_J REAL, J REAL, component TEXT, E_v REAL, v REAL, branch TEXT)"
        )
        cursor.execute("CREATE TABLE upper_states (id INTEGER PRIMARY KEY)")
        cursor.execute("CREATE TABLE lower_states (id INTEGER PRIMARY KEY)")
        cursor.execute(
            "INSERT INTO lines (A, B, upper_state, lower_state, air_wavelength, vacuum_wavelength, wavenumber, E_J, J, component, E_v, v) VALUES (1.0, 2.0, 1, 2, 500.0, 600.0, 700.0, 0.1, 1, 'comp', 0.2, 3)"
        )
        cursor.execute("INSERT INTO upper_states (id) VALUES (1)")
        cursor.execute("INSERT INTO lower_states (id) VALUES (2)")
        conn.commit()
    return db_path