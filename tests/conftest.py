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
