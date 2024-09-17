import pytest
from Moose import Simulation
import pandas as pd
import numpy as np

from scipy.special import voigt_profile
from sqlite3 import DatabaseError
from importlib import resources
from pandas.testing import assert_frame_equal
from numpy.testing import (
    assert_array_equal,
    assert_array_almost_equal,
    assert_almost_equal,
)


class TestSimulation:
    rng = np.random.default_rng()

    @pytest.mark.parametrize("db_name", ["OHAX", "OHAX.db"])
    @pytest.mark.parametrize("interval", [(200, 900), (300, 320)])
    @pytest.mark.parametrize("v_max", [None, 10, 1])
    @pytest.mark.parametrize("J_max", [None, 100, 10])
    def test_query_DB(self, db_name, interval, v_max, J_max):
        db = Simulation.query_DB(db_name, interval, v_max=v_max, J_max=J_max)
        assert db.shape[0] > 0
        assert db["air_wavelength"].min() >= interval[0]
        assert db["air_wavelength"].max() <= interval[1]
        if v_max:
            assert db["v"].max() <= v_max
        if J_max:
            assert db["J"].max() <= J_max

    def test_query_DB_vacuum(self):
        db = Simulation.query_DB("OHAX", (300, 320), mode="vacuum")
        assert db["air_wavelength"].min() > 300
        assert db["air_wavelength"].max() > 320
        assert db["vacuum_wavelength"].min() >= 300
        assert db["vacuum_wavelength"].max() <= 320

    def test_query_DB_no_lambda_range(self):
        db_none = Simulation.query_DB("OHAX", None)
        db_unlimited = Simulation.query_DB("OHAX")
        assert_frame_equal(db_none, db_unlimited)

    @pytest.mark.parametrize("name", ["Unknown", "db.txt"])
    def test_query_DB_not_included(self, name):
        with pytest.raises(FileNotFoundError):
            Simulation.query_DB(name)

    def test_query_DB_local_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Simulation.query_DB("Unknown", path="./")

    def test_query_DB_not_SQL(self, tmp_path):
        invalid_db_path = tmp_path.joinpath("Moose_invalid_db.db")
        invalid_db_path.write_text("Moose test file of invalid database format")
        with pytest.raises(DatabaseError):
            Simulation.query_DB(invalid_db_path.name, path=tmp_path)

    @pytest.mark.parametrize(
        "species",
        [f.stem for f in resources.files("Moose").joinpath("data").glob("*.db")],
    )
    def test_query_DB_Absorption(self, species):
        with pytest.raises(pd.errors.DatabaseError):
            Simulation.query_DB(species, kind="absorption")

    def test_create_stick_spectrum(self):
        db = pd.DataFrame(
            {
                "J": [1, 1],
                "E_v": [0, 0],
                "E_J": [1, 1],
                "A": [1, 1],
                "air_wavelength": [300, 400],
            }
        )
        sticks = Simulation.create_stick_spectrum(300, 3000, db)
        assert_array_equal(sticks, np.array([[300.0, 0.5], [400, 0.5]]))
        assert sticks[:, 1].sum() == 1

        db = pd.DataFrame(
            {
                "J": [1, 1],
                "E_v": [0, 0],
                "E_J": [1, 50],
                "A": [1, 1],
                "air_wavelength": [300, 400],
            }
        )
        for t, result in zip(
            [30, 300, 3000],
            [
                [9.12934467e-01, 8.70655330e-02],
                [0.55848119, 0.44151881],
                [0.50587474, 0.49412526],
            ],
        ):
            sticks = Simulation.create_stick_spectrum(t, t, db)
            assert_array_almost_equal(sticks, np.array([[300.0, 400.0], result]).T)
            assert sticks[:, 1].sum() == 1

    @pytest.mark.parametrize("db", [None, np.zeros(10), {}])
    def test_create_stick_spectrum_not_valid_DB(self, db):
        with pytest.raises(TypeError):
            Simulation.create_stick_spectrum(300, 300, db)

    @pytest.mark.parametrize("points", [10, 1000, 10000])
    @pytest.mark.parametrize("resolution", [10, 1000])
    @pytest.mark.parametrize("pad", [1])
    def test_equidistant_mesh(self, points, resolution, pad):
        sim = np.array([np.linspace(0, 10, points), self.rng.random(points)])
        equid = Simulation.equidistant_mesh(sim, wl_pad=pad, resolution=resolution)
        assert_almost_equal(equid[:, 1].sum(), sim[:, 1].sum())
        assert_almost_equal(sim[:, 0].min() - pad, equid[:, 0].min())

    def test_vgt(self):
        x = np.linspace(-10, 10, 1000)
        assert_array_equal(Simulation.vgt(x, 0.2, 0.2, 0, 1, 0), voigt_profile(x, 0.2, 0.2))
        assert_array_almost_equal(Simulation.vgt(x, 0.1, 0.1, 0.1, 1, 0), voigt_profile(x - 0.1, 0.1, 0.1))

    def test_apply_voigt(self):
        sticks = np.array([np.linspace(-1, 1, 100), np.zeros(100)]).T
        sticks[50, 1] = 1
        assert_array_almost_equal(
            Simulation.apply_voigt(sticks, 0.1, 0.1, False)[:, 1],
            np.convolve(sticks[:, 1], voigt_profile(sticks[:, 0], 0.1, 0.1), mode="same"),
        )

    def test_match_spectra(self):
        sim = np.array([np.linspace(0, 100, 100), self.rng.random(100)]).T
        meas = np.array([np.linspace(0, 100, 70), self.rng.random(70)]).T

        matched = Simulation.match_spectra(meas, sim)
        assert matched.shape == meas.shape

        meas = np.array([np.linspace(-200, 200, 70), self.rng.random(70)]).T
        with pytest.raises(ValueError):
            Simulation.match_spectra(meas, sim)

    def test_match_with_equidistant_mesh_padding(self):
        sticks = np.array([np.linspace(100, 200, 1000), self.rng.random(1000)]).T
        meas = np.array([np.linspace(100 - 9, 200 + 9, 100), self.rng.random(100)]).T
        equid = Simulation.equidistant_mesh(sticks, wl_pad=10)
        matched = Simulation.match_spectra(meas, equid)
        assert matched[0, 0] > sticks[0, 0] - 10
        assert matched[-1, 0] < sticks[-1, 0] + 10
        equid = Simulation.equidistant_mesh(sticks, wl_pad=8)
        with pytest.raises(ValueError):
            matched = Simulation.match_spectra(meas, equid)

    def test_model_for_fit(self):
        db = pd.DataFrame(
            {
                "J": [1, 1],
                "E_v": [0, 0],
                "E_J": [1, 50],
                "A": [1, 1],
                "air_wavelength": [300, 400],
            }
        )
        outcome = np.array(
            [
                1.00000000e00,
                1.14824424e-02,
                2.01582383e-03,
                2.88194160e-04,
                7.67004068e-05,
                1.83496645e-05,
                0.00000000e00,
                1.36603964e-03,
                8.85238182e-03,
                7.90343912e-01,
            ]
        )
        simulated = Simulation.model_for_fit(
            np.linspace(300, 400, 10),
            T_rot=300,
            T_vib=300,
            sigma=1,
            gamma=1,
            mu=0,
            A=1,
            b=0,
            sim_db=db,
        )
        assert_array_almost_equal(simulated, outcome)
        with pytest.raises(TypeError):
            Simulation.model_for_fit(np.linspace(300, 400, 10), T_rot=300, T_vib=300, sigma=1, gamma=1, mu=0, A=1, b=0)


@pytest.mark.lmfit
class TestParameters:
    def test_default_parameters(self):
        import lmfit

        params = lmfit.create_params(**Simulation.default_params)
        assert isinstance(params, lmfit.Parameters)
        assert params["T_vib"].expr is None
        assert params["T_rot"].expr is None

    def test_thermal_parameters(self):
        import lmfit

        params = lmfit.create_params(**Simulation.thermal_default_params)
        assert isinstance(params, lmfit.Parameters)
        assert params["T_vib"].expr == "T_rot"
        assert params["T_rot"].expr is None
