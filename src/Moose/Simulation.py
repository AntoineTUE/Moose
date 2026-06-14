"""
Provides functionality to simulate and fit spectra, akin to MassiveOES.

However (for now) it focusses on multiprocessing support and not introducing unnecessary package or package version dependencies.

Inspired by [MassiveOES](https://bitbucket.org/OES_muni/massiveoes/src/master/) and uses the underlying database files, compiled by J. Vorac and P. Synek.
"""

import sqlite3 as sql
import pandas as pd
import numpy as np
from importlib import resources
import pathlib
from scipy.special import voigt_profile
import scipy.integrate
import scipy.signal
import scipy.constants as const
from typing import Literal

from numpy.typing import NDArray

from .utils.maintenance import deprecated_keywords
from .utils.profiler import profile
from .utils.caching import array_cache
from .utils.db_io import get_database_path

kB = const.physical_constants["Boltzmann constant in inverse meters per kelvin"][0] / 100

default_params = {
    "sigma": {"value": 0.05, "min": 0.0001, "max": 0.3},
    "gamma": {"value": 0.05, "min": 0.0001, "max": 0.3},
    "mu": {"value": 0, "min": -2, "max": 2},
    "T_rot": {"value": 1000, "min": 250, "max": 10000},
    "T_vib": {"value": 1000, "min": 250, "max": 10000},
    "A": {"value": 1, "min": 0.2, "max": 2},
    "b": {"value": 0, "min": -0.05, "max": 0.05},
    "resolution": {"value": 100, "vary": False},
    "wl_pad": {"value": 10, "vary": False},
}

thermal_default_params = {
    "sigma": {"value": 0.05, "min": 0.0001, "max": 0.3},
    "gamma": {"value": 0.05, "min": 0.0001, "max": 0.3},
    "mu": {"value": 0, "min": -2, "max": 2},
    "T_rot": {"value": 1000, "min": 250, "max": 10000},
    "T_vib": {"value": 1000, "expr": "T_rot"},
    "A": {"value": 1, "min": 0.2, "max": 2},
    "b": {"value": 0, "min": -0.05, "max": 0.05},
    "resolution": {"value": 100, "vary": False},
    "wl_pad": {"value": 10, "vary": False},
}


@array_cache(maxsize=32)
def query_DB(
    db_name: str,
    wl: tuple = (0, 1e9),
    kind: Literal["emission", "absorption"] = "emission",
    mode: Literal["air", "vacuum"] = "air",
    v_max: int | None = None,
    J_max: int | None = None,
    path: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    """Query a SQL database that must contain line-by-line information, compatible with the format used by [MassiveOES](https://bitbucket.org/OES_muni/massiveoes).

    Args:
        db_name (str): The name of the database file to query.
        wl (tuple, optional): A wavelength range to constrain the query to. Defaults to (0,np.inf).
        kind (str, optional): The `kind` of spectrum that you want to create, either `emission` or `absorption`. The latter is not really tested. Defaults to 'emission'.
        mode (str, optional): A selection of the `mode` for wavelength, either in air or vacuum equivalent. Defaults to 'air'.
        v_max (int, optional): Maximum vibrational quantum number `v` for the query. Defaults to None.
        J_max (int, optional): Maximum rotational quantum number `J` for the query. Defaults to None.
        path (str, optional): The path to the folder containing database files. Defaults to the location of pre-packed databases.

    Raises:
        FileNotFoundError: If there is no database file with name `db_name` found in the location `path`.
        sql.DatabaseError: If the SQL query failed, due to incompatible database format, or errors in input

    Returns:
        A pandas DataFrame containing the result of the query.

    See also [create_stick_spectrum][Moose.Simulation.create_stick_spectrum]
    """
    path = pathlib.Path(path) if path is not None else get_database_path()
    db_name = db_name if db_name.endswith(".db") else f"{db_name}.db"
    wl_min, wl_max = map(float, wl) if wl is not None else (0.0, 1e9)
    db_path = path.joinpath(db_name)
    if not db_path.exists():
        errmsg = f'No such database, the file "{db_path.as_posix()}" was not found...'
        raise FileNotFoundError(errmsg)
    with db_path.open("rb") as f:
        header = f.read(100)
    if not header.startswith(b"SQLite format 3\x00"):
        errmsg = "File does not contain a valid SQL3 database..."
        raise sql.DatabaseError(errmsg)

    if kind.lower() == "emission":
        q_kind = "A"
        q_from_state = "upper"
        q_to_state = "lower"
    elif kind.lower() == "absorption":
        q_kind = "B"
        q_from_state = "lower"
        q_to_state = "upper"
    else:
        msg = f"Expected either 'emission' or 'absorption', got {kind}"
        raise ValueError(msg)
    if mode.lower() not in ["air", "vacuum"]:
        msg = f"`mode` should be either `air` or `vacuum`, got {mode}"
        raise ValueError(msg)

    query = f"""SELECT {q_kind}, upper_state, branch, vacuum_wavelength, air_wavelength, wavenumber, lower_state, 
    upper_states.E_J as E_J, upper_states.J as J, upper_states.component as component, upper_states.E_v as E_v, upper_states.v as v,
    lower_states.J as J_lower, lower_states.v as v_lower, lower_states.component as component_lower,
    (lower_states.v-upper_states.v) as Dv, (lower_states.J-upper_states.J) as DJ 
    FROM lines 
    INNER JOIN {q_from_state}_states on lines.{q_from_state}_state={q_from_state}_states.id
    INNER JOIN {q_to_state}_states on lines.{q_to_state}_state={q_to_state}_states.id
    WHERE lines.{mode.lower()}_wavelength between :wl_min and :wl_max
    """

    params = {"wl_min": wl_min, "wl_max": wl_max}
    if J_max is not None:
        params["Jmax"] = J_max
        query += f" and {q_from_state}_states.J<=:Jmax"
    if v_max is not None:
        params["vmax"] = v_max
        query += f" and {q_from_state}_states.v<=:vmax "

    query += f" ORDER BY {mode.lower()}_wavelength"

    with sql.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    return df


def create_stick_spectrum(
    T_vib: float,
    T_rot: float,
    pop: float = 1,
    df_db: pd.DataFrame = None,
    kind: Literal["Absorption", "Emission"] = "Emission",
    wl_mode: Literal["air", "vacuum"] = "air",
) -> NDArray[np.float64]:
    """Create a stick spectrum based on the data retrieved from a SQL database with the [query_DB][Moose.Simulation.query_DB] function.

    Alternatively, can be provided with any pandas DataFrame that has the requisite columns for the calculation.

    The Dataframe provided as `df_db` must have the columns: `["J","E_J",E_v"]`.

    In addition it must have one of either `["A","B"]` (the Einstein coefficients), and `["air_wavelength","vacuum_wavelength"]`.

    The stick spectrum is computed assuming a Boltzmann distribution for both the vibrational and rotational populations.

    Args:
        T_vib (float):          Vibrational temperature
        T_rot (float):          Rotational temperature
        pop (float, optional):  A population scaling factor.
        df_db (pd.DataFrame):   A pandas DataFrame containing the database data.
        kind (str):             Either 'Absorption' or 'Emission' depending on the kind of spectrum to simulate.
        wl_mode (str):          Either 'air' or 'vacuum' depending which equivalent we want for the wavelength.

    See also:
        * [query_DB][Moose.Simulation.query_DB]
        * [equidistant_mesh][Moose.Simulation.equidistant_mesh]
    """
    # Simply check for None, so other compatible objects can be passed in that don't look like a dataframe at first (dask delayed)
    if df_db is None:
        raise TypeError("No Dataframe with line-by-line data supplied.")
    sticks = np.zeros((df_db.shape[0], 2), np.float64)
    sticks[:, 0] = df_db[f"{wl_mode}_wavelength"]
    pops = (2 * df_db["J"] + 1) * np.exp(-df_db["E_v"] / (kB * T_vib) - df_db["E_J"] / (kB * T_rot))
    # Einstein B coefficient if kind=="Absorption" else Einstein A
    sticks[:, 1] = pop * pops / pops.sum() * (df_db["B"] if kind.startswith(("A", "a")) else df_db["A"])
    return sticks


@array_cache(maxsize=128)
def compute_equidistant_bins(wl: NDArray, pad: float = 10, resolution: int = 100) -> tuple[NDArray, NDArray]:
    """Compute an equidistant vector over the range of `wl`, at the specified resolution and with edges padded by `pad`.

    Returns a vector of the equidistantly spaces values and the indices that would sort `wl` into it.

    The input vector (or 1D array) `wl` does not need to be sorted.

    The indices are computed such that if they are used to insert elements into an array, the order would be preserved.

    For any index value `v`, that is used to insert into an array `a`, that: `a[i-1]<v<=a[i]`.

    See also [numpy.searchsorted][].

    Args:
        wl (NDArray): An input array of (potentially  unsorted) wavelenghts (or other values).
        pad (float, optional): An amount to pad the range of `wl` by. Defaults to 10.
        resolution (int, optional): Resolution in points per unit (of `x`) for the equidistant samples; spacing would be 1/resolution. Defaults to 100.

    Returns:
        A tuple of equidistant values, and the sorting indices that would sort `wl` between the equidistant points.
    """
    wl_min = wl.min() - pad
    wl_max = wl.max() + pad
    delta = wl_max - wl_min
    points = int(delta * resolution) + 1
    wl_equidistant = np.linspace(wl_min, wl_max, points)
    idx_right = np.searchsorted(wl_equidistant, wl, side="left")

    return wl_equidistant, idx_right


@array_cache(maxsize=128)
def equidistant_mesh(sim: NDArray[np.float64], wl_pad: float = 10, resolution: int = 100) -> NDArray[np.float64]:
    """Create an equidistant mesh from a (stick) simulation, where the mesh resolution per nanometer is controlled by the `resolution`.

    The simulated line intensities are rebinned onto the equidistant mesh by summing their values, if multiple lines fall into the same bin.

    In fact, each line contribution is binned to the two nearest bins, weighted by the linear distance between the line position and bins.

    This avoids discontinuities caused by lines jumping between bins for high resolution spectra and somewhat 'preserves' the information of the line position w.r.t. the bins.

    The operation preserves the sum of the intensities.

    Edge effects are avoided by extending the mesh beyond the input spectrum, with zero-padding for the intensities.

    This is controlled with the `wl_pad` argument, which specifies the amount of nanometer to pad.

    This padding is essential for later line-broadening (see [`apply_voigt`][..apply_voigt])

    Args:
        sim (np.array):     The 2D numpy array containing a simulation
        wl_pad (float):     The padding of the wavelength axis in nm to avoid edge effects
        resolution (int):   The resolution at which to construct the equidistant mesh (per nanometer) compared to the simulation (default: 100)

    Important:
        This function makes use of the [Moose.utils.caching.array_cache][].

        To get the best performance out of the cache, make sure that any input array is a C-contiguous array (check the array attribute `flags.c_contiguous`).

        If so, hashing is a zero-copy operation, but for a F-contiguous array a copy first needs to be made each time before hashing!

    Returns:
        A 2D array containing the mesh grid positions and corresponding stick values.

    See also:
      * [create_stick_spectrum][Moose.Simulation.create_stick_spectrum]
      * [apply_voigt][Moose.Simulation.apply_voigt]
    """
    wl = sim[:, 0]
    ys = sim[:, 1]

    wl_new, idx_right = compute_equidistant_bins(wl, wl_pad, resolution)
    idx_left = idx_right - 1
    bin_left = wl_new[idx_left]
    weights_right = (wl - bin_left) * resolution
    weights_left = 1 - weights_right

    equid = np.zeros((wl_new.size, 2), dtype=np.float64)
    equid[:, 0] = wl_new
    # Use np.add.at ufunc to accumulate at repeated indices
    # the alternative np.bincount (with `minlength=points`) appears slower for relevant use cases
    np.add.at(equid[:, 1], idx_right, weights_right * ys)
    np.add.at(equid[:, 1], idx_left, weights_left * ys)
    return equid


@array_cache()
def vgt(sigma: float, gamma: float, points: int, dx: float, truncate: None | float = None) -> NDArray:
    """Calculate a normalized Voigt profile, centered on a (equidistant) grid of size `points` with spacing `dx`.

    Essentially it computes a Voigt profile over a range of i.e. wavelength *change*, instead of wavelenght itself.

    This means it is scaled correctly to the data, but does not depend on the exact range of data.

    Because of this, it can be cached more effectively, as a fit may attempt to optimize a wavelength shift (param `mu`).

    Parameters such as the simulation resolution (`resolution`) and padding (`wl_pad`) are much less frequently optimized (or in need thereof).

    Note:
        Uses the standard deviation for the Gaussian width, while the Lorentzian Half-Width-at-Half-Maximum for Lorentzian width.
        These two widths are not fully equivalent in definition!
        Also note: if either broadening is 0, falls back to resp. a pure Gaussian or Lorentzian profile.
        This is to maintain consistency with the scipy implementation that is used, see [scipy.special.voigt_profile][].

    Important:
        This function makes use of the [Moose.utils.caching.array_cache][].

        To get the best performance out of the cache, make sure that any input array is a C-contiguous array (check the array attribute `flags.c_contiguous`).

        If so, hashing is a zero-copy operation, but for a F-contiguous array a copy first needs to be made each time before hashing!

    Args:
        sigma (float):          Gaussian broadening parameter, the standard deviation
        gamma (float):          Lorentzian broadening parameter, half width at half maximum
        points (int):           Amount of points in the grid to evaluate the Voigt at.
        dx (float):             The spacing between points in the grid.
        truncate (None|float):  If `None` (the default), don't truncate the range to calculate the Voigt over, else evaluate over a range $FWHM*truncate$ around center.

    Returns:
        A normalized and centered Voigt profile.
    """
    x_range = (np.arange(points) - (points - 1) / 2.0) * dx
    if truncate is not None:
        V = np.zeros_like(x_range)
        fwhm = 0.5343 * gamma * 2 + np.sqrt(0.2169 * (gamma * 2) ** 2 + (np.sqrt(8 * np.log(2)) * sigma) ** 2)
        mask = (x_range > -fwhm * truncate / 2) & (x_range < fwhm * truncate / 2)
        V[mask] = voigt_profile(x_range[mask], sigma, gamma)
    else:
        V = voigt_profile(x_range, sigma, gamma)
    return V / V.sum()


@deprecated_keywords("norm")
@array_cache(maxsize=128)
def apply_voigt(
    sim: NDArray, sigma: float, gamma: float, norm: bool | None = None, truncate: None | float = None
) -> NDArray:
    """Apply Voigt broadening to a simulated equidistant spectrum, preserving the integral and sum.

    The x-axis of the simulation `sim` must be an equidistant grid.

    See also [equidistant_mesh][Moose.Simulation.equidistant_mesh].

    Note that `sigma` is defined as the Gaussian standard deviation, while `gamma` is the Lorentzian Half-Width-at-Half-Maximum.

    That means these widths are not one-to-one comparable but should be converted to HWHM of FHWM.

    If either `gamma` or `sigma` are respectively 0, will apply solely `Gaussian` or `Lorentzian` broadening.

    Important:
        This function makes use of the [Moose.utils.caching.array_cache][].

        To get the best performance out of the cache, make sure that any input array is a C-contiguous array (check the array attribute `flags.c_contiguous`).

        If so, hashing is a zero-copy operation, but for a F-contiguous array a copy first needs to be made each time before hashing!

    Warning:
        Though the function accepts a "norm=True" keyword argument, this is considered deprecated behaviour and will not do anything.

        It will be removed in the future, which will cause an error to be thrown.

    Arguments:
        sim (np.array):         A (stick) simulation
        sigma (float):          The Gaussian sigma (standard deviation) for the voigt
        gamma (float):          The Lorentzian gamma (HWHM) for the voigt
        norm (bool):            DEPRECATED, has no effect ~Boolean to toggle normalizing (default: False)~

    Returns:
        A 2D array of the same shape as the input array `sim`, but convolved with a voigt profile.
    """
    x = sim[:, 0]
    dim = x.size
    step = x[dim // 2] - x[dim // 2 - 1]

    convolved = np.zeros_like(sim)
    convolved[:, 0] = sim[:, 0]

    v = vgt(sigma, gamma, points=x.size, dx=step, truncate=truncate)
    convolved[:, 1] = scipy.signal.fftconvolve(sim[:, 1], v, mode="same")
    # TODO: only return convolved y data, since x does not change? save memory/cpu impact.
    return convolved


def match_spectra(meas: np.array, sim: np.array, shift=0) -> np.ndarray:
    """Match a simulation to the same x-axis as the measurement using interpolation, with an optional shift.

    Make sure the simulation spans a larger range, fully containing the experimental range.

    If this is not the case, the missing data is assumed 0.

    Additionally, the x-axis of the simulation must be strictly increasing (see [numpy.interp][])

    Effectively downsamples the simulation to the measurement x data points, interpolating the y values, for residual minimization.

    Arguments:
        meas (np.array):    A 2D array containing a single measurement of emission as function of wavelength
        sim  (np.array):    A 2D array containing a simulated spectrum.
        shift (float):      Wavelength shift to apply in nanometer, default: 0.

    Returns:
        np.ndarray          :   A 2D array of the simulation, evaluated at the same grid coordinates as the measurement.
    """
    # TODO: implement first/second order corrections on the measured x data
    x_meas = meas if np.ndim(meas) == 1 else meas[:, 0]
    x_meas = x_meas - shift
    # Check that arg `xp` (of np.interp) is strictly increasing first, this is required but not checked by numpy.
    if np.diff(sim[:, 0]).min() < 0:
        raise ValueError("The x-axis of `sim` must be strictly increasing")
    matched_y = np.interp(x_meas, sim[:, 0], sim[:, 1], left=0, right=0)
    return np.column_stack((x_meas, matched_y))


def model_for_fit(
    x: np.array,
    sigma: float,
    gamma: float,
    mu: float,
    T_rot: float,
    T_vib: float,
    A: float = 1,
    b: float = 0,
    resolution: int = 100,
    wl_pad: float = 10,
    sim_db: pd.DataFrame = None,
    normalize: bool = True,
    **kwargs,
) -> NDArray[np.float64]:
    """Model function with function signature compatible for usage with [lmfit.model.Model][].

    Creates and broadens an equidistant stick spectrum from the provided simulation database.

    After broadening, resamples the simulation to the same coordinates as the (measured) data.

    Returns a spectrum normalized on the interval $(b,A+b)$, if `normalize=True`.

    If `normalize=False`, the argument `A` is used as a population scaling factor (the `pop` argument) for [create_stick_spectrum][..]

    Example usage:
    ```python
    import lmfit
    import Moose

    db = Moose.query_DB("N2CB")

    params = lmfit.create_params(**Moose.default_params)
    model = lmfit.Model(Moose.model_for_fit,normalize=True,sim_db=db, independent_vars=["x"])

    result = model.fit(data=...,x=..., params=params)
    ```

    Arguments:
        x (np.array):               The x-axis of the (measured) data that we want to compare/fit against, or want to evaluate the simulation at.
        sigma (float):              Gaussian broadening width of Voigt, the standard deviation.
        gamma (float):              Lorentzian broadening width of Voigt, the Half-Width-at-Half-Maximum.
        mu (float):                 The shift in x-coordinates between data and simulation, negative shift is towards longer wavelength
        T_rot (float):              The rotational temperature in Kelvin
        T_vib (float):              The vibrational temperature in Kelvin
        A (float):                  The amplitude scaling factor of the spectrum, if `normalize=True`, else a population scaling factor for the stick spectrum (default: 1)
        b (float):                  The offset w.r.t. 0 of the spectrum (default: 0)
        sim_db (DataFrame):         The DataFrame containing the database used for the simulation.
        wl_pad (float):             The amount of nanometer to pad the x-axis of the simulation with to avoid edge effects. Default: 10
        resolution (int):           The resolution per nanometer of  the equidistant mesh compared to bin/sample simulation by (default: 100)
        normalize (bool):           A flag to normalize the spectrum, before scaling by `A` and `b`.
        mode (str, optional):       The mode of the spectrum, i.e. 'Emission' versus 'Absorption' (default: Emission)
        wl_mode (str, optional):    Whether to use 'air' vs 'vacuum' wavelength (default: air)

    Returns:
        A vector representing the signal intensity calculated from the simulation, which can be used for a minimization/fitting procedure.
    """
    sticks = create_stick_spectrum(
        T_vib,
        T_rot,
        1 if normalize else A,
        df_db=sim_db,
        kind=kwargs.pop("mode", "Emission"),
        wl_mode=kwargs.pop("wl_mode", "air"),
    )
    refined = equidistant_mesh(sticks, wl_pad=wl_pad, resolution=resolution)
    simulation = apply_voigt(refined, sigma, gamma, truncate=kwargs.pop("truncate", None))
    sim_matched = match_spectra(x, simulation, shift=mu)
    val = sim_matched[:, 1]
    if normalize is True:
        # normalize to [0,1] rather than integral=1
        val = A * (val - val.min()) / (val.max() - val.min())
    return val + b
