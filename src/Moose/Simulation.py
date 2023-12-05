"""
Provides functionality to simulate and fit spectra, akin to MassiveOES.

However (for now) it focusses on multiprocessing support and not introducing unneccesary package or package version dependencies.

Inspired by [MassiveOES](https://bitbucket.org/OES_muni/massiveoes/src/master/) and uses the underlying database files, compiled by J Vorac and P. Synek.
"""

import sqlite3 as sql
import pandas as pd
import numpy as np
import pkg_resources as pkg
import pathlib
from scipy.special import voigt_profile
import scipy.integrate
import scipy.signal
import scipy.interpolate
import scipy.constants as const
from scipy.stats import binned_statistic
from typing import Literal

_default_params = {
    'sigma':{'value':0.05, 'min':0.0001, 'max':0.3}, 
    'gamma': {'value':0.05,'min':0.0001,'max': 0.3},
    'mu': {'value': 0, 'min':-2, 'max':2}, 
    'T_rot': {'value': 1000, 'min': 250, 'max': 10000}, 
    'T_vib': {'value': 1000, 'min': 250, 'max': 10000},
    'A': {'value': 1, 'min': 0.2, 'max': 2}, 
    'b': {'value':0, 'min': -0.05,'max': 0.05},
    'resolution': {'value':100, 'vary':False},
    'wl_pad': {'value':10 ,'vary': False } 
    }

def query_DB(db_name:str, wl:tuple=(0,np.inf), kind:str='emission',mode:str='air', v_max=None, J_max=None, path:str=pkg.resource_filename('Moose', 'data')) -> pd.DataFrame:
    """Queries a SQL database that must contain line-by-line information, compatible with the format used by [MassiveOES](https://bitbucket.org/OES_muni/massiveoes).

    Args:
        db_name (str): _description_
        wl (tuple, optional): _description_. Defaults to (0,np.inf).
        kind (str, optional): _description_. Defaults to 'emission'.
        mode (str, optional): _description_. Defaults to 'air'.
        v_max (_type_, optional): _description_. Defaults to None.
        J_max (_type_, optional): _description_. Defaults to None.
        path (str, optional): _description_. Defaults to pkg.resource_filename('Moose', 'data').

    Raises:
        FileNotFoundError: _description_
        sql.DatabaseError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    if '.db' not in db_name:
        db_name += '.db'
    wl_min, wl_max = wl
    db_path = pathlib.Path(path).joinpath(db_name)
    if not db_path.exists():
        raise FileNotFoundError('No such database, the file "{}" was not found...'.format(db_path))
    with db_path.open('rb') as f:
        header = f.read(100)
    if header[:16]!=b'SQLite format 3\x00':
        raise sql.DatabaseError('File does not contain a valid SQL3 database...')
    else:
        conn = sql.connect(db_path)

    if kind=='emission':
        q_kind = 'A'
        q_join = 'upper_states on upper_state=upper_states.id'
    elif kind =='absorption':
        q_kind='B'
        q_join = 'lower_states on lower_state=lower_states.id'
        
    if not J_max:
        q_j = ''
    else:
        q_j = f' and J <= {J_max}'
    if not v_max:
        q_v = ''
    else:
        q_v = f' and v <= {v_max}'
        

    q_mode='{}_wavelength'.format(mode) # vacuum vs air wavelength equivalent

    if wl_min != 0 and wl_max != np.inf:
        q_wl = f" where lines.{q_mode} between {wl_min} and {wl_max}{q_j}{q_v}"
    else:
        q_wl = ''
        
    query = f"SELECT lines.id, {q_kind}, upper_state, branch, vacuum_wavelength, air_wavelength, wavenumber, lower_state, E_J, J, component, E_v, v from lines inner join {q_join}{q_wl} ORDER BY {q_mode}"
    df = pd.read_sql_query(query,conn)
    conn.close()
    
    return df

def create_stick_spectrum(T_vib:float,T_rot:float,df_db:pd.DataFrame=None, mode:Literal['Absorption','Emission']='Emission', wl_mode: Literal['air', 'vacuum']='air'):
    """Create a stick spectrum based on the data retrieved from a SQL database.
    
    Arguments:
        T_vib:          Vibrational temperature
        T_rot:          Rotational temperature
        df_db:          A pandas DataFrame containing the database data
        mode:           Either 'Absorption' or 'Emission' depending on mode we want to use
        wl_mode:        Either 'air' or 'vacuum' depending which equivalent we want.
    """

    if type(df_db) == type(None):
        raise TypeError('No Dataframe with database data supplied as kwarg')
    kB=const.physical_constants['Boltzmann constant in inverse meters per kelvin'][0]/100
    pops = (2*df_db['J']+1)*np.exp(-df_db['E_v']/(kB*T_vib)-df_db['E_J']/(kB*T_rot))
    # pops/= scipy.integrate.trapezoid(pops,df_db['{}_wavelength'.format(wl_mode)])
    pops /= pops.sum()
    if mode=='Emission':
        y = pops*df_db['A']
    elif mode == 'Absorption':
        y = pops*df_db['B']
    return np.array([df_db['{}_wavelength'.format(wl_mode)], y]).T

def equidistant_mesh(sim:np.array, wl_pad: float=10, resolution:int=100):
    '''Creates an equidistant, mesh from a simulation, where the mesh resolution per nanometer is controlled by the `resolution`.
    
    The simulated line strengths are rebinned onto the equidistant mesh by summing their values.
    
    Arguments:
        sim:            The 2D numpy array containing a simulation
        wl_pad:         The padding of the wavelength axis in nm to avoid edge effects
        resolution:     The resolution at which to construct the equidistant mesh (per nanometer) compared to the simulation (default: 100)
    '''
    # points = int((sim[-1,0]-sim[0,0]+2*wl_pad)/(sim[1,0]-sim[0,0])*factor+1)
    delta = sim[-1,0]-sim[0,0] +2*wl_pad
    points = int(delta*resolution)
    equid = np.linspace(sim[0,0].min()-wl_pad, sim[:,0].max()+wl_pad, points)
    binned, _, _ = binned_statistic(sim[:,0], sim[:,1], statistic='sum', bins=equid)
    wl_grid = equid[:-1]+(equid[1]-equid[0])/2 # grid at middle points of intervals
    
    return np.array([wl_grid,binned]).T
    

def vgt (x:np.array,sigma:float,gamma:float,mu:float,a:float,b:float):
    """Voigt profile implementation, thinly wraps the scipy implementation.

    Args:
        x (np.array): the x-axis array for the voigt profile
        sigma (float): Gaussian broadening parameter, the standard deviation
        gamma (float): Lorentzian broadening parameter, half width at half maximum
        mu (float): Shift parameter with respect to the center of the x-axis, in the same units as `x`.
        a (float): Amplitude scaling factor
        b (float): Offset with respect to 0 of the values.

    Returns:
        np.array: Voigt profile as a function of `x`
    """
    return a*voigt_profile(x-mu,sigma,gamma)+b

def apply_voigt(sim:np.array,sigma:float, gamma:float, norm:bool=False):
    '''Applies Voigt broadening to a simulated stick spectrum, optionally normalizing the surface to 1.
    
    To avoid repeated (different) normalisations from being used while fitting, it defaults to False.
    
    Arguments:
        sim:        A (stick) simulation
        sigma:      The Gaussian sigma for the voigt
        gamma:      The Lorentzian gamma (HWHM) for the voigt
        norm:       Boolean to toggle normalizing (default: False)
    '''
    x = sim[:,0]
    dim = int(len(x))
    if dim % 2 ==0:
        mu = (x[int(dim/2)-1]+x[int(dim/2)])/2
    else:
        mu = x[int(dim/2)]

    v = vgt(x,sigma,gamma,mu,1,0)
    conv = scipy.signal.fftconvolve(sim[:,1], v, mode='same')
    if norm: 
        conv /= scipy.integrate.trapezoid(conv,x)
    
    return np.array([x,conv]).T

def match_spectra(meas: np.array, sim: np.array):
    '''Matches a simulation to the same x-axis as the measurement using interpolation.
    
    Make sure the simulation spans a larger range, fully containing the experimental range.
    
    Effectively downsamples the simulation to the measurement x data points, interpolating the y values, for residual minimization
    
    Arguments:
        meas (np.array)   :   A 2D array containing a single measurement of emission as function of wavelength
        sim  (np.array)   :   A 2D array containing a simulated spectrum.
        
    Returns:
        - np.array          :   A 2D array of the simulation, evaluated at the same grid coordinates as the measurement.
    '''

    interp=scipy.interpolate.interp1d(sim[:,0], sim[:,1])
    try:
        matched_y=interp(meas[:,0])
    except ValueError as e:
       raise ValueError(f"Wavelength outside of interpolation range, try setting a larger value for `wl_pad` kwarg:\n{e.args}")
    return np.array([meas[:,0], matched_y]).T

def model_for_fit(x:np.array,sigma:float,gamma:float, mu:float, T_rot:float,T_vib:float,A:float=1,b:float=0, resolution:int=100, wl_pad:float=10, sim_db: pd.DataFrame = None,**kwargs):
    """Model function with function signature compatible for usage with [LMFit](https://lmfit.github.io/lmfit-py/).
    
    Creates and broadens an equidistant stick spectrum from the provided simulation database.
    
    After broadening, resamples the simulation to the same coordinates as the (measured) data.
    
    Returns a spectrum normalized on the interval [b,A].

    Arguments:
        x:              The x-axis of the (measured) data that we want to compare/fit against
        sigma:          Gaussian broadening width of Voigt
        gamma:          Lorentzian broadening width of Voigt
        mu:             The shift in x-coordinates between data and simulation, negative shift is towards longer wavelength
        T_rot:          The rotational temperature
        T_vib:          The vibrational temperature
        A:              The amplitude scaling factor of the spectrum (default: 1)
        b:              The offset w.r.t. 0 of the spectrum (default: 0)
        sim_db:         The pandas.DataFrame containing the database used for the simulation.
        wl_pad:         The amount of nanometer to pad the x-axis of the simulation with to avoid edge effects. Default: 10
        resolution:     The resolution per nanometer of  the equidistant mesh compared to bin/sample simulation by (default: 100)
        mode:           The mode of the spectrum, i.e. 'Emission' versus 'Absorption' (default: Emission)
        wl_mode:        Whether to use 'air' vs 'vacuum' wavelength (default: air)
        
    Returns:
        np.array:       A 1D vector representing the signal intensity calculated from the simulation, which can be used for the minimisation procedure.
    """
    sticks = create_stick_spectrum(T_vib,T_rot,sim_db, mode=kwargs.pop('mode', 'Emission'), wl_mode=kwargs.pop('wl_mode', 'air'))
    refined = equidistant_mesh(sticks, wl_pad = wl_pad, resolution=resolution)
    simulation = apply_voigt(refined,sigma,gamma)
    sim_matched=match_spectra((x-mu).reshape(-1,1),simulation)
    
    # normalize to [0,1] rather than integral=1
    val=sim_matched[:,1]
    sim_matched[:,1] = (val-val.min())/(val.max()-val.min())
    
    return A*sim_matched[:,1]+b

