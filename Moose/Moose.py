"""Provides functionality to simulate and fit spectra, akin to MassiveOES.
However (for now) it focusses on multiprocessing support and not introducing unneccesary package or package version dependencies.
Inspired by MassiveOES and uses the underlying database files, compiled by J Vorac and P. Synek.

https://bitbucket.org/OES_muni/massiveoes/src/master/
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


def query_DB(db_name:str, wl:tuple=(0,np.inf), kind:str='emission',mode:str='air', path:str=pkg.resource_filename('Moose', 'data')) -> pd.DataFrame:
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

    q_mode='{}_wavelength'.format(mode) # vacuum vs air wavelength equivalent

    if wl_min != 0 and wl_max != np.inf:
        q_lines = "(SELECT * FROM lines WHERE {} BETWEEN {} and {})".format(q_mode,wl_min,wl_max)
    else:
        q_lines = "lines"
    
    query = "SELECT air_wavelength,vacuum_wavelength,{},J,E_J,E_v,wavenumber FROM {} INNER JOIN {} ORDER BY {}".format(q_kind, q_lines, q_join, q_mode)
    df = pd.read_sql_query(query,conn)
    conn.close()
    
    return df

def create_stick_spectrum(T_vib,T_rot,df_db:pd.DataFrame=None, mode:'Absorption/Emission'='Emission', wl_mode='air'):
    """Create a stick spectrum based on the data retrieved from a SQL database.
    
    Arguments:
    T_vib:          Vibrational temperature
    T_rot:          Rotational temperature
    df_db:          A pandas DataFrame containing the database data
    mode:           either 'Absorption' or 'Emission' depending on mode we want to use
    wl_mode:        either 'air' or 'vacuum' depending which equivalent we want.
    """

    if type(df_db) == type(None):
        raise TypeError('No Dataframe with database data supplied as kwarg')
    kB=const.physical_constants['Boltzmann constant in inverse meters per kelvin'][0]/100
    pops = (2*df_db['J']+1)*np.exp(-df_db['E_v']/(kB*T_vib)-df_db['E_J']/(kB*T_rot))
    pops/= scipy.integrate.trapezoid(pops,df_db['{}_wavelength'.format(wl_mode)])
    if mode=='Emission':
        y = pops*df_db['A']
    elif mode == 'Absorption':
        y = pops*df_db['B']
    
    return np.array([df_db['{}_wavelength'.format(wl_mode)], y]).T

def equidistant_mesh(sim:np.array, wl_pad=2, resolution=3000):
    '''Creates an equidistant, finer mesh from a simulation
    
    Arguments:
        sim:        The 2D numpy array containing a simulation
        wl_pad:     The padding of the wavelength axis in nm to avoid edge effects
        resolution: The amount of equidistant points per nm for the mesh    
    '''
    x = sim[:,0]
    x_start, x_end = min(x)-wl_pad, max(x)+wl_pad
    points = int(np.abs((x_end-x_start))*resolution)
    
    new_sim = np.zeros((points,2))
    new_sim[:,0] = np.linspace(x_start,x_end,points)

    for i in range(*x.shape):
        fine_index = int((x[i]-x_start)*resolution+0.5)
        new_sim[fine_index,1] += sim[i,1]
    
    return new_sim

def vgt (x,sigma,gamma,mu,a,b):
    """Voigt profile implementation"""
    return a*voigt_profile(x-mu,sigma,gamma)+b

def apply_voigt(sim,sigma, gamma):
    '''Applies voigt broadening to a simulated stick spectrum, normalizing the surface to 1.
    
    Arguments:
        sim:        A (stick) simulation
        sigma:      The Gaussian sigma for the voigt
        gamma:      The Lorentzian gamma (HWHM) for the voigt
    '''
    x = sim[:,0]
    dim = int(len(x))
    if dim % 2 ==0:
        mu = (x[int(dim/2)-1]+x[int(dim/2)])/2
    else:
        mu = x[int(dim/2)]

    v = vgt(x,sigma,gamma,mu,1,0)
    conv = scipy.signal.fftconvolve(sim[:,1], v, mode='same')
    conv /= scipy.integrate.trapezoid(conv,x)
    
    return np.array([x,conv]).T

def match_spectra(meas: np.array, sim: np.array):
    '''Matches a simulation to the same x-axis as the measurement using interpolation.
    Make sure the simulation spans a larger range, fully containing the experimental range.
    Effectively downsamples the simulation to the measurement x data points, interpolating the y values, for residual minimization'''

    interp=scipy.interpolate.interp1d(sim[:,0], sim[:,1])
    matched_y=interp(meas[:,0])
    return np.array([meas[:,0], matched_y]).T

def model_for_fit(x,sigma,gamma, mu, T_rot,T_vib,A=1,b=0,sim_db: pd.DataFrame = None, **kwargs):
    """Model function with function signature compatible for usage with LMFit.
    Creates and broadens an equidistant stick spectrum from the provided simulation database.
    After broadening, resamples the simulation to the same coordinates as the (measured) data.
    Returns a spectrum normalized on the interval [b,A].

    Arguments:
        x:              The x-axis of the (measured) data that we want to compare/fit against
        sigma:          Gaussian broadening width of Voigt
        gamma:          Lorentzian broadening width of Voigt
        mu:             The shift in x-coordinates between data and simulation
        T_rot:          The rotational temperature
        T_vib:          The vibrational temperature
        A:              The amplitude scaling factor of the spectrum (default: 1)
        b:              The offset w.r.t. 0 of the spectrum (default: 0)
        sim_db:         The pandas.DataFrame containing the database used for the simulation.
        wl_pad:         The amount of nanometer to pad the x-axis of the simulation with to avoid edge effects. Default: 2
        resolution:     The resolution in points per nanometer for the resampling of the simulation to an equidistant mesh
        mode:           The mode of the spectrum, i.e. 'Emission' versus 'Absorption' (default: Emission)
        wl_mode:        Whether to use 'air' vs 'vacuum' wavelength (default: air)
    """
    wl_pad = kwargs.pop('wl_pad',2)
    resolution = kwargs.pop('resolution', 3000)
    sticks = create_stick_spectrum(T_vib,T_rot,sim_db, mode=kwargs.pop('mode', 'Emission'), wl_mode=kwargs.pop('wl_mode', 'air'))
    refined = equidistant_mesh(sticks, wl_pad = wl_pad, resolution=resolution)
    simulation = apply_voigt(refined,sigma,gamma)
    sim_matched=match_spectra(np.array([x-mu,x-mu]).T,simulation)
    
    # normalize to [0,1] rather than integral=1
    val=sim_matched[:,1]
    sim_matched[:,1] = (val-val.min())/(val.max()-val.min())
    
    return A*sim_matched[:,1]+b

def set_param(params,param_name,value=0,min=-np.inf, max=np.inf, vary=True):
    '''Function to set/modify a single parameter'''
    params[param_name].value=value
    params[param_name].min=min
    params[param_name].max=max
    params[param_name].vary=vary

def set_params(params,param_dict, print=False):
    '''Function to set/modify a bunch of parameters using a dict'''
    for param in param_dict.keys():
        set_param(params,param,**param_dict[param])
    if print:
        params.pretty_print()