"""
Contains convenience functions, to streamline the workflow a bit when using `lmfit` as a fitting library.

Whenever `lmfit` is installed in the currently active environment, these functions are imported into the `Moose` namespace as well.

This behaviour occurs regardless of if `Moose` has been installed with the [fit] optional dependencies or not.

"""

import lmfit
import numpy as np
from .Simulation import default_params, thermal_default_params, model_for_fit, query_DB

def set_param(params: lmfit.Parameters,param_name:str,value:float=0,min:float=-np.inf, max:float=np.inf, vary:bool=True):
    '''Function to set/modify a single parameter in `lmfit.Parameters` object'''
    params[param_name].set(value=value,min=min,max=max, vary=vary)

def set_params(params:lmfit.Parameters,param_dict:dict=default_params, print:bool=False):
    '''Function to set/modify a bunch of parameters using a dict in a `lmfit.Parameters` object.'''
    for param in param_dict.keys():
        set_param(params,param,**param_dict[param])
    if print:
        params.pretty_print()
        

def make_model(species:str, range:tuple[float,float]=(0,np.inf),resolution:int=100, wl_pad:float=10,**kwargs):
    """Convenience function to create a `lmfit.Model` instance already prepared with the line-by-line database for fitting.
    
    Args:
        species (str): The name of the species database to query.
        range (tuple[float,float]): The wavelength range in nanometer to restric the database query to. Defaults to (0,np.inf)
        resolution (int):   The resolution per nanometer of  the equidistant mesh compared to bin/sample simulation by (default: 100)
        wl_pad (float): The amount of nanometer to pad the x-axis of the simulation with to avoid edge effects. Default: 10

    Returns:
        model (lmfit.Model): A `Model` object that can be used for fitting
        params (lmfit.Parameters): A `Parameters` object that can be used for fitting, containing default initial values and ranges.
    """
    pars = kwargs.pop('params', default_params)
    db = query_DB(species, range, **kwargs)
    # param_names = kwargs.pop('param_names', [p for p in _default_params.keys()]) # Makes it possible to change what are considered fit parameters. Be default only `wl_pad` and `resolution` are excluded
    model = lmfit.Model(model_for_fit,sim_db=db)
    params = lmfit.create_params(**pars)
    params['wl_pad'].value = wl_pad
    params['resolution'].value = resolution
    return model, params