import xarray as xr
import logging
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


from xclim.sdba.measures import rmse
from xclim import atmos, sdba
from xclim.core.units import convert_units_to


from xscen.config import CONFIG, load_config
from xscen.io import save_to_zarr
from xscen.common import  maybe_unstack,unstack_fill_nan
from xscen.scr_utils import measure_time, send_mail


load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
workdir = Path(CONFIG['paths']['workdir'])
refdir = Path(CONFIG['paths']['refdir'])
logger = logging.getLogger('xscen')

def compute_properties(sim, ref, period):
    # TODO add more diagnostics, xclim.sdba and from Yannick (R2?)
    nan_count = sim.to_array().isnull().sum('time').mean('variable')
    hist = sim.sel(time=period)
    ref = ref.sel(time=period)

    # Je load deux des variables pour essayer d'éviter les KilledWorker et Timeout
    out = xr.Dataset(data_vars={

        'tx_mean_rmse': rmse(atmos.tx_mean(hist.tasmax, freq='MS').chunk({'time': -1}),
                             atmos.tx_mean(ref.tasmax, freq='MS').chunk({'time': -1})),
        'tn_mean_rmse': rmse(atmos.tn_mean(tasmin=hist.tasmin, freq='MS').chunk({'time': -1}),
                             atmos.tn_mean(tasmin=ref.tasmin, freq='MS').chunk({'time': -1})),
        'prcptot_rmse': rmse(atmos.precip_accumulation(hist.pr, freq='MS').chunk({'time': -1}),
                             atmos.precip_accumulation(ref.pr, freq='MS').chunk({'time': -1})),
        'nan_count': nan_count,
    })

    return out


def calculate_properties(ds, step, diag_dict, unstack=False, unit_conversion={}):
    """
    Calculate properties in the dictionary.
    If the property is monthly or seasonal, we only keep the first month/season.

    :param ds: Input dataset (with tasmin, tasmax, pr) and the attrs we want to be passed to the final dataset
    :param step: Type of input (ref, sim or scen)
    :param diag_dict: Dictionary of properties to calculate. needs key func, var and args
    :param unit_conversion: Dictionary {variable: units to convert to}
    :return: A dataset with all properties
    """

    # we will need to have same units to be able to do measures properly
    # it won't always be able to convert after properties (eg. kg2 m-4 s-2 -> mm2 d-2 doesn't work)
    # so we have to do it now
    for var, unit in unit_conversion.items():
        ds[var] = convert_units_to(ds[var], unit)

    region_name=ds.attrs["cat/domain"]
    for i, (name, prop_dict) in enumerate(diag_dict.items()):
        logger.info(f"Calculating {step} diagnostic {name}")
        prop = eval(prop_dict['func'])(da=ds[prop_dict['var']], **prop_dict['args']).load()

        # TODO: create something more general that keeps all months/seasons
        if "season" in prop.coords:
            prop = prop.isel(season=0)
            prop=prop.drop('season')
        if "month" in prop.coords:
            prop = prop.isel(month=0)
            prop=prop.drop('month')

        if unstack:
            prop = unstack_fill_nan(
                prop,
                coords=refdir / f'coords_{region_name}.nc',
            )
            prop=prop.transpose("lat", "lon")

        prop.attrs['measure']=prop_dict['measure'] if 'measure' in prop_dict else 'bias'


        # put all properties in one dataset
        if i == 0:
            all_prop = prop.to_dataset(name=name)
        else:
            all_prop[name] = prop
    all_prop.attrs.update(ds.attrs)


    return all_prop








def measures_and_heatmap(ref, sims):
    """
    calculate the measures of the difference with the properties of ref and the properties of each sim and create the heat map
    :param ref: reference dataset
    :param sims: list of datasets to compare with ref. Each will be a row on the heatmap.
    """
    hmap = []
    all_measures=[]
    for sim in sims:
        row =[]
        # iterate through all available properties
        for i, var_name in enumerate(sorted(sim.data_vars)):
            # get property
            prop_sim = sim[var_name]
            prop_ref = ref[var_name]

            #choose right measure
            measure_name= prop_sim.attrs['measure'] if 'measure' in prop_sim.attrs else 'bias'
            measure= getattr(sdba.measures,measure_name)

            #calculate bias
            bias_sim_prop = measure(sim=prop_sim, ref=prop_ref).load()

            # put all bias in one dataset
            if i == 0:
                all_bias_sim_prop = bias_sim_prop.to_dataset(name=var_name)
            else:
                all_bias_sim_prop[var_name] = bias_sim_prop


            #mean the absolute value of the bias over all positions and add to heat map
            if measure_name == 'ratio': #if ratio, best is 1, this moves "best to 0 to compare with bias
                row.append(abs(bias_sim_prop -1).mean().values)
            else:
                row.append(abs(bias_sim_prop).mean().values)
        all_bias_sim_prop.attrs.update(sim.attrs)
        all_measures.append(all_bias_sim_prop)
        # append all properties
        hmap.append(row)



    # plot heat map of biases ( 1 column per properties, 1 column for sim , 1 column for scen)
    hmap = np.array(hmap)
    # normalize to 0-1 -> best-worst
    hmap = np.array(
        [(c - min(c)) / (max(c) - min(c)) if max(c) != min(c) else [0.5] * len(c) for c in
         hmap.T]).T

    return all_measures, hmap




def email_nan_count(path, region_name):
    ds_ref_props_nan_count = xr.open_zarr(path, decode_timedelta=False).load()
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.winter.copy()
    cmap.set_under('white')
    ds_ref_props_nan_count.nan_count.plot(ax=ax, vmin=1, vmax=1000, cmap=cmap)
    ax.set_title(
        f'Reference {region_name} - NaN count \nmax {ds_ref_props_nan_count.nan_count.max().item()}')
    plt.close('all')
    send_mail(
        subject=f'Reference for region {region_name} - Success',
        msg=f"Action 'makeref' succeeded for region {region_name}.",
        attachments=[fig]
    )

