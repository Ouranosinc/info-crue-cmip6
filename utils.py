import xarray as xr
import logging
from pathlib import Path
import shutil

from xclim.sdba.measures import rmse
from xclim import atmos, sdba

from xscen.config import CONFIG, load_config
from xscen.io import save_to_zarr
from xscen.common import  maybe_unstack

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


def calculate_properties(ds, pcat, step, unstack=False, diag_dict=CONFIG['diagnostics']['properties']):
    """
    Calculate diagnostic in the dictionary, save them all in one zarr and updates the catalog.
    The function verifies that a catalog entry doesn't exist already.
    The initial calculations are made in the workdir, but move to a permanent location afterwards.

    If the property is monthly or seasonal, we only keep the first month/season.

    :param ds: Input dataset (with tasmin, tasmax, pr) and the attrs we want to be passed to the final dataset
    :param pcat: Project Catalogue to update
    :param step: Type of input (ref, sim or scen)
    :param diag_dict: Dictionnary of properties to calculate. needs key func, var and args
    :return: A dataset with all properties
    """
    region_name = ds.attrs['cat/domain']
    if not pcat.exists_in_cat(domain=region_name, processing_level=f'diag_{step}', id=ds.attrs['cat/id']):
        for i, (name, prop) in enumerate(diag_dict.items()):
            logger.info(f"Calculating diagnostic {name}")
            prop = eval(prop['func'])(da=ds[prop['var']], **prop['args']).load()

            if unstack:
                prop = maybe_unstack(
                    prop,
                    stack_drop_nans=CONFIG['custom']['stack_drop_nans'],
                    coords=refdir / f'coords_{region_name}.nc',
                    rechunk={d: CONFIG['custom']['out_chunks'][d] for d in ['lat', 'lon']}
                )

            if ("season" or "month") in prop.coords:
                prop = prop[0]

            # put all properties in one dataset
            if i == 0:
                all_prop = prop.to_dataset(name=name)
            else:
                all_prop[name] = prop
        all_prop.attrs.update(ds.attrs)

        path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=region_name,
                                                               sim_id=ds.attrs['cat/id'],
                                                               step=step))
        path_diag_exec = f"{workdir}/{path_diag.name}"
        save_to_zarr(ds=all_prop, filename=path_diag_exec, mode='o', itervar=True)
        shutil.move(path_diag_exec, path_diag, dirs_exist_ok=True)
        pcat.update_from_ds(ds=all_prop,
                            info_dict={'processing_level': f'diag_{step}'},
                            path=str(path_diag))
        return all_prop