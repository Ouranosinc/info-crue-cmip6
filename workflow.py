from dask.distributed import Client, performance_report
from dask import compute, config as dskconf
import atexit
from pathlib import Path
import pandas as pd
import xarray as xr
import shutil
import numpy as np
import json
import logging
from matplotlib import pyplot as plt
import dask.array as dsk
import os
from dask.diagnostics import ProgressBar

from xclim import atmos
from xclim.core.calendar import convert_calendar, get_calendar, date_range_like
from xclim.sdba import properties, measures
from xclim.core.formatting import update_xclim_history
import xclim as xc

# add link to xscen
import yaml
import sys
with open("paths_ESPO-R.yml", "r") as stream:
    out = yaml.safe_load(stream)
sys.path.extend([out['paths']['xscen']])

from xscen.checkups import fig_compare_and_diff, fig_bias_compare_and_diff
from xscen.catalog import ProjectCatalog, parse_directory, parse_from_ds, DataCatalog
from xscen.extraction import search_data_catalogs, extract_dataset
from xscen.io import save_to_zarr, rechunk
from xscen.config import CONFIG, load_config
from xscen.common import minimum_calendar, translate_time_chunk, stack_drop_nans, unstack_fill_nan, maybe_unstack
from xscen.regridding import regrid
from xscen.biasadjust import train, adjust
from xscen.scr_utils import measure_time, send_mail, send_mail_on_exit, timeout

# Load configuration
load_config('paths_ESPO-R.yml', 'config_ESPO-R.yml', verbose=(__name__ == '__main__'), reset=True)
logger = logging.getLogger('xscen')
workdir = Path(CONFIG['paths']['workdir'])
refdir = Path(CONFIG['paths']['refdir'])
# TODO: before doing it for real, change the mode, but for testing it is in overwrite
mode = 'o'


def compute_properties(sim, ref, ref_period, fut_period):
    ds_hist = sim.sel(time=ref_period)

    # Je load deux des variables pour essayer d'éviter les KilledWorker et Timeout
    pr_threshes = ref.pr.quantile([0.9, 0.99], dim='time', keep_attrs=True).load()
    out = xr.Dataset(data_vars={
        'pr_wet_freq_q99_hist': properties.relative_frequency(ds_hist.pr, op='>=',
                                                              thresh=pr_threshes.sel(quantile=0.99, drop=True),
                                                              group='time'),
        'tx_mean_rmse': rmse(atmos.tx_mean(ds_hist.tasmax, freq='MS').chunk({'time': -1}),
                             atmos.tx_mean(ref.tasmax, freq='MS').chunk({'time': -1})),
        'tn_mean_rmse': rmse(atmos.tn_mean(tasmin=ds_hist.tasmin, freq='MS').chunk({'time': -1}),
                             atmos.tn_mean(tasmin=ref.tasmin, freq='MS').chunk({'time': -1})),
        'prcptot_rmse': rmse(atmos.precip_accumulation(ds_hist.pr, freq='MS').chunk({'time': -1}),
                             atmos.precip_accumulation(ref.pr, freq='MS').chunk({'time': -1})),
        'nan_count': sim.to_array().isnull().sum('time').mean('variable'),
    })
    out['pr_wet_freq_q99_hist'].attrs[
        'long_name'] = 'Relative frequency of days with precip over the 99th percentile of the reference, in the present.'

    if fut_period is not None:
        out['pr_wet_freq_q99_fut'] = properties.relative_frequency(sim.pr.sel(time=fut_period).chunk({'time': -1}),
                                                                   op='>=',
                                                                   thresh=pr_threshes.sel(quantile=0.99, drop=True),
                                                                   group='time')
        out['pr_wet_freq_q99_fut'].attrs[
            'long_name'] = 'Relative frequency of days with precip over the 99th percentile of the reference, in the future.'

    return out


def fix_cordex_na(ds):
    """AWS-stored CORDEX datasets are all on the same standard calendar, this converts
    the data back to the original calendar, removing added NaNs.
    """
    orig_calendar = ds.attrs.get('original_calendar', 'standard')

    if orig_calendar in ['365_day', '360_day']:
        logger.info(f'Converting calendar to {orig_calendar}')
        ds = convert_calendar(ds, 'noleap')  # drops Feb 29th
        if orig_calendar == '360_day':
            time = date_range_like(ds.time, calendar='360_day')
            ds = ds.where(~ds.time.dt.dayofyear.isin([31, 90, 151, 243, 304]), drop=True)
            if ds.time.size != time.size:
                raise ValueError('I thought I had it woups. Conversion to 360_day failed.')
            ds['time'] = time
    return ds


@measures.check_same_units_and_convert
@update_xclim_history
def rmse(sim: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    """Root mean square error.

    The root mean square error on the time dimension between the simulation and the reference.

    Parameters
    ----------
    sim : xr.DataArray
      Data from the simulation (a time-series for each grid-point)
    ref : xr.DataArray
      Data from the reference (observations) (a time-series for each grid-point)

    Returns
    -------
    xr.DataArray,
      Root mean square error between the simulation and the reference
    """

    def _rmse(sim, ref):
        return np.mean(np.sqrt((sim - ref) ** 2), axis=-1)

    out = xr.apply_ufunc(
        _rmse,
        sim,
        ref,
        input_core_dims=[['time'], ['time']],
        dask='parallelized',
    )
    out.attrs.update(sim.attrs)
    out.attrs["long_name"] = "Root mean square"
    return out


if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    dask_perf_file = Path(CONFIG['paths']['reports']) / 'perf_report_template.html'
    dask_perf_file.parent.mkdir(exist_ok=True, parents=True)

    fut_period = slice(*map(str, CONFIG['custom']['future_period']))
    ref_period = slice(*map(str, CONFIG['custom']['ref_period']))

    atexit.register(send_mail_on_exit, subject=CONFIG['scr_utils']['subject'])

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = ProjectCatalog.create(CONFIG['paths']['project_catalog'], project=CONFIG['project'], overwrite=True)

    # load project catalog
    pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

    # ---MAKEREF---
    for region_name, region_dict in CONFIG['custom']['regions'].items():
        if (
                "makeref" in CONFIG["tasks"]
                and not pcat.exists_in_cat(domain=region_name, processing_level='extracted', source='ERA5-Land')
        ):
            with (
                    Client(n_workers=3, threads_per_worker=5, memory_limit="15GB", **daskkws),
                    performance_report(dask_perf_file.with_name(f'perf_report_makeref_{region_name}.html')),
                    measure_time(name='makeref', logger=logger)
            ):
                # search
                cat_ref = search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                # extract
                dc = cat_ref.popitem()[1]
                ds_ref = extract_dataset(catalog=dc,
                                         region=region_dict,
                                         **CONFIG['extraction']['reference']['extract_dataset']
                                         )
                ds_ref['time'] = ds_ref.time.dt.floor('D')
                dref_ref = ds_ref.chunk({'time': -1})  # time period already cut in extract
                ds_ref_props = compute_properties(dref_ref, dref_ref, ref_period, None).chunk({'lon': -1, 'lat': -1})

                if CONFIG['custom']['stack_drop_nans']:
                    variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                         'variables_and_timedeltas'].keys())
                    ds_ref = stack_drop_nans(
                        ds_ref,
                        ds_ref[variables[0]].isel(time=130, drop=True).notnull(),
                        to_file=f'{refdir}/coords_{region_name}.nc'
                    )
                ds_ref = ds_ref.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})

                # convert calendars
                ds_refnl = convert_calendar(ds_ref, "noleap")
                #ds_refnl.attrs['cat/id'] = f"{ds_refnl.attrs['cat/id']}_noleap"
                ds_ref360 = convert_calendar(ds_ref, "360_day", align_on="year")
                #ds_ref360.attrs['cat/id'] = f"{ds_ref360.attrs['cat/id']}_360day"

                save_to_zarr(ds_ref, f"{refdir}/ref_{region_name}_default.zarr", auto_rechunk=False,
                             compute=True, encoding=CONFIG['custom']['encoding'], mode=mode)
                save_to_zarr(ds_refnl, f"{refdir}/ref_{region_name}_noleap.zarr", auto_rechunk=False,
                             compute=True, encoding=CONFIG['custom']['encoding'], mode=mode),
                save_to_zarr(ds_ref360, f"{refdir}/ref_{region_name}_360day.zarr", auto_rechunk=False,
                             compute=True, encoding=CONFIG['custom']['encoding'], mode=mode),
                save_to_zarr(ds_ref_props, f"{refdir}/ref_{region_name}_properties.zarr", auto_rechunk=False,
                             compute=True, mode='o')

                logger.info('Reference generated, painting nan count and sending plot.')
                dref_props = xr.open_zarr(f"{refdir}/ref_{region_name}_properties.zarr").load()
                dref_props.attrs.update(ds_ref.attrs)

                fig, ax = plt.subplots(figsize=(10, 10))
                cmap = plt.cm.winter.copy()
                cmap.set_under('white')
                dref_props.nan_count.plot(ax=ax, vmin=1, vmax=1000, cmap=cmap)
                ax.set_title(
                    f'Reference {region_name} - NaN count \nmax {dref_props.nan_count.max().item()} out of {dref_ref.time.size}')
                plt.close('all')

                # update cat
                for ds, name in zip([ds_ref, ds_refnl, ds_ref360,dref_props], ['default', 'noleap', '360day', 'properties']):

                    pcat.update_from_ds(ds=ds, path=f"{refdir}/ref_{region_name}_{name}.zarr",
                                        info_dict= {'id': f"{ds_ref.attrs['cat/id']}_{name}",
                                                    #'domain': region_name
                                                    })

                send_mail(
                    subject=f'Reference for region {region_name} - Success',
                    msg=f"Action 'makeref' succeeded for region {region_name}.",
                    attachments=[fig]
                )

    for sim_id in CONFIG['ids']:
        for region_name, region_dict in CONFIG['custom']['regions'].items():
            if not pcat.exists_in_cat(domain=region_name, processing_level='final', id=sim_id):

                fmtkws = {'region_name': region_name,
                          'sim_id': sim_id}
                print(fmtkws)
                # ---REGRID---
                if (
                        "regrid" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, processing_level='regridded', id=sim_id)
                ):
                    with (
                            Client(n_workers=5, threads_per_worker=3, memory_limit="10GB", **daskkws),
                            performance_report(dask_perf_file.with_name(f'perf_report_regrid_{sim_id}_{region_name}.html')),
                            measure_time(name='regrid', logger=logger)
                    ):

                        # search the data that we need
                        cat_sim = search_data_catalogs(**CONFIG['extraction']['simulations']['search_data_catalogs'])

                        # extract
                        dc = cat_sim[sim_id]
                        ds_sim = extract_dataset(catalog=dc,
                                                 **CONFIG['extraction']['simulations']['extract_dataset'],
                                                 )
                        # clean_up_ds
                        name = ds_sim.member_id.item()
                        for attrname, attrval in ds_sim.attrs.items():
                            if isinstance(attrval, str):
                                try:
                                    val = json.loads(attrval)
                                except json.JSONDecodeError:
                                    pass
                                else:
                                    if isinstance(val, dict) and name in val:
                                        ds_sim.attrs[attrname] = val[name]

                        ds_sim['time'] = ds_sim.time.dt.floor('D')

                        variables = list(ds_sim.data_vars)
                        ds_sim = ds_sim.drop_vars(
                            (
                                    set(ds_sim.data_vars.keys()).union(
                                        name for name, crd in ds_sim.coords.items() if name not in crd.dims)
                                    - set(variables) - {'lon', 'lat'}
                            )
                        )

                        ds_sim = fix_cordex_na(ds_sim)

                        # get reference
                        #ds_refnl = xr.open_zarr(f"{refdir}/ref_{region_name}_noleap.zarr",decode_timedelta=False)
                        ds_refnl = pcat.search(id=f'ECMWF_ERA5-Land_NAM_noleap',domain=region_name).to_dataset_dict().popitem()[1]

                        # regrid
                        ds_sim_regrid = regrid(
                            ds=ds_sim,
                            ds_grid=ds_refnl,
                            **CONFIG['regrid']
                        )

                        # chunk time dim
                        ds_sim_regrid = ds_sim_regrid.chunk(translate_time_chunk({'time': '4year'},
                                                                                 get_calendar(ds_sim_regrid),
                                                                                 ds_sim_regrid.time.size
                                                                                 )
                                                            )

                        # save to zarr
                        path_rg = f"{workdir}/{sim_id}_regridded.zarr"
                        save_to_zarr(ds=ds_sim_regrid,
                                     filename=path_rg,
                                     auto_rechunk=False,
                                     encoding=CONFIG['custom']['encoding'],
                                     compute=True,
                                     mode=mode
                                     )
                        pcat.update_from_ds(ds=ds_sim_regrid, path=path_rg)

                #  ---RECHUNK---
                if (
                        "rechunk" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, processing_level='regridded_and_rechunked',id=sim_id)
                ):
                    with (
                            Client(n_workers=2, threads_per_worker=5, memory_limit="18GB", **daskkws),
                            performance_report(
                                dask_perf_file.with_name(f'perf_report_rechunk_{sim_id}_{region_name}.html')),
                            measure_time(name=f'rechunk', logger=logger)
                    ):
                        path_rc = f"{workdir}/{sim_id}_regchunked.zarr"
                        rechunk(path_in=f"{workdir}/{sim_id}_regridded.zarr",
                                path_out=path_rc,
                                chunks_over_dim=CONFIG['custom']['chunks'],
                                **CONFIG['rechunk'],
                                overwrite=True)

                        ds_sim_rechunked = xr.open_zarr(path_rc, decode_timedelta=False)

                        pcat.update_from_ds(ds=ds_sim_rechunked,
                                            path=path_rc,
                                            info_dict={'processing_level': 'regridded_and_rechunked'})

                # --- SIM PROPERTIES ---
                if ("simproperties" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_simprops")
                ):
                    with (
                            Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                            performance_report(
                                dask_perf_file.with_name(f'perf_report_simprops_{sim_id}_{region_name}.html')),
                            measure_time(name=f'simproperties', logger=logger),
                            timeout(3600, task='simproperties')
                    ):
                        #ds_sim = xr.open_zarr(workdir / f'{sim_id}_regchunked.zarr')
                        ds_sim = pcat.search(id=sim_id,
                                            processing_level='regridded_and_rechunked').to_dataset_dict().popitem()[1]

                        simcal = get_calendar(ds_sim)
                        #ds_ref = xr.open_zarr(refdir / f"ref_{region_name}_{simcal}.zarr")
                        ds_ref = pcat.search(id=f'ECMWF_ERA5-Land_NAM_{simcal}',
                                             domain=region_name).to_dataset_dict().popitem()[1]

                        out = compute_properties(ds_sim, ds_ref, ref_period, fut_period)
                        out.attrs.update(ds_sim.attrs)

                        out_path = Path(CONFIG['paths']['checkups'].format(
                            region_name=region_name, sim_id=sim_id, step='sim'
                        ))
                        out_path.parent.mkdir(exist_ok=True, parents=True)
                        save_to_zarr(ds=out,
                                     filename=out_path,
                                     auto_rechunk=False,
                                     mode=mode,
                                     itervar=True
                                     )

                        logger.info('Sim properties computed, painting nan count and sending plot.')
                        dsim_props = unstack_fill_nan(xr.open_zarr(out_path), coords=refdir / f'coords_{region_name}.nc')
                        nan_count = dsim_props.nan_count.load()

                        fig, ax = plt.subplots(figsize=(12, 8))
                        cmap = plt.cm.winter.copy()
                        cmap.set_under('white')
                        nan_count.plot(ax=ax, vmin=1, vmax=1000, cmap=cmap)
                        ax.set_title(
                            f'Raw simulation {sim_id} {region_name} - NaN count \nmax {nan_count.max().item()} out of {ds_sim.time.size}')
                        send_mail(
                            subject=f'Properties of {sim_id} {region_name} - Success',
                            msg=f"Action 'simproperties' succeeded.",
                            attachments=[fig]
                        )
                        plt.close('all')

                        pcat.update_from_ds(ds=out,
                                            info_dict={'id': f"{sim_id}_simprops",
                                                       #'domain': region_name,
                                                       #'processing_level': "properties",
                                                       #'frequency': ds_sim.attrs['cat/frequency']
                                                       },
                                            path=out_path)

                # ---BIAS ADJUST---
                for var, conf in CONFIG['biasadjust']['variables'].items():

                    # ---TRAIN ---
                    if (
                            "train" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_training_{var}")
                    ):
                        with (
                                Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                                measure_time(name=f'train {var}', logger=logger)
                        ):
                            # load hist ds (simulation)
                            #ds_hist = xr.open_zarr(f"{workdir}/{sim_id}_regchunked.zarr")
                            ds_hist = pcat.search(id=sim_id,processing_level='regridded_and_rechunked').to_dataset_dict().popitem()[1]

                            # load ref ds
                            # choose right calendar
                            simcal = get_calendar(ds_hist)
                            refcal = minimum_calendar(simcal,
                                                      CONFIG['custom']['maximal_calendar'])
                            #ds_ref = (xr.open_zarr(f"{refdir}/ref_{region_name}_{refcal}.zarr"))
                            ds_ref = pcat.search(id=f'ECMWF_ERA5-Land_NAM_{refcal}',
                                                 domain=region_name).to_dataset_dict().popitem()[1]


                            # training
                            with measure_time(name=f'train {var}') as mt:
                                ds_tr = train(dref=ds_ref,
                                              dhist=ds_hist,
                                              var=[var],
                                              **conf['training_args'])

                                path_tr = f"{workdir}/{sim_id}_{var}_training.zarr"

                                save_to_zarr(ds=ds_tr,
                                             filename=path_tr,
                                             auto_rechunk=False,
                                             mode='o')
                                pcat.update_from_ds(ds=ds_tr,
                                                    info_dict={'id': f"{sim_id}_training_{var}",
                                                               'domain': region_name,
                                                               'processing_level': "training",
                                                               'frequency': ds_hist.attrs['cat/frequency']
                                                                },
                                                    path=path_tr)

                    # ---ADJUST---
                    if (
                            "adjust" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='biasadjusted',
                                                       variable=var)
                    ):
                        with (
                                Client(n_workers=6, threads_per_worker=3, memory_limit="10GB", **daskkws),
                                measure_time(name=f'adjust {var}', logger=logger)
                        ):
                            # load sim ds
                            #ds_sim = xr.open_zarr(f"{workdir}/{sim_id}_regchunked.zarr")
                            ds_sim = pcat.search(id=sim_id,
                                                 processing_level='regridded_and_rechunked').to_dataset_dict().popitem()[1]
                            #ds_tr = xr.open_zarr(f"{workdir}/{sim_id}_{var}_training.zarr")
                            ds_tr = pcat.search(id=f'{sim_id}_training_{var}').to_dataset_dict().popitem()[1]

                            ds_scen = adjust(dsim=ds_sim,
                                             dtrain=ds_tr,
                                             **conf['adjusting_args'])
                            path_adj = f"{workdir}/{sim_id}_{var}_adjusted.zarr"
                            ds_scen.lat.encoding.pop('chunks')
                            ds_scen.lon.encoding.pop('chunks')
                            save_to_zarr(ds=ds_scen,
                                         filename=path_adj,
                                         auto_rechunk=False,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_scen, path=path_adj)

                # ---CLEAN UP ---
                if (
                        "clean_up" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='cleaned_up')
                ):
                    with (
                            Client(n_workers=4, threads_per_worker=3, memory_limit="15GB", **daskkws),
                            performance_report(
                                dask_perf_file.with_name(f'perf_report_cleanup_{sim_id}_{region_name}.html')),
                            measure_time(name=f'cleanup', logger=logger)
                    ):
                        cat = search_data_catalogs(data_catalogs=[CONFIG['paths']['project_catalog']],
                                                        variables_and_timedeltas= {'tasmax':'1D', 'tasmin':'1D', 'pr':'1D'},
                                                        allow_resampling= False,
                                                        allow_conversion= True,
                                                        other_search_criteria= {'id': [sim_id], 'processing_level':["biasadjusted"]}
                                                    )
                        dc = cat.popitem()[1]
                        ds = extract_dataset(catalog=dc,
                                                  periods=['1950', '2100'],
                                             to_level='cleaned_up'
                                                  )
                        ds.attrs['cat/id']=sim_id
                        # remove all global attrs that don;t come from the catalogue
                        for attr in list(ds.attrs.keys()):
                            if attr[:4] != 'cat/':
                                del ds.attrs[attr]

                        # unstack nans
                        if CONFIG['custom']['stack_drop_nans']:
                            ds = unstack_fill_nan(ds, coords=f"{refdir}/coords_{region_name}.nc")
                            ds = ds.chunk({d: CONFIG['custom']['chunks'][d] for d in ds.dims})

                        # add final attrs
                        for var, attrs in CONFIG['clean_up']['attrs'].items():
                            obj = ds if var == 'global' else ds[var]
                            for attrname, attrtmpl in attrs.items():
                                obj.attrs[attrname] = attrtmpl.format( **fmtkws)

                        # only keep specific var attrs
                        for var in ds.data_vars.values():
                            for attr in list(var.attrs.keys()):
                                if attr not in CONFIG['clean_up']['final_attrs_names']:
                                    del var.attrs[attr]
                        path_cu = f"{workdir}/{sim_id}_cleaned_up.zarr"
                        save_to_zarr(ds=ds,
                                     filename=path_cu,
                                     auto_rechunk=False,
                                     mode='o')
                        pcat.update_from_ds(ds=ds, path=path_cu,
                                            info_dict= {'processing_level': 'cleaned_up'})

                # ---FINAL ZARR ---
                if (
                        "final_zarr" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='final',
                                                   format='zarr')
                ):
                    with (
                            Client(n_workers=3, threads_per_worker=5, memory_limit="20GB", **daskkws),
                            performance_report(
                                dask_perf_file.with_name(f'perf_report_final_zarr_{sim_id}_{region_name}.html')),
                            measure_time(name=f'final zarr rechunk', logger=logger)
                    ):
                        fi_path = Path(f"{CONFIG['paths']['output']}".format(**fmtkws))
                        fi_path.parent.mkdir(exist_ok=True, parents=True)

                        rechunk(path_in=f"{workdir}/{sim_id}_cleaned_up.zarr",
                                path_out=fi_path,
                                chunks_over_dim=CONFIG['custom']['out_chunks'],
                                **CONFIG['rechunk'],
                                overwrite=True)
                        ds = xr.open_zarr(fi_path)
                        pcat.update_from_ds(ds=ds, path=str(fi_path), info_dict= {'processing_level': 'final'})

                # --- SCEN PROPS ---
                if (
                        "scenproperties" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_scenprops")
                ):
                    with (
                            Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                            performance_report(
                                dask_perf_file.with_name(f'perf_report_scenprops_{sim_id}_{region_name}.html')),
                            measure_time(name=f'scenprops', logger=logger),
                            timeout(5400, task='scenproperties')
                    ):
                        #ds_scen = xr.open_zarr(f"{CONFIG['paths']['output']}".format(**fmtkws))
                        ds_scen = pcat.search(id=sim_id,processing_level='final').to_dataset_dict().popitem()[1]


                        scen_cal = get_calendar(ds_scen)
                        ds_ref = maybe_unstack(
                            #xr.open_zarr(refdir / f"ref_{region_name}_{scen_cal}.zarr"),
                            pcat.search(id=f'ECMWF_ERA5-Land_NAM_{scen_cal}',domain=region_name).to_dataset_dict().popitem()[1],
                            stack_drop_nans=CONFIG['custom']['stack_drop_nans'],
                            coords=refdir / f'coords_{region_name}.nc',
                            rechunk={d: CONFIG['custom']['out_chunks'][d] for d in ['lat', 'lon']}
                        )

                        out = compute_properties(ds_scen, ds_ref, ref_period, fut_period)
                        out.attrs.update(ds_scen.attrs)

                        out_path = CONFIG['paths']['checkups'].format(
                            region_name=region_name, sim_id=sim_id, step='scen'
                        )

                        save_to_zarr(ds=out,
                                     filename=out_path,
                                     auto_rechunk=False,
                                     mode=mode,
                                     itervar=True
                                     )

                        pcat.update_from_ds(ds=out,
                                            info_dict={'id': f"{sim_id}_scenprops",
                                                       #'domain': region_name,
                                                       #'processing_level': "properties",
                                                       #'frequency': ds_sim.attrs['cat/frequency']
                                                       },
                                            path=str(out_path))

                # --- CHECK UP ---
                if "check_up" in CONFIG["tasks"]:
                    with (
                            Client(n_workers=6, threads_per_worker=3, memory_limit="10GB", **daskkws),
                            performance_report(
                                dask_perf_file.with_name(f'perf_report_checkup_{sim_id}_{region_name}.html')),
                            measure_time(name=f'checkup', logger=logger)
                    ):

                        ref = xr.open_zarr(refdir / f"ref_{region_name}_properties.zarr").load()
                        #ref = pcat.search(id=f'ECMWF_ERA5-Land_NAM_properties',domain=region_name).to_dataset_dict().popitem()[1].load(),
                        sim = maybe_unstack(
                            xr.open_zarr(Path(CONFIG['paths']['checkups'].format(step='sim', **fmtkws))),
                            #pcat.search(id=f'{sim_id}_simprops', domain=region_name).to_dataset_dict().popitem()[1],
                            coords=refdir / f'coords_{region_name}.nc',
                            stack_drop_nans=CONFIG['custom']['stack_drop_nans']
                        ).load()
                        # TODO: try to make it work with search opening
                        scen = xr.open_zarr(CONFIG['paths']['checkups'].format(step='scen', **fmtkws)).load()
                        #scen = pcat.search(id=f'{sim_id}_scenprops', domain=region_name).to_dataset_dict().popitem()[1]

                        fig_dir = Path(CONFIG['paths']['checkfigs'].format(**fmtkws))
                        fig_dir.mkdir(exist_ok=True, parents=True)
                        paths = []

                        # NaN count
                        fig_compare_and_diff(
                            sim.nan_count.rename('sim'),
                            scen.nan_count.rename('scen'),
                            title='Comparing NaN counts.'
                        ).savefig(fig_dir / 'Nan_count.png')
                        paths.append(fig_dir / 'Nan_count.png')

                        # Extremes - between fut and hist
                        fig_compare_and_diff(
                            scen.pr_wet_freq_q99_hist.rename('historical'),
                            scen.pr_wet_freq_q99_fut.rename('future'),
                            title='Comparing frequency of extremes future vs present.'
                        ).savefig(fig_dir / 'Extremes_pr_scen_hist-fut.png')
                        paths.append(fig_dir / 'Extremes_pr_scen_hist-fut.png')

                        for var in ['pr_wet_freq_q99_hist', 'tx_mean_rmse', 'tn_mean_rmse', 'prcptot_rmse']:
                            fig_bias_compare_and_diff(
                                ref[var], sim[var], scen[var],
                            ).savefig(fig_dir / f'{var}_bias_compare.png')
                            paths.append(fig_dir / f'{var}_bias_compare.png')

                        send_mail(
                            subject=f"{sim_id}/{region_name} - Succès",
                            msg=f"Toutes les étapes demandées pour la simulation {sim_id}/{region_name} ont été accomplies.",
                            attachments=paths
                        )
                        plt.close('all')
                        # when region is done erase workdir
                        shutil.rmtree(workdir)
                        os.mkdir(workdir)

        if (
                "concat" in CONFIG["tasks"]
                and not pcat.exists_in_cat(domain='concat_regions', id=sim_id, processing_level='final', format='zarr')
        ):
            dskconf.set(num_workers=12)
            ProgressBar().register()

            print(f'Contenating {sim_id}.')

            list_dsR = []
            for region_name in CONFIG['custom']['regions']:
                fmtkws = {'region_name': region_name,
                          'sim_id': sim_id}

                #dsR = xr.open_zarr(CONFIG['paths']['output'].format(region_name=region_name, sim_id=sim_id))
                dsR = pcat.search(id=sim_id,
                                  domain=region_name,
                                  processing_level='final').to_dataset_dict().popitem()[1]

                list_dsR.append(dsR)

            dsC = xr.concat(list_dsR, 'lat')
            dsC.attrs['title'] = f"ESPO-R5 v1.0.0 - {sim_id}"
            dsC.attrs['cat/domain'] = f"NAM"
            dsC.attrs.pop('intake_esm_dataset_key')

            dsC_path = CONFIG['paths']['concat_output'].format(sim_id=sim_id)
            dsC.attrs['cat/path'] = ''
            save_to_zarr(ds=dsC,
                         filename=dsC_path,
                         auto_rechunk=False,
                         mode='o')
            pcat.update_from_ds(ds=dsC, info_dict={'domain': 'concat_regions'},
                                path=dsC_path)

            print('All concatenations done for today.')
