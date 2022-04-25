from dask.distributed import Client, performance_report
from dask import config as dskconf
import atexit
from pathlib import Path
import xarray as xr
import shutil
import numpy as np
import json
import logging
from matplotlib import pyplot as plt
import os
from dask.diagnostics import ProgressBar
import pandas as pd
from calendar import isleap

from xclim import atmos,sdba
from xclim.core.calendar import convert_calendar, get_calendar, date_range_like
from xclim.core.units import convert_units_to
from xclim.sdba import properties, measures, construct_moving_yearly_window, unpack_moving_yearly_window
from xclim.core.formatting import update_xclim_history
from xclim.sdba.measures import rmse

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
load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
logger = logging.getLogger('xscen')
workdir = Path(CONFIG['paths']['workdir'])
# TODO: before doing it for real, change the mode, but for testing it is in overwrite
mode = 'o'


def compute_properties(sim, ref, ref_period, fut_period):
    # TODO add more diagnostics, xclim.sdba and from Yannick (R2?)
    if fut_period:
        fut_period = slice(*map(str, fut_period))
    ref_period = slice(*map(str, ref_period))

    hist = sim.sel(time=ref_period)

    # Je load deux des variables pour essayer d'éviter les KilledWorker et Timeout
    pr_threshes = ref.pr.quantile([0.9, 0.99], dim='time', keep_attrs=True).load()
    out = xr.Dataset(data_vars={
        'pr_wet_freq_q99_hist': properties.relative_frequency(hist.pr, thresh=pr_threshes.sel(quantile=0.99, drop=True),
                                                              group='time', op='>='),
        'tx_mean_rmse': rmse(atmos.tx_mean(hist.tasmax, freq='MS').chunk({'time': -1}),
                             atmos.tx_mean(ref.tasmax, freq='MS').chunk({'time': -1})),
        'tn_mean_rmse': rmse(atmos.tn_mean(tasmin=hist.tasmin, freq='MS').chunk({'time': -1}),
                             atmos.tn_mean(tasmin=ref.tasmin, freq='MS').chunk({'time': -1})),
        'prcptot_rmse': rmse(atmos.precip_accumulation(hist.pr, freq='MS').chunk({'time': -1}),
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



if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    dask_perf_file = Path(CONFIG['paths']['reports']) / 'perf_report_template.html'
    dask_perf_file.parent.mkdir(exist_ok=True, parents=True)
    atexit.register(send_mail_on_exit, subject=CONFIG['scr_utils']['subject'])

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = ProjectCatalog.create(CONFIG['paths']['project_catalog'], project=CONFIG['project'], overwrite=True)

    # load project catalog
    pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

    for sim_id in CONFIG['ids']:
        for region_name, region_dict in CONFIG['custom']['regions'].items():
            if not pcat.exists_in_cat(domain=region_name, processing_level='final', id=sim_id):

                fmtkws = {'region_name': region_name,
                          'sim_id': sim_id}
                print(fmtkws)
                # ---REGRID---
                if (
                        "regrid" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, processing_level='regridded', id=f"ref_{sim_id}")
                ):
                    with (
                            Client(n_workers=5, threads_per_worker=3, memory_limit="10GB", **daskkws),
                            performance_report(dask_perf_file.with_name(f'perf_report_regrid_{sim_id}_{region_name}.html')),
                            measure_time(name='regrid', logger=logger)
                    ):

                        # search the data that we need
                        cat_sim = search_data_catalogs(**CONFIG['extraction']['simulations']['search_data_catalogs'],
                                                       other_search_criteria={'id': sim_id}
                                                       )
                        # extract
                        dc = cat_sim.popitem()[1]
                        ds_sim = extract_dataset(catalog=dc,
                                                 region=region_dict,
                                                 **CONFIG['extraction']['simulations']['extract_dataset'],
                                                 )

                        if CONFIG['custom']['stack_drop_nans']:
                            variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                                 'variables_and_timedeltas'].keys())
                            ds_sim = stack_drop_nans(
                                ds_sim,
                                ds_sim[variables[0]].isel(time=130, drop=True).notnull(),
                                to_file=f'{workdir}/coords_{sim_id}_{region_name}.nc'
                            )
                        # change calendar. need similar calendar sim and ref to calculate properties
                        ds_sim= convert_calendar(ds_sim, 'noleap')

                        # chunk time dim
                        ds_sim = ds_sim.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_sim.dims})

                        # save to zarr
                        path_rg = f"{workdir}/{sim_id}_regridded.zarr"
                        save_to_zarr(ds=ds_sim,
                                     filename=path_rg,
                                     auto_rechunk=False,
                                     encoding=CONFIG['custom']['encoding'],
                                     compute=True,
                                     mode=mode
                                     )
                        pcat.update_from_ds(ds=ds_sim, path=path_rg)

                        # TODO: ususally we will have to regrid but for testing CMIP5 ref is already in the right format
                        # get reference that has the domain equivalent to the source of sim
                        cat_ref = search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'],
                                                        other_search_criteria={'domain':ds_sim.attrs['cat/source']})

                        # extract
                        dc = cat_ref.popitem()[1]
                        ds_ref_regrid = extract_dataset(catalog=dc,
                                                 region=region_dict,
                                                 **CONFIG['extraction']['reference']['extract_dataset']
                                                 )
                        # TODO: maybe remove this after CMIP5
                        ds_ref_regrid['tasmax'] = convert_units_to(ds_ref_regrid['tasmax'], 'K')
                        ds_ref_regrid['tasmin'] = convert_units_to(ds_ref_regrid['tasmin'], 'K')

                        #TODO: remove this after CMIP5. remove the whole in time. fake the time axis
                        ds_ref_regrid = convert_calendar(ds_ref_regrid, 'noleap') # avoid the feb 29th in 1996 lenght problem
                        fake_time = pd.date_range('1971-01-01','2000-12-31')
                        fake_time = pd.DatetimeIndex(data=(t for t in fake_time if not (isleap(t.year) and t.month ==2 and t.day ==29)))
                        ds_ref_regrid['time'] = fake_time
                        ds_ref_regrid = convert_calendar(ds_ref_regrid, 'noleap')

                        if CONFIG['custom']['stack_drop_nans']:
                            variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                                 'variables_and_timedeltas'].keys())
                            ds_ref_regrid = stack_drop_nans(
                                ds_ref_regrid,
                                ds_ref_regrid[variables[0]].isel(time=130, drop=True).notnull(),
                                to_file=f'{workdir}/coords_{sim_id}_{region_name}.nc'
                            )


                        ds_ref_regrid = ds_ref_regrid.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_ref_regrid.dims})

                        save_to_zarr(ds_ref_regrid, f"{workdir}/ref_{sim_id}_{region_name}.zarr", auto_rechunk=False,
                                     compute=True, encoding=CONFIG['custom']['encoding'], mode=mode)

                        pcat.update_from_ds(ds=ds_ref_regrid, path = f"{workdir}/ref_{sim_id}_{region_name}.zarr",
                                            info_dict={'id': f"ref_{ds_sim.attrs['cat/id']}",
                                                       'processing_level': 'regridded'
                                                       })


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
                        # properties on ref
                        ds_ref = pcat.search(id=f'ref_{sim_id}', domain=region_name).to_dataset_dict().popitem()[1]
                        ds_ref=ds_ref.chunk({'time': -1})
                        ds_ref_props = compute_properties(ds_ref, ds_ref,CONFIG['custom']['ref_period'], None)
                        #ds_ref_props.attrs.update(ds_ref.attrs)

                        path_ref = Path(CONFIG['paths']['checkups'].format(region_name=region_name, sim_id=sim_id, step='ref'))
                        path_ref.parent.mkdir(exist_ok=True, parents=True)
                        save_to_zarr(ds_ref_props,path_ref , auto_rechunk=False,compute=True, mode='o')

                        ds_ref_props_unstack = unstack_fill_nan(xr.open_zarr(path_ref), coords=workdir / f'coords_{sim_id}_{region_name}.nc').load()
                        fig_ref, ax = plt.subplots(figsize=(10, 10))
                        cmap = plt.cm.winter.copy()
                        cmap.set_under('white')
                        ds_ref_props_unstack.nan_count.plot(ax=ax, vmin=1, vmax=1000, cmap=cmap)
                        ax.set_title(
                            f'Reference {region_name} - NaN count \nmax {ds_ref_props_unstack.nan_count.max().item()} out of {ds_ref.time.size}')
                        plt.close('all')

                        pcat.update_from_ds(ds=ds_ref_props, path=str(path_ref),
                                                info_dict={'id': f"{sim_id}_refprops",
                                                           'domain': region_name,
                                                           'frequency': ds_ref.attrs['cat/frequency'],
                                                           'processing_level': 'properties'
                                                           })



                        # sim properties
                        ds_sim = pcat.search(id=sim_id,domain=region_name,processing_level='extracted').to_dataset_dict().popitem()[1]
                        ds_sim = ds_sim.chunk({'time': -1})
                        ds_sim_props = compute_properties(ds_sim, ds_ref, CONFIG['custom']['ref_period'], CONFIG['custom']['future_period'])
                        #ds_sim_props.attrs.update(ds_sim.attrs)

                        path_sim = Path(CONFIG['paths']['checkups'].format(region_name=region_name, sim_id=sim_id, step='sim'))
                        save_to_zarr(ds=ds_sim_props, filename=path_sim, auto_rechunk=False, mode=mode, itervar=True )

                        logger.info('Sim and Ref properties computed, painting nan count and sending plot.')

                        ds_sim_props_unstack = unstack_fill_nan(ds_sim_props, coords=workdir / f'coords_{sim_id}_{region_name}.nc')
                        nan_count = ds_sim_props_unstack.nan_count.load()

                        fig_sim, ax = plt.subplots(figsize=(12, 8))
                        cmap = plt.cm.winter.copy()
                        cmap.set_under('white')
                        nan_count.plot(ax=ax, vmin=1, vmax=1000, cmap=cmap)
                        ax.set_title(
                            f'Raw simulation {sim_id} {region_name} - NaN count \nmax {nan_count.max().item()} out of {ds_sim.time.size}')
                        send_mail(
                            subject=f'Properties of {sim_id} {region_name} - Success',
                            msg=f"Action 'simproperties' succeeded.",
                            attachments=[fig_ref, fig_sim]
                        )
                        plt.close('all')

                        pcat.update_from_ds(ds=ds_sim_props,
                                            info_dict={'id': f"{sim_id}_simprops",
                                                       'domain': region_name,
                                                       'frequency': ds_sim.attrs['cat/frequency'],
                                                       'processing_level': 'properties'
                                                       },
                                            path=str(path_sim))

                # ---BIAS ADJUST---
                for var, conf in CONFIG['biasadjust_qm']['variables'].items():

                    # ---TRAIN QM---
                    if (
                            "train_qm" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_training_qm_{var}")
                    ):
                        with (
                                Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                                measure_time(name=f'train_qm {var}', logger=logger)
                        ):
                            # load hist ds (simulation)
                            ds_hist = pcat.search(id=sim_id,processing_level='extracted',domain=region_name).to_dataset_dict().popitem()[1]

                            # load ref ds
                            ds_ref = pcat.search(id=f'ref_{sim_id}', domain=region_name).to_dataset_dict().popitem()[1]

                            # training
                            ds_tr = train(dref=ds_ref,
                                          # TODO: time slice only for CMIP5, remove when do CMIP6
                                          dhist=ds_hist.sel(time=slice('1971','2000')),
                                          var=[var],
                                          **conf['training_args']
                                            )

                            path_tr = f"{workdir}/{sim_id}_{var}_training_qm.zarr"

                            save_to_zarr(ds=ds_tr,
                                         filename=path_tr,
                                         auto_rechunk=False,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_tr,
                                                info_dict={'id': f"{sim_id}_training_qm_{var}",
                                                           'domain': region_name,
                                                           'processing_level': "training",
                                                           'frequency': ds_hist.attrs['cat/frequency']
                                                            },
                                                path=path_tr)

                    # ---ADJUST QM---
                    if (
                            "adjust_qm" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=sim_id,
                                                       processing_level= ['biasadjusted','half_biasadjusted'],
                                                       variable=var)
                    ):
                        with (
                                Client(n_workers=6, threads_per_worker=3, memory_limit="10GB", **daskkws),
                                measure_time(name=f'adjust_qm {var}', logger=logger)
                        ):
                            # load sim ds
                            ds_sim = pcat.search(id=sim_id, domain=region_name,
                                                 processing_level='extracted').to_dataset_dict().popitem()[1]
                            ds_tr = pcat.search(id=f'{sim_id}_training_qm_{var}', domain=region_name).to_dataset_dict().popitem()[1]

                            #if more adjusting needed (pr), the level must reflect that
                            plevel = 'half_biasadjusted' if var in CONFIG['biasadjust_ex']['variables'] else 'biasadjusted'
                            ds_scen_qm = adjust(dsim=ds_sim,
                                             dtrain=ds_tr,
                                             to_level = plevel,
                                             **conf['adjusting_args'])
                            path_adj = f"{workdir}/{sim_id}_{var}_{plevel}.zarr"
                            #ds_scen_qm.lat.encoding.pop('chunks')
                            #ds_scen_qm.lon.encoding.pop('chunks')
                            save_to_zarr(ds=ds_scen_qm,
                                         filename=path_adj,
                                         auto_rechunk=False,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_scen_qm, path=path_adj)


                for var, conf in CONFIG['biasadjust_ex']['variables'].items():
                    # ---TRAIN EXTREME---
                    if (
                            "train_ex" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_training_ex_{var}")
                    ):
                        with (
                                Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                                measure_time(name=f'train_ex {var}', logger=logger)
                        ):
                            # load hist ds (simulation)
                            ds_hist = pcat.search(id=sim_id, domain=region_name,
                                                 processing_level='extracted').to_dataset_dict().popitem()[1]

                            # load ref ds

                            ds_ref = pcat.search(id=f'ref_{sim_id}', domain=region_name).to_dataset_dict().popitem()[1]

                            # training
                            ds_tr = train(dref=ds_ref,
                                          # TODO: time slice only for CMIP5, remove when do CMIP6
                                          dhist=ds_hist.sel(time=slice('1971', '2000')),
                                          var=[var],
                                          **conf['training_args'])

                            path_tr = f"{workdir}/{sim_id}_{var}_training_ex.zarr"

                            save_to_zarr(ds=ds_tr,
                                         filename=path_tr,
                                         auto_rechunk=False,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_tr,
                                                info_dict={'id': f"{sim_id}_training_ex_{var}",
                                                           'domain': region_name,
                                                           'processing_level': "training",
                                                           'frequency': ds_hist.attrs['cat/frequency']
                                                           },
                                                path=path_tr)

                    # ---ADJUST EXTREME---
                    if (
                            "adjust_ex" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=sim_id,
                                                       processing_level='biasadjusted',
                                                       variable=var)
                    ):
                        with (
                                Client(n_workers=6, threads_per_worker=3, memory_limit="10GB", **daskkws),
                                measure_time(name=f'adjust_ex {var}', logger=logger)
                        ):
                            # load scen from quantile mapping
                            ds_scen_qm = pcat.search(id=sim_id, domain=region_name,
                                                 processing_level='half_biasadjusted').to_dataset_dict().popitem()[1]
                            # load sim
                            ds_sim = pcat.search(id=sim_id, domain=region_name,
                                                 processing_level='extracted').to_dataset_dict().popitem()[1]

                            ds_tr = pcat.search(id=f'{sim_id}_training_ex_{var}', domain=region_name).to_dataset_dict().popitem()[1]

                            sim_period = slice(*map(str, CONFIG['custom']['sim_period']))

                            # test adjust with moving window
                            ds_sim=convert_calendar(ds_sim, 'noleap').sel(time = sim_period)
                            sim_win = construct_moving_yearly_window(ds_sim,
                                                                     **CONFIG['biasadjust_ex']['moving_yearly_window'])

                            scen_win = construct_moving_yearly_window(ds_scen_qm[var].sel(time = sim_period),
                                                                      **CONFIG['biasadjust_ex']['moving_yearly_window'])

                            ds_scen_ex_win = adjust(dsim=sim_win,
                                                dtrain=ds_tr,
                                                xclim_adjust_args = {'scen': scen_win,
                                                                     'frac': 0.25},
                                                **conf['adjusting_args'])

                            ds_scen_ex = unpack_moving_yearly_window(ds_scen_ex_win)
                            ds_scen_ex = ds_scen_ex.chunk({'time':-1})

                            """
                            ds_scen_ex = adjust(dsim=ds_sim,
                                                dtrain=ds_tr,
                                                xclim_adjust_args = {
                                                    'scen': ds_scen_qm[var].sel(time = sim_period),
                                                    'frac': 0.25
                                                                        },
                                                **conf['adjusting_args'])
                            """
                            path_adj = f"{workdir}/{sim_id}_{var}_biasadjusted.zarr"
                            ds_scen_ex.lat.encoding.pop('chunks')
                            ds_scen_ex.lon.encoding.pop('chunks')
                            save_to_zarr(ds=ds_scen_ex,
                                         filename=path_adj,
                                         auto_rechunk=False,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_scen_ex, path=path_adj)

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
                        cat = search_data_catalogs(**CONFIG['clean_up']['search_data_catalogs'],
                                                    other_search_criteria= {'id': [sim_id],
                                                                            'processing_level':["biasadjusted"],
                                                                            'domain': region_name}
                                                    )
                        dc = cat.popitem()[1]
                        ds = extract_dataset(catalog=dc,
                                             to_level='cleaned_up'
                                                  )
                        ds.attrs['cat/id']=sim_id

                        # convert pr units
                        ds['pr'] = convert_units_to(ds['pr'], 'mm/d')

                        # remove all global attrs that don't come from the catalogue
                        for attr in list(ds.attrs.keys()):
                            if attr[:4] != 'cat/':
                                del ds.attrs[attr]

                        # unstack nans
                        if CONFIG['custom']['stack_drop_nans']:
                            ds = unstack_fill_nan(ds, coords=f"{workdir}/coords_{sim_id}_{region_name}.nc")
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
                        ds_scen = pcat.search(id=sim_id,processing_level='final', domain=region_name).to_dataset_dict().popitem()[1]

                        ds_ref = maybe_unstack(
                            pcat.search(id=f'ref_{sim_id}', domain=region_name).to_dataset_dict().popitem()[1],
                            stack_drop_nans=CONFIG['custom']['stack_drop_nans'],
                            coords=workdir / f'coords_{sim_id}_{region_name}.nc',
                            rechunk={d: CONFIG['custom']['out_chunks'][d] for d in ['lat', 'lon']}
                        )

                        ds_scen_props = compute_properties(ds_scen, ds_ref, CONFIG['custom']['ref_period'], CONFIG['custom']['future_period'])

                        path_scen = CONFIG['paths']['checkups'].format(region_name=region_name, sim_id=sim_id, step='scen')

                        save_to_zarr(ds=ds_scen_props,filename=path_scen,auto_rechunk=False,mode=mode,itervar=True)

                        pcat.update_from_ds(ds=ds_scen_props,
                                            info_dict={'id': f"{sim_id}_scenprops",
                                                       'domain': region_name,
                                                       'frequency': ds_scen.attrs['cat/frequency'],
                                                       'processing_level': 'properties'
                                                       },
                                            path=str(path_scen))

                # --- CHECK UP ---
                if "check_up" in CONFIG["tasks"]:
                    with (
                            Client(n_workers=6, threads_per_worker=3, memory_limit="10GB", **daskkws),
                            performance_report(
                                dask_perf_file.with_name(f'perf_report_checkup_{sim_id}_{region_name}.html')),
                            measure_time(name=f'checkup', logger=logger)
                    ):

                        ref = maybe_unstack(
                            pcat.search(id=f"{sim_id}_refprops", domain=region_name).to_dataset_dict().popitem()[1].load(),
                            coords=workdir / f'coords_{sim_id}_{region_name}.nc',
                            stack_drop_nans=CONFIG['custom']['stack_drop_nans']
                        ).load()
                        sim = maybe_unstack(
                            pcat.search(id=f'{sim_id}_simprops', domain=region_name).to_dataset_dict().popitem()[1],
                            coords=workdir / f'coords_{sim_id}_{region_name}.nc',
                            stack_drop_nans=CONFIG['custom']['stack_drop_nans']
                        ).load()

                        scen = pcat.search(id=f'{sim_id}_scenprops', domain=region_name).to_dataset_dict().popitem()[1].load()

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

                dsR = pcat.search(id=sim_id,
                                  domain=region_name,
                                  processing_level='final').to_dataset_dict().popitem()[1]

                list_dsR.append(dsR)

            dsC = xr.concat(list_dsR, 'lat')

            dsC.attrs['title'] = f"ESPO-R5 v1.0.0 - {sim_id}"
            dsC.attrs['cat/domain'] = f"NAM"
            dsC.attrs.pop('intake_esm_dataset_key')

            dsC_path = CONFIG['paths']['concat_output'].format(sim_id=sim_id)
            dsC.attrs.pop('cat/path')
            save_to_zarr(ds=dsC,
                         filename=dsC_path,
                         auto_rechunk=False,
                         mode='o')
            pcat.update_from_ds(ds=dsC, info_dict={'domain': 'concat_regions'},
                                path=dsC_path)

            print('All concatenations done for today.')
