from dask.distributed import Client
from dask import config as dskconf
import atexit
from pathlib import Path
import xarray as xr
import shutil
import logging
import numpy as np
from matplotlib import pyplot as plt
import os
import xesmf

import xclim as xc
import xscen as xs
from xclim.core.calendar import convert_calendar, get_calendar, date_range_like
from xclim.core.units import convert_units_to
from xclim.sdba import properties, measures, construct_moving_yearly_window, unpack_moving_yearly_window

# from xscen.checkups import fig_compare_and_diff, fig_bias_compare_and_diff
# from xscen.catalog import ProjectCatalog, parse_directory, parse_from_ds, DataCatalog
# from xscen.extraction import search_data_catalogs, extract_dataset
# from xscen.io import save_to_zarr, rechunk
# from xscen.config import CONFIG, load_config
# from xscen.common import minimum_calendar, translate_time_chunk, stack_drop_nans, unstack_fill_nan, maybe_unstack
# from xscen.regridding import regrid
# from xscen.biasadjust import train, adjust
# from xscen.scr_utils import measure_time, send_mail, send_mail_on_exit, timeout, TimeoutException
# from xscen.finalize import clean_up

import xscen as xs
from xscen.utils import minimum_calendar, translate_time_chunk, stack_drop_nans, unstack_fill_nan, maybe_unstack
from xscen.io import rechunk
from xscen import (
    ProjectCatalog,
    search_data_catalogs,
    extract_dataset,
    save_to_zarr,
    load_config,
    CONFIG,
    regrid_dataset,
    train, adjust,
    measure_time, send_mail, send_mail_on_exit, timeout, TimeoutException,
    clean_up
)


from utils import calculate_properties, measures_and_heatmap,email_nan_count,move_then_delete, save_move_update, python_scp

server ='neree'


# Load configuration
if server == 'neree':
    load_config('paths_neree.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
else:
    load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
logger = logging.getLogger('xscen')

workdir = Path(CONFIG['paths']['workdir'])
regriddir = Path(CONFIG['paths']['regriddir'])
refdir = Path(CONFIG['paths']['refdir'])

mode = 'o'



if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    atexit.register(send_mail_on_exit, subject=CONFIG['scripting']['subject'])

    # defining variables
    ref_period = slice(*map(str, CONFIG['custom']['ref_period']))
    sim_period = slice(*map(str, CONFIG['custom']['sim_period']))
    ref_source = CONFIG['extraction']['ref_source']

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = ProjectCatalog.create(CONFIG['paths']['project_catalog'], project=CONFIG['project'], overwrite=True)

    # load project catalog
    pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

    # ---MAKEREF---
    for region_name, region_dict in CONFIG['custom']['regions'].items():
        if (
                "makeref" in CONFIG["tasks"]
                and not pcat.exists_in_cat(domain=region_name, processing_level='nancount', source=ref_source)
        ):
            # default
            if not pcat.exists_in_cat(domain=region_name, calendar='default', source=ref_source):
                with (Client(n_workers=3, threads_per_worker=5, memory_limit="15GB", **daskkws)):
                    # search
                    cat_ref = search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = extract_dataset(catalog=dc,
                                             region=region_dict,
                                             **CONFIG['extraction']['reference']['extract_dataset']
                                             )['D']

                    # stack
                    if CONFIG['custom']['stack_drop_nans']:
                        variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                             'variables_and_freqs'].keys())
                        ds_ref = stack_drop_nans(
                            ds_ref,
                            ds_ref[variables[0]].isel(time=130, drop=True).notnull(),
                            #to_file=f'{refdir}/coords_{region_name}.nc'
                        )
                    ds_ref = ds_ref.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})

                    save_move_update(ds=ds_ref,
                                     pcat=pcat,
                                     init_path=f"{workdir}/ref_{region_name}_default.zarr",
                                     final_path=f"{refdir}/ref_{region_name}_default.zarr",
                                     info_dict={'calendar': 'default'
                                                },
                                     server=server)

            # noleap
            if not pcat.exists_in_cat(domain=region_name, calendar='noleap', source=ref_source):
                with (Client(n_workers=3, threads_per_worker=5, memory_limit="15GB", **daskkws)):

                    ds_ref = pcat.search(source=ref_source,calendar='default',domain=region_name).to_dask()

                    # convert calendars
                    ds_refnl = convert_calendar(ds_ref, "noleap")
                    save_move_update(ds=ds_refnl,
                                     pcat=pcat,
                                     init_path=f"{workdir}/ref_{region_name}_noleap.zarr",
                                     final_path=f"{refdir}/ref_{region_name}_noleap.zarr",
                                     info_dict={'calendar': 'noleap'},
                                     server=server)
            # 360_day
            if not pcat.exists_in_cat(domain=region_name, calendar='360_day', source=ref_source):
                with (Client(n_workers=3, threads_per_worker=5, memory_limit="15GB", **daskkws)) :

                    ds_ref = pcat.search(source=ref_source,calendar='default',domain=region_name).to_dask()

                    ds_ref360 = convert_calendar(ds_ref, "360_day", align_on="year")
                    save_move_update(ds=ds_ref360,
                                     pcat=pcat,
                                     init_path=f"{workdir}/ref_{region_name}_360day.zarr",
                                     final_path=f"{refdir}/ref_{region_name}_360day.zarr",
                                     info_dict={'calendar': '360_day'},
                                     server=server)

            # nan_count
            if not pcat.exists_in_cat(domain=region_name, processing_level='nancount', source=ref_source):
                with (Client(n_workers=3, threads_per_worker=5, memory_limit="15GB", **daskkws)):

                    # search
                    cat_ref = search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = extract_dataset(catalog=dc,
                                             region=region_dict,
                                             **CONFIG['extraction']['reference']['extract_dataset']
                                             )['D']

                    print(ds_ref.chunks)
                    # drop to make faster
                    dref_ref = ds_ref.drop_vars('dtr')

                    dref_ref = dref_ref.chunk({'time': -1,'lat':30, 'lon':30})

                    # diagnostics
                    if 'diagnostics' in CONFIG['tasks']:

                        # diagnostics
                        ds_ref_prop, _ = xs.properties_and_measures(
                            ds=dref_ref,
                            **CONFIG['extraction']['reference'][
                                'properties_and_measures']
                        )

                        ds_ref_prop = ds_ref_prop.chunk({'lat':30, 'lon':30})

                        path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=region_name,
                                                                               sim_id=ds_ref.attrs['cat:id'],
                                                                               step='ref'))
                        path_diag_exec = f"{workdir}/{path_diag.name}"


                        save_move_update(ds=ds_ref_prop,
                                         pcat=pcat,
                                         init_path=path_diag_exec,
                                         final_path=path_diag,
                                         server=server
                                         )

                    # nan count
                    ds_ref_props_nan_count = dref_ref.to_array().isnull().sum('time').mean('variable').chunk(
                        {'lon': 10, 'lat': 10})
                    ds_ref_props_nan_count = ds_ref_props_nan_count.to_dataset(name='nan_count')
                    ds_ref_props_nan_count.attrs.update(ds_ref.attrs)


                    save_move_update(ds=ds_ref_props_nan_count,
                                     pcat=pcat,
                                     init_path=f"{workdir}/ref_{region_name}_nancount.zarr",
                                     final_path=f"{refdir}/ref_{region_name}_nancount.zarr",
                                     info_dict={'processing_level': 'nancount'},
                                     server=server
                                     )

                    # plot nan_count and email
                    email_nan_count(path=f"{refdir}/ref_{region_name}_nancount.zarr", region_name=region_name)

    cat_sim = search_data_catalogs(
       **CONFIG['extraction']['simulation']['search_data_catalogs'])
    for sim_id, dc_id in cat_sim.items():
        for region_name, region_dict in CONFIG['custom']['regions'].items():
            #depending on the final tasks, check that the final file doesn't already exists
            final = {'final_zarr': dict(domain=region_name, processing_level='final', id=sim_id),
                     'diagnostics': dict(domain=region_name, processing_level='diag-improved', id=sim_id)}
            final_task = 'diagnostics' if 'diagnostics' in CONFIG[
                "tasks"] else 'final_zarr'
            if not pcat.exists_in_cat(**final[final_task]):
                fmtkws = {'region_name': region_name, 'sim_id': sim_id}
                logger.info('Adding config to log file')
                f1 = open(CONFIG['logging']['handlers']['file']['filename'], 'a+')
                f2 = open('config.yml', 'r')
                f1.write(f2.read())
                f1.close()
                f2.close()

                logger.info(fmtkws)

                # ---EXTRACT---
                if (
                        "extract" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, processing_level='extracted', id=sim_id)
                ):
                    while True:  # if code bugs forever, it will be stopped by the timeout and then tried again
                        try:
                            with (
                                    Client(n_workers=2, threads_per_worker=5, memory_limit="30GB", **daskkws),
                                    measure_time(name='extract', logger=logger),
                                    timeout(3600, task='extract')
                            ):

                                # buffer is need to take a bit larger than actual domain, to avoid weird effect at the edge
                                # domain will be cut to the right shape during the regrid
                                region_dict['buffer']=3
                                ds_sim = extract_dataset(catalog=dc_id,
                                                         region=region_dict,
                                                         **CONFIG['extraction']['simulation']['extract_dataset'],
                                                         )['D']
                                ds_sim['time'] = ds_sim.time.dt.floor('D') # probably this wont be need when data is cleaned

                                # need lat and lon -1 for the regrid
                                ds_sim = ds_sim.chunk(CONFIG['extract']['chunks'])

                                # save to zarr
                                path_cut = f"{workdir}/{sim_id}_extracted.zarr"
                                save_to_zarr(ds=ds_sim,
                                             filename=path_cut,
                                             encoding=CONFIG['custom']['encoding'],
                                             mode=mode
                                             )
                                pcat.update_from_ds(ds=ds_sim, path=path_cut)

                        except TimeoutException:
                            pass
                        else:
                            break
                # ---REGRID---
                if (
                        "regrid" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, processing_level='regridded', id=sim_id)
                ):
                    with (
                            Client(n_workers=2, threads_per_worker=5, memory_limit="25GB", **daskkws),
                            measure_time(name='regrid', logger=logger)
                    ):


                        ds_input = pcat.search(id=sim_id,
                                               processing_level='extracted',
                                               domain=region_name).to_dask()

                        ds_target = pcat.search(**CONFIG['regrid']['target'],
                                                domain=region_name).to_dask()

                        ds_regrid = regrid_dataset(
                            ds=ds_input,
                            ds_grid=ds_target,
                            **CONFIG['regrid']['regrid_dataset']
                        )
                        # chunk time dim
                        ds_regrid = ds_regrid.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_regrid.dims})

                        # save to zarr
                        path_rg = f"{workdir}/{sim_id}_{region_name}_regridded.zarr"
                        save_to_zarr(ds=ds_regrid,
                                     filename=path_rg,
                                     encoding=CONFIG['custom']['encoding'],
                                     mode=mode
                                     )
                        pcat.update_from_ds(ds=ds_regrid, path=path_rg)


                # ---BIAS ADJUST---
                for var, conf in CONFIG['biasadjust_qm']['variables'].items():

                    # ---TRAIN QM ---
                    if (
                            "train_qm" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_training_qm_{var}")
                    ):
                        with (
                                Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                                measure_time(name=f'train_qm {var}', logger=logger)
                        ):
                            # load hist ds (simulation)
                            ds_hist = pcat.search(id=sim_id,
                                                  processing_level='regridded',
                                                  domain=region_name).to_dask()

                            # load ref ds
                            # choose right calendar
                            simcal = get_calendar(ds_hist)
                            refcal = minimum_calendar(simcal, CONFIG['custom']['maximal_calendar'])
                            ds_ref = pcat.search(source = ref_source,
                                                 calendar=refcal,
                                                 domain=region_name).to_dask()


                            # training
                            ds_tr = train(dref=ds_ref,
                                          dhist=ds_hist,
                                          var=[var],
                                          **conf['training_args'])

                            #save and update
                            path_tr = f"{workdir}/{sim_id}_{var}_training_qm.zarr"
                            save_to_zarr(ds=ds_tr,
                                         filename=path_tr,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_tr,
                                                info_dict={'id': f"{sim_id}_training_qm_{var}",
                                                           'domain': region_name,
                                                           'processing_level': "training",
                                                           'xrfreq': ds_hist.attrs['cat:xrfreq']
                                                            },# info_dict needed to reopen correctly in next step
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
                            # load sim ds and training dataset
                            ds_sim = pcat.search(id=sim_id,
                                                 processing_level = 'regridded',
                                                 domain=region_name).to_dask()
                            ds_tr = pcat.search(id=f'{sim_id}_training_qm_{var}', domain=region_name).to_dask()

                            #if more adjusting needed (pr), the level must reflect that
                            plevel = 'half_biasadjusted' if (var in CONFIG['biasadjust_ex']['variables'])\
                                                            and ('train_ex' in CONFIG['tasks']) else 'biasadjusted'


                            #adjust
                            ds_scen_qm = adjust(dsim=ds_sim,
                                             dtrain=ds_tr,
                                             to_level = plevel,
                                             **conf['adjusting_args'])

                            #save and update
                            path_adj = f"{workdir}/{sim_id}_{var}_{plevel}.zarr"
                            save_to_zarr(ds=ds_scen_qm,
                                         filename=path_adj,
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
                            # load hist and ref
                            ds_hist = pcat.search(id=sim_id, domain=region_name,
                                                 processing_level='regridded').to_dask()
                            simcal = get_calendar(ds_hist)
                            refcal = minimum_calendar(simcal, CONFIG['custom']['maximal_calendar'])

                            ds_ref = pcat.search(domain=region_name,
                                                 source=ref_source,
                                                 calendar=refcal
                                                 ).to_dask()


                            # training
                            ds_tr = train(dref=ds_ref,
                                          dhist=ds_hist,
                                          var=[var],
                                          **conf['training_args'])

                            #save and update
                            path_tr = f"{workdir}/{sim_id}_{var}_training_ex.zarr"
                            save_to_zarr(ds=ds_tr,
                                         filename=path_tr,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_tr,
                                                info_dict={'id': f"{sim_id}_training_ex_{var}",
                                                           'domain': region_name,
                                                           'processing_level': "training",
                                                           'xrfreq': ds_hist.attrs['cat:xrfreq']
                                                           },# info_dict needed to reopen correctly in next step
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
                                                 processing_level='half_biasadjusted').to_dask()
                            # load sim and extreme training dataset
                            ds_sim = pcat.search(id=sim_id, domain=region_name,
                                                 processing_level='regridded').to_dask()

                            simcal = get_calendar(ds_sim)
                            refcal = minimum_calendar(simcal, CONFIG['custom']['maximal_calendar'])
                            if simcal != refcal:
                                ds_sim = convert_calendar(ds_sim, refcal)

                            ds_tr = pcat.search(id=f'{sim_id}_training_ex_{var}', domain=region_name).to_dask()


                            # adjustement on moving window
                            ds_sim = ds_sim.sel(time = sim_period)
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

                            #save and update
                            path_adj = f"{workdir}/{sim_id}_{var}_biasadjusted.zarr"
                            ds_scen_ex.lat.encoding.pop('chunks')
                            ds_scen_ex.lon.encoding.pop('chunks')
                            save_to_zarr(ds=ds_scen_ex,
                                         filename=path_adj,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_scen_ex, path=path_adj)

                # ---CLEAN UP ---
                if (
                        "clean_up" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='cleaned_up')
                ):
                    while True:  # if code bugs forever, it will be stopped by the timeout and then tried again
                        try:
                            with (
                                    Client(n_workers=4, threads_per_worker=3, memory_limit="15GB", **daskkws),
                                    measure_time(name=f'cleanup', logger=logger),
                                    timeout(7200, task='clean_up')
                            ):
                                #get all adjusted data
                                cat = search_data_catalogs(**CONFIG['clean_up']['search_data_catalogs'],
                                                           other_search_criteria= { 'id': [sim_id],
                                                                                    'processing_level':["biasadjusted"],
                                                                                    'domain': region_name}
                                                            )
                                dc = cat.popitem()[1]
                                ds = extract_dataset(catalog=dc,
                                                     periods= CONFIG['custom']['sim_period']
                                                          )['D']


                                # can't put in config because of dynamic path
                                maybe_unstack_dict = {'stack_drop_nans': CONFIG['custom']['stack_drop_nans'],
                                                    'rechunk':{d: CONFIG['custom']['chunks'][d]
                                                               for d in ['lon','lat', 'time']},
                                                    #'coords':f"{refdir}/coords_{region_name}.nc"
                                                    }


                                ds = clean_up(ds = ds,
                                             maybe_unstack_dict = maybe_unstack_dict,
                                             **CONFIG['clean_up']['xscen_clean_up'])

                                # TODO: put in clean_up
                                ds['pr'] = ds.pr.round(10)


                                #save and update
                                path_cu = f"{workdir}/{sim_id}_cleaned_up.zarr"
                                save_to_zarr(ds=ds,
                                             filename=path_cu,
                                             mode='o')
                                pcat.update_from_ds(ds=ds, path=path_cu)

                        except TimeoutException:
                            pass
                        else:
                            break

                # ---FINAL ZARR ---
                if (
                        "final_zarr" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='final',
                                                   format='zarr')
                ):
                    with (
                            Client(n_workers=3, threads_per_worker=5, memory_limit="20GB", **daskkws),
                            measure_time(name=f'final zarr rechunk', logger=logger)
                    ):
                        #rechunk and move to final destination
                        fi_path = Path(f"{CONFIG['paths']['output']}".format(**fmtkws))
                        fi_path.parent.mkdir(exist_ok=True, parents=True)

                        rechunk(path_in=f"{workdir}/{sim_id}_cleaned_up.zarr",
                                path_out=fi_path,
                                chunks_over_dim=CONFIG['custom']['out_chunks'],
                                **CONFIG['rechunk'],
                                overwrite=True)

                        # if  delete workdir, but save log and regridded
                        if CONFIG['custom']['delete_in_final_zarr']:

                            if server == 'neree':
                                for name, paths in CONFIG['scp_list'].items():
                                    python_scp(source_path=paths['source'],
                                            destination_path=paths['dest'],
                                            **CONFIG['scp'])
                                    ds = xr.open_zarr(paths['dest'])
                                    pcat.update_from_ds(ds, paths['dest'])
                                move_then_delete(dirs_to_delete=[workdir],
                                                 moving_files=[],
                                                 pcat=pcat)
                            else:
                                final_regrid_path = f"{regriddir}/{sim_id}_{region_name}_regridded.zarr"
                                path_log = CONFIG['logging']['handlers']['file']['filename']
                                move_then_delete(dirs_to_delete=[workdir],
                                                 moving_files=
                                                 [[f"{workdir}/{sim_id}_regridded.zarr", final_regrid_path],
                                                  [path_log, CONFIG['paths']['logging'].format(**fmtkws)]],
                                                 pcat=pcat)

                        # add final file to catalog
                        ds = xr.open_zarr(fi_path)
                        pcat.update_from_ds(ds=ds, path=str(fi_path), info_dict= {'processing_level': 'final'})


                # ---DIAGNOSTICS ---
                if (
                        "diagnostics" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='diag-improved')
                ):
                    with (
                            Client(n_workers=8, threads_per_worker=5, memory_limit="5GB", **daskkws),
                            measure_time(name=f'diagnostics', logger=logger)
                    ):

                        for step, step_dict in CONFIG['diagnostics'].items():
                            ds_input = pcat.search(
                                id=sim_id,
                                domain=region_name,
                                **step_dict['input']
                            ).to_dask().chunk({'time': -1})

                            dref_for_measure = None
                            if 'dref_for_measure' in step_dict:
                                dref_for_measure = pcat.search(
                                    domain=region_name,
                                    **step_dict['dref_for_measure']).to_dask()

                            prop, meas = xs.properties_and_measures(
                                ds=ds_input,
                                dref_for_measure=dref_for_measure,
                                to_level_prop=f'diag-{step}-prop',
                                to_level_meas=f'diag-{step}-meas',
                                rechunk = {'lat':30, 'lon':30, 'time':-1},
                                **step_dict['properties_and_measures']
                            )

                            for ds in [prop, meas]:
                                path_diag = Path(
                                    CONFIG['paths']['diagnostics'].format(
                                        region_name=region_name,
                                        sim_id=sim_id,
                                        level= ds.attrs['cat:processing_level']))
                                path_diag_exec = f"{workdir}/{path_diag.name}"

                                save_to_zarr(ds=ds,
                                             filename=path_diag_exec,
                                             mode='o',
                                             itervar=True,
                                             rechunk={'lat':30, 'lon':30})
                                if server == 'neree':
                                    pcat.update_from_ds(ds=ds,
                                                        path=str(path_diag_exec))
                                else:
                                    shutil.move(path_diag_exec, path_diag)
                                    pcat.update_from_ds(ds=ds,
                                                        path=str(path_diag))

                        meas_datasets = pcat.search(
                            processing_level=['diag-sim-meas',
                                              'diag-scen-meas'],
                            id=sim_id,
                            domain=region_name).to_dataset_dict(xarray_open_kwargs={'decode_timedelta':False})

                        # make sur sim is first (for improved)
                        order_keys = [f'{sim_id}.{region_name}.diag-sim-meas.fx',
                                      f'{sim_id}.{region_name}.diag-scen-meas.fx']
                        meas_datasets = {k: meas_datasets[k] for k in order_keys}

                        hm = xs.diagnostics.measures_heatmap(meas_datasets)

                        ip = xs.diagnostics.measures_improvement(meas_datasets)

                        for ds in [hm, ip]:
                            path_diag = Path(
                                CONFIG['paths']['diagnostics'].format(
                                    region_name=ds.attrs['cat:domain'],
                                    sim_id=ds.attrs['cat:id'],
                                    level=ds.attrs['cat:processing_level']))
                            if server == 'neree':
                                path_diag = f"{workdir}/{path_diag.name}"
                            save_to_zarr(ds=ds, filename=path_diag, mode='o',rechunk={'lat':100, 'lon':100})
                            pcat.update_from_ds(ds=ds, path=path_diag)

                        # # if delete workdir, but keep regridded and log
                        if CONFIG['custom']['delete_in_diag']:

                            logger.info('Move files and delete workdir.')

                            if server == 'neree':
                                for name, paths in CONFIG['scp_list'].items():
                                    source_path = Path(paths['source'].format(**fmtkws))
                                    dest = Path(paths['dest'].format(**fmtkws))
                                    python_scp(source_path= source_path,
                                            destination_path=dest,
                                            **CONFIG['scp'])

                                    dest = dest / source_path.name
                                    if dest.suffix == '.zarr' and source_path.exists():
                                        ds = pcat.search(path=str(source_path)).to_dask()
                                        pcat.update_from_ds(ds, str(dest ))

                                move_then_delete(dirs_to_delete=[workdir],
                                                 moving_files=[],
                                                 pcat=pcat)
                            else:
                                final_regrid_path = f"{regriddir}/{sim_id}_{region_name}_regridded.zarr"
                                path_log = CONFIG['logging']['handlers']['file'][
                                    'filename']
                                move_then_delete(dirs_to_delete= [workdir],
                                                 moving_files =
                                                 [[f"{workdir}/{sim_id}_regridded.zarr",final_regrid_path],
                                                  [path_log, CONFIG['paths']['logging'].format(**fmtkws)]],
                                                  pcat=pcat)

                        send_mail(
                            subject=f"{sim_id}/{region_name} - Succès",
                            msg=f"Toutes les étapes demandées pour la simulation {sim_id}/{region_name} ont été accomplies.",
                        )

    # --- INDIVIDUAL WL ---
    if 'individual_wl' in CONFIG['tasks']:
        dict_input = pcat.search(CONFIG['individual_wl']['input']).to_dataset_dict()
        for name_input, ds_input in dict_input.items():
            for wl in CONFIG['individual_wl']['wl']:
                if not pcat.exists_in_cat(id=ds_input.attrs['cat:id'],
                                          processing_level=f"+{wl}C"):
                    with (
                            Client(n_workers=6, threads_per_worker=5,
                                   memory_limit="4GB", **daskkws),
                            measure_time(name=f'individual_wl', logger=logger)
                    ):
                        # needed for some indicators (ideally would have been calculated in clean_up...)
                        ds_input = ds_input.assign(tas=xc.atmos.tg(ds=ds_input))

                        # cut dataset on the wl window
                        ds_wl = xs.extract.subset_warming_level(ds_input, wl=wl)
                        # calculate indicators & climatological mean and reformat
                        ds_hor_wl = xs.aggregate.produce_horizon(ds_wl)

                        # save and update
                        path_hor_wl = CONFIG['paths']['wl'].format(
                            **xs.utils.get_cat_attrs(ds_hor_wl))
                        save_to_zarr(ds=ds_hor_wl,filename = path_hor_wl,mode = 'o')
                        pcat.update_from_ds(ds=ds_hor_wl, path=path_hor_wl)