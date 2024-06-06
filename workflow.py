from dask.distributed import Client
from dask import config as dskconf
import atexit
from pathlib import Path
import xarray as xr
import shutil
import logging
import dask
import scipy
import os
import numpy as np
from datetime import timedelta
from dask.diagnostics import ProgressBar
import cftime
from contextlib import contextmanager


import xclim as xc
from xclim import sdba
from xclim.core.calendar import convert_calendar, get_calendar
from xclim.sdba import  construct_moving_yearly_window, unpack_moving_yearly_window


if 'ESMFMKFILE' not in os.environ:
    os.environ['ESMFMKFILE'] = str(Path(os.__file__).parent.parent / 'esmf.mk')
import xscen as xs
from xscen.utils import minimum_calendar, stack_drop_nans
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


from utils import (
    move_then_delete,
    save_move_update,
    python_scp,
    save_and_update,
    )

path = 'configuration/paths_l.yml'
config = 'configuration/config-NRCAN2014.yml'

# Load configuration
load_config(path, config, verbose=(__name__ == '__main__'), reset=True)
server = CONFIG['server']
logger = logging.getLogger('xscen')

workdir = Path(CONFIG['paths']['workdir'])
regriddir = Path(CONFIG['paths']['regriddir'])
refdir = Path(CONFIG['paths']['refdir'])


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="intake_esm",
        message="The default of observed=False is deprecated and will be changed to True in a future version of pandas. "
                "Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="intake_esm",
        message="DataFrame.applymap has been deprecated. Use DataFrame.map instead.",
    )

    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    atexit.register(send_mail_on_exit, subject=CONFIG['scripting']['subject'])

    # defining variables
    ref_period = slice(*map(str, CONFIG['custom']['ref_period']))
    sim_period = slice(*map(str, CONFIG['custom']['sim_period']))
    ref_source = CONFIG['extraction']['ref_source']
    tdd = CONFIG['tdd']

    @contextmanager
    def context(client_kw=None, measure_time_kw=None, timeout_kw=None):
        """ Set up context for each task."""
        # set default
        client_kw = client_kw or {'n_workers': 4, 'threads_per_worker': 3,
                                  'memory_limit': "7GB"}
        measure_time_kw = measure_time_kw or {'name': "undefined task"}
        timeout_kw = timeout_kw or {'seconds': int(1e5), 'task': "undefined task"}

        # call context
        with (Client(**client_kw, **daskkws),
              measure_time(**measure_time_kw, logger=logger),
              timeout(**timeout_kw)):
            yield


    def do_task(task, **kwargs):
        task_in_list = task in CONFIG["tasks"]
        not_already_done = not pcat.exists_in_cat(**kwargs)
        return task_in_list and not_already_done

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = ProjectCatalog.create(CONFIG['paths']['project_catalog'],
                                     project=CONFIG['project'])

    # load project catalog
    pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

    # ---MAKEREF---
    for region_name, region_dict in CONFIG['custom']['regions'].items():
        if (
                "makeref" in CONFIG["tasks"]
                and not pcat.exists_in_cat(domain=region_name, source=ref_source,
                                           processing_level='diag-ref-prop',)
        ):
            # default
            if not pcat.exists_in_cat(domain=region_name, calendar='default',
                                      source=ref_source):
                with context(**CONFIG['extraction']['reference']['context']):

                    # search
                    cat_ref = search_data_catalogs(
                        **CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = extract_dataset(
                        catalog=dc,
                        region=region_dict,
                        **CONFIG['extraction']['reference']['extract_dataset']
                                             )['D']

                    #standardize units
                    ds_ref = xs.clean_up(ds_ref,
                                         **CONFIG['extraction']['reference']['clean_up'])
                    ds_ref['pr'] = xc.core.units.convert_units_to(ds_ref['pr'],
                                                                  'kg m-2 s-1',
                                                                  context='hydro')

                    ds_ref = ds_ref.chunk(
                        {d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})

                    # stack
                    if CONFIG['custom']['stack_drop_nans']:

                        variables = list(
                            CONFIG['extraction']['reference']['search_data_catalogs'][
                                             'variables_and_freqs'].keys())
                        ds_ref = stack_drop_nans(
                            ds_ref,
                            ds_ref[variables[0]].isel(time=130, drop=True).notnull(),
                        )
                    ds_ref = ds_ref.chunk(
                        {d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})
                    save_move_update(
                        ds=ds_ref,
                        pcat=pcat,
                        init_path=f"{workdir}/ref_{region_name}_default.zarr",
                        final_path=f"{refdir}/ref_{region_name}_default.zarr",
                        info_dict={'calendar': 'default'},
                        server=server,
                        **CONFIG['scp'])

            # noleap
            if not pcat.exists_in_cat(domain=region_name, calendar='noleap',
                                      source=ref_source):
                with context(**CONFIG['extraction']['reference']['context']):

                    ds_ref = pcat.search(source=ref_source, calendar='default',
                                         domain=region_name).to_dask()

                    # convert calendars
                    ds_refnl = convert_calendar(ds_ref, "noleap")
                    save_move_update(
                        ds=ds_refnl,
                        pcat=pcat,
                        init_path=f"{workdir}/ref_{region_name}_noleap.zarr",
                        final_path=f"{refdir}/ref_{region_name}_noleap.zarr",
                        info_dict={'calendar': 'noleap'},
                        server=server,
                        **CONFIG['scp'])
            # 360_day
            if not pcat.exists_in_cat(domain=region_name, calendar='360_day',
                                      source=ref_source):
                with context(**CONFIG['extraction']['reference']['context']):

                    ds_ref = pcat.search(source=ref_source, calendar='default',
                                         domain=region_name).to_dask()

                    ds_ref360 = convert_calendar(ds_ref, "360_day", align_on="year")
                    save_move_update(
                        ds=ds_ref360,
                        pcat=pcat,
                        init_path=f"{workdir}/ref_{region_name}_360day.zarr",
                        final_path=f"{refdir}/ref_{region_name}_360day.zarr",
                        info_dict={'calendar': '360_day'},
                        server=server,
                        **CONFIG['scp'])

            # diag
            if ((not pcat.exists_in_cat(domain=region_name,
                                        processing_level='diag-ref-prop',
                                        source=ref_source)) and
                    ('diagnostics' in CONFIG['tasks'])):
                with context(**CONFIG['extraction']['reference']['context']):

                    # search
                    cat_ref = search_data_catalogs(
                        **CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = extract_dataset(
                        catalog=dc,
                        region=region_dict,
                        **CONFIG['extraction']['reference']['extract_dataset']
                                             )['D']

                    # drop to make faster
                    dref_ref = ds_ref.drop_vars('dtr')

                    dref_ref = dref_ref.chunk(CONFIG['extract']['ref_chunk'])

                    # diagnostics
                    ds_ref_prop, _ = xs.properties_and_measures(
                        ds=dref_ref,
                        **CONFIG['extraction']['reference'][
                            'properties_and_measures']
                    )

                    ds_ref_prop = ds_ref_prop.chunk(
                        **CONFIG['extract']['ref_prop_chunk'])

                    path_diag = Path(CONFIG['paths']['diagnostics'].format(
                        region_name=region_name,
                        sim_id=ds_ref_prop.attrs['cat:id'],
                        level=ds_ref_prop.attrs['cat:processing_level']))

                    path_diag_exec = f"{workdir}/{path_diag.name}"

                    save_move_update(ds=ds_ref_prop,
                                     pcat=pcat,
                                     init_path=path_diag_exec,
                                     final_path=path_diag,
                                     server=server,
                                     **CONFIG['scp']
                                     )

    cat_sim = search_data_catalogs(
       **CONFIG['extraction']['simulation']['search_data_catalogs'])
    for sim_id, dc_id in cat_sim.items():
        for region_name, region_dict in CONFIG['custom']['regions'].items():
            # depending on the final tasks, check that the final doesn't already exists
            final = {'final_zarr': dict(domain=region_name,
                                        processing_level='final',id=sim_id),
                     'diagnostics': dict(domain=region_name,
                                         processing_level='diag-improved', id=sim_id),
                     }
            final_task = 'diagnostics' if 'diagnostics' in CONFIG[
                "tasks"] else 'final_zarr'
            if not pcat.exists_in_cat(**final[final_task]):
                cur_dict = {'region_name': region_name, 'sim_id': sim_id}
                logger.info('Adding config to log file')
                f1 = open(CONFIG['logging']['handlers']['file']['filename'], 'a+')
                f2 = open(config, 'r')
                f1.write(f2.read())
                f1.close()
                f2.close()

                logger.info(cur_dict)

                # inside the loops, we have a default id and domain
                def do_task_loop(task,id=sim_id, domain=region_name, **kwargs):
                    return do_task(task,id=id, domain=domain, **kwargs)

                # ---EXTRACT---
                if do_task_loop(task="extract", processing_level='extracted'):
                    # if code bugs forever, will be stopped by the timeout and try again
                    while True:
                        try:
                            with context(**CONFIG['extraction']['simulation']['context']):

                                # buffer is need to take a bit larger than actual domain
                                #  to avoid weird effect at the edge
                                # dom will be cut to the right shape during the regrid
                                region_dict['buffer'] = 3
                                ds_sim = extract_dataset(
                                    catalog=dc_id,
                                    region=region_dict,
                                    **CONFIG['extraction']['simulation'][
                                        'extract_dataset'],)['D']
                                ds_sim['time'] = ds_sim.time.dt.floor('D')

                                # need lat and lon -1 for the regrid
                                ds_sim = ds_sim.chunk(CONFIG['extract']['sim_chunks'])

                                # save to zarr
                                path_cut = f"{workdir}/{sim_id}_{region_name}_extracted.zarr"
                                save_to_zarr(ds=ds_sim,
                                             filename=path_cut,
                                             encoding=CONFIG['custom']['encoding'],
                                             )
                                pcat.update_from_ds(ds=ds_sim, path=path_cut)

                        except TimeoutException:
                            pass
                        else:
                            break
                # ---REGRID---
                # note: works well with xesmf 0.7.1. scheduler explodes with 0.8.2.
                # shoule be back with 0.8.5
                if do_task_loop(task='regrid', processing_level='regridded'):
                    with context(**CONFIG['regrid']['context']):


                        ds_input = pcat.search(id=sim_id,
                                               processing_level='extracted',
                                               domain=region_name).to_dask()
                        print(ds_input.chunks)

                        ds_target = pcat.search(**CONFIG['regrid']['target'],
                                                domain=region_name).to_dask()

                        print(ds_target.chunks)

                        ds_regrid = regrid_dataset(
                            ds=ds_input,
                            ds_grid=ds_target,
                            **CONFIG['regrid']['regrid_dataset']
                        )

                        # chunk time dim
                        ds_regrid = ds_regrid.chunk(
                            {d: CONFIG['custom']['chunks'][d] for d in ds_regrid.dims})

                        # save to zarr
                        path_rg = f"{workdir}/{sim_id}_{region_name}_regridded.zarr"
                        save_to_zarr(ds=ds_regrid,
                                     filename=path_rg,
                                     encoding=CONFIG['custom']['encoding'],
                                     )
                        pcat.update_from_ds(ds=ds_regrid, path=path_rg)

                # --- UNIVARIATE ---
                for var, conf in CONFIG['biasadjust_qm']['variables'].items():

                    # ---TRAIN QM ---
                    if do_task_loop(task="train_qm", id=f"{sim_id}_training_qm_{var}"):
                        with context(**CONFIG['biasadjust_qm']['context']['train']):
                            # load hist ds (simulation)
                            ds_hist = pcat.search(id=sim_id,
                                                  processing_level='regridded',
                                                  domain=region_name).to_dask()

                            # load ref ds
                            # choose right calendar
                            simcal = get_calendar(ds_hist)
                            refcal = minimum_calendar(
                                simcal, CONFIG['custom']['maximal_calendar'])
                            ds_ref = pcat.search(source=ref_source,
                                                 calendar=refcal,
                                                 domain=region_name).to_dask()

                            # make sure sim and ref have the same units

                            # training
                            ds_tr = train(dref=ds_ref,
                                          dhist=ds_hist,
                                          var=[var],
                                          **conf['training_args'])

                            # save and update
                            path_tr = f"{workdir}/{sim_id}_{region_name}_{var}_training_qm.zarr"
                            save_to_zarr(ds=ds_tr,
                                         filename=path_tr,
                                         mode='o')
                            pcat.update_from_ds(
                                ds=ds_tr,
                                info_dict={'id': f"{sim_id}_training_qm_{var}",
                                           'domain': region_name,
                                           'processing_level': "training",
                                           'xrfreq': ds_hist.attrs['cat:xrfreq']
                                            }, # needed to reopen correctly in next step
                                path=path_tr)

                    # ---ADJUST QM---
                    if do_task_loop(task='adjust_qm', variable=var,
                                    processing_level=['biasadjusted',
                                                      'half_biasadjusted'],):
                        with context(**CONFIG['biasadjust_qm']['context']['adjust']):
                            # load sim ds and training dataset
                            ds_sim = pcat.search(id=sim_id,
                                                 processing_level='regridded',
                                                 domain=region_name).to_dask()
                            ds_tr = pcat.search(id=f'{sim_id}_training_qm_{var}',
                                                domain=region_name).to_dask()

                            # if more adjusting needed (pr), the level must reflect that
                            plevel = 'half_biasadjusted' if (var in CONFIG[
                                'biasadjust_ex']['variables'])\
                                 and ('train_ex' in CONFIG['tasks']) else 'biasadjusted'

                            # adjust
                            ds_scen_qm = adjust(dsim=ds_sim,
                                                dtrain=ds_tr,
                                                to_level=plevel,
                                                **conf['adjusting_args'])

                            # save and update
                            path_adj = f"{workdir}/{sim_id}_{region_name}_{var}_{plevel}.zarr"
                            save_to_zarr(ds=ds_scen_qm,
                                         filename=path_adj,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_scen_qm, path=path_adj)

                for var, conf in CONFIG['biasadjust_ex']['variables'].items():
                    # ---TRAIN EXTREME---
                    if do_task_loop(task='train_ex',id=f"{sim_id}_training_ex_{var}"):
                        with context(**CONFIG['biasadjust_ex']['context']['train']):
                            # load hist and ref
                            ds_hist = pcat.search(id=sim_id,
                                                  domain=region_name,
                                                  processing_level='regridded'
                                                  ).to_dask()
                            simcal = get_calendar(ds_hist)
                            refcal = minimum_calendar(
                                simcal,
                                CONFIG['custom']['maximal_calendar'])

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
                            path_tr = f"{workdir}/{sim_id}_{region_name}_{var}_training_ex.zarr"
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
                    if do_task_loop(task='adjust_ex', variable=var,processing_level='biasadjusted'):
                        with context(**CONFIG['biasadjust_ex']['context']['adjust']):
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
                            path_adj = f"{workdir}/{sim_id}_{region_name}_{var}_biasadjusted.zarr"
                            ds_scen_ex.lat.encoding.pop('chunks')
                            ds_scen_ex.lon.encoding.pop('chunks')
                            save_to_zarr(ds=ds_scen_ex,
                                         filename=path_adj,
                                         mode='o')
                            pcat.update_from_ds(ds=ds_scen_ex, path=path_adj)

                # ---CLEAN UP ---
                if do_task_loop(task= "clean_up",processing_level='cleaned_up'):
                    while True:  # if code bugs forever, it will be stopped by the timeout and then tried again
                        try:
                            with context(**CONFIG['clean_up']['context']):
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

                                #TODO: for mbcn, try to do it above
                                ds= ds.chunk({'time':-1})

                                ds = clean_up(ds = ds,
                                              **CONFIG['clean_up']['xscen_clean_up'])

                                #save and update
                                path_cu = f"{workdir}/{sim_id}_{region_name}_cleaned_up.zarr"
                                save_to_zarr(ds=ds,
                                             filename=path_cu,
                                             mode='o')
                                pcat.update_from_ds(ds=ds, path=path_cu)

                        except TimeoutException:
                            pass
                        else:
                            break

                # ---FINAL ZARR ---
                if do_task_loop(task='final_zarr', processing_level='final',format='zarr' ):
                    with context(**CONFIG['final_zarr']['context']):
                        #rechunk and move to final destination
                        fi_path = Path(f"{CONFIG['paths']['output']}".format(**cur_dict))
                        fi_path.parent.mkdir(exist_ok=True, parents=True)

                        rechunk(path_in=f"{workdir}/{sim_id}_{region_name}_cleaned_up.zarr",
                                path_out=fi_path,
                                chunks_over_dim=CONFIG['custom']['out_chunks'],
                                **CONFIG['rechunk'],
                                overwrite=True)

                        # add final file to catalog
                        ds = xr.open_zarr(fi_path)
                        pcat.update_from_ds(ds=ds, path=str(fi_path), info_dict= {'processing_level': 'final'})


                        # if  delete workdir, but save log and regridded
                        if CONFIG['custom']['delete_in_final_zarr']:


                            if server == 'n':
                                # rename log with details of current dataset
                                os.rename(f"{workdir}/logger.log",f"{workdir}/logger_{sim_id}_{region_name}.log")

                                for name, paths in CONFIG['scp_list'].items():
                                    source_path = Path(paths['source'].format(**cur_dict))
                                    dest = Path(paths['dest'].format(**cur_dict))
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
                                path_log = CONFIG['logging']['handlers']['file']['filename']
                                move_then_delete(dirs_to_delete=[workdir],
                                                 moving_files=
                                                 [[f"{workdir}/{sim_id}_{region_name}_regridded.zarr", final_regrid_path],
                                                  [path_log, CONFIG['paths']['logging'].format(**cur_dict)]],
                                                 pcat=pcat)

                # --- HEALTH CHECKS ---
                if do_task_loop(task='health_checks', processing_level='health_checks'):
                    with context(**CONFIG['diagnostics']['context'],
                                 measure_time_kw=dict(name=f'health_checks')):
                        ds_input = pcat.search(id=sim_id,processing_level='final',
                                               domain=region_name).to_dataset(**tdd)
                        hc = xs.diagnostics.health_checks(
                            ds=ds_input,
                            **CONFIG['diagnostics']['health_checks'])

                        hc.attrs.update(ds_input.attrs)
                        hc.attrs['cat:processing_level'] = 'health_checks'
                        path = CONFIG['paths']['checks'].format(**cur_dict)
                        xs.save_and_update(ds=hc,path=path, pcat=pcat)


                # ---DIAGNOSTICS ---
                if do_task_loop(task='diagnostics', processing_level='diag-improved'):
                    with context(**CONFIG['diagnostics']['context'],
                                 measure_time_kw=dict(name=f'diagnostics')):

                        for step, step_dict in CONFIG['diagnostics']['steps'].items():

                            # trick because regridde QC-MBCn-RDRS doesn't exist
                            if step =='sim' and region_name=='QC-MBCn-RDRS':
                                diag_reg = 'QC-RDRS'
                            else:
                                diag_reg=region_name

                            ds_input = pcat.search(
                                id=sim_id,
                                domain=diag_reg,
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
                                **step_dict['properties_and_measures']
                            )

                            for ds in [prop, meas]:
                                ds.attrs['cat:domain']=region_name
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
                                             rechunk=CONFIG['extract']['ref_prop_chunk']
                                )
                                if server == 'n':
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
                            domain=region_name).to_dataset_dict(**tdd)

                        # make sur sim is first (for improved)
                        order_keys = [f'{sim_id}.{region_name}.diag-sim-meas.fx',
                                      f'{sim_id}.{region_name}.diag-scen-meas.fx']
                        meas_datasets = {k: meas_datasets[k] for k in order_keys}

                        ip = xs.diagnostics.measures_improvement(meas_datasets)

                        for ds in [ ip]:
                            path_diag = Path(
                                CONFIG['paths']['diagnostics'].format(
                                    region_name=ds.attrs['cat:domain'],
                                    sim_id=ds.attrs['cat:id'],
                                    level=ds.attrs['cat:processing_level']))
                            if server == 'n':
                                path_diag = f"{workdir}/{path_diag.name}"
                            save_to_zarr(ds=ds, filename=path_diag, mode='o',rechunk={'lat':100, 'lon':100})
                            pcat.update_from_ds(ds=ds, path=path_diag)

                        # # if delete workdir, but keep regridded and log
                        if CONFIG['custom']['delete_in_diag']:

                            logger.info('Move files and delete workdir.')

                            if server == 'n':
                                # rename log with details of current dataset
                                os.rename(f"{workdir}/logger.log",f"{workdir}/logger_{sim_id}_{region_name}.log")

                                # scp files
                                for name, paths in CONFIG['scp_list'].items():
                                    source_path = Path(paths['source'].format(**cur_dict))
                                    dest = Path(paths['dest'].format(**cur_dict))
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
                                move_then_delete(
                                    dirs_to_delete= [
                                        workdir
                                    ],
                                                 moving_files =
                                                 [[f"{workdir}/{sim_id}_{region_name}_regridded.zarr",final_regrid_path],
                                                  [path_log, CONFIG['paths']['logging'].format(**cur_dict)]
                                                  ],
                                                  pcat=pcat)

                        send_mail(
                            subject=f"{sim_id}/{region_name} - Succès",
                            msg=f"Toutes les étapes demandées pour la simulation {sim_id}/{region_name} ont été accomplies.",
                        )

    # --- INDIVIDUAL WL ---
    if 'individual_wl' in CONFIG['tasks']:
        dict_input = pcat.search(**CONFIG['individual_wl']['input']).to_dataset_dict()
        for name_input, ds_input in dict_input.items():
            for wl in CONFIG['individual_wl']['wl']:
                if not pcat.exists_in_cat(id=ds_input.attrs['cat:id'],
                                        domain=ds_input.attrs['cat:domain'],
                                        processing_level=f"+{wl}C"):
                    with (
                            Client(n_workers=2, threads_per_worker=5,
                                   memory_limit="20GB", **daskkws),
                            measure_time(name=f'individual_wl', logger=logger)
                    ):


                        # cut dataset on the wl window
                        ds_wl = xs.extract.subset_warming_level(ds_input, wl=wl)

                        if ds_wl:
                            #chunks
                            ds_wl = ds_wl.chunk(CONFIG['individual_wl']['chunks'])

                            # needed for some indicators (ideally would have been calculated in clean_up...)
                            ds_wl = ds_wl.assign(tas=xc.atmos.tg(ds=ds_wl)).load()

                            # calculate indicators & climatological mean and reformat
                            ds_hor_wl = xs.aggregate.produce_horizon(ds_wl, to_level="{wl}")

                            # save and update
                            save_and_update(ds_hor_wl, CONFIG['paths']['wl'], pcat)


    # --- HORIZONS ---
    if 'horizons' in CONFIG['tasks']:
        dict_input = pcat.search(**CONFIG['horizons']['input']).to_dataset_dict()
        for name_input, ds_input in dict_input.items():
            for period in CONFIG['horizons']['periods']:
                if not pcat.exists_in_cat(id=ds_input.attrs['cat:id'],
                                          domain =ds_input.attrs['cat:domain'],
                                          processing_level=f"horizon{period[0]}-{period[1]}"):
                    with (
                            Client(n_workers=2, threads_per_worker=5,
                                   memory_limit="20GB", **daskkws),
                            measure_time(name=f"horizon {period} for {ds_input.attrs['cat:id']}", logger=logger)
                    ):

                        # needed for some indicators (ideally would have been calculated in clean_up...)
                        ds_cut = ds_input.sel(time= slice(*map(str, period)))
                        if ds_cut.attrs['cat:type']=='reconstruction':
                            ds_cut= xs.utils.unstack_fill_nan(ds_cut)
                        ds_cut= ds_cut.chunk(CONFIG['horizons']['chunks'])
                        ds_cut = ds_cut.assign(tas=xc.atmos.tg(ds=ds_cut))#.load()

                        ds_hor = xs.aggregate.produce_horizon(
                            ds_cut,
                            period=period,
                            **CONFIG['horizons']['produce_horizon']
                        )

                        # save and update
                        save_and_update(ds_hor, CONFIG['paths']['horizons'], pcat)

    # --- DELTAS ---
    if 'deltas' in CONFIG['tasks']:
        dict_input = pcat.search(**CONFIG['deltas']['input']).to_dataset_dict(**tdd)
        for name_input, ds_input in dict_input.items():
            id = ds_input.attrs['cat:id']
            plevel = ds_input.attrs['cat:processing_level']
            if not pcat.exists_in_cat(id=id,
                                      domain=ds_input.attrs['cat:domain'],
                                      processing_level=f"delta-{plevel}"):
                with (
                        Client(n_workers=2, threads_per_worker=5, memory_limit="12GB", **daskkws),
                        measure_time(name=f"delta {id} {plevel}", logger=logger)
                ):
                    # get ref dataset
                    ds_ref = pcat.search(id=id,
                                         **CONFIG['deltas']['reference']).to_dask(**tdd)

                    # concat past and future
                    ds_concat = xr.concat([ds_input,ds_ref], dim='horizon',
                                          combine_attrs='override')

                    # compute delta
                    ds_delta = xs.aggregate.compute_deltas(
                        ds=ds_concat,
                        reference_horizon=ds_ref.horizon.values[0],
                        to_level=f'delta-{plevel}'
                    )

                    # save and update
                    save_and_update(ds_delta, CONFIG['paths']['deltas'], pcat,
                                    rechunk = {'lat': -1, 'lon':-1} )

    # --- ENSEMBLES ---
    if 'ensembles' in CONFIG['tasks']:
        # compute ensembles
        for ens_name, ens_inputs in CONFIG['ensembles']['inputs'].items():
            if not pcat.exists_in_cat(processing_level= ens_name):
                with (
                        ProgressBar(),
                        measure_time(name=f'ensemble {ens_name}', logger=logger)
                ):
                    datasets = pcat.search(**ens_inputs).to_dataset_dict(**tdd)

                    weights = xs.ensembles.generate_weights(datasets=datasets)

                    ds_ens = xs.ensemble_stats(datasets=datasets,
                                               weights=weights,
                                               to_level=ens_name)

                    # save and update
                    save_and_update(ds_ens, CONFIG['paths']['ensembles'], pcat,
                                    rechunk={'lat':-1, 'lon':-1, 'season':1})

        # compute difference between ensembles
        for diff_name, diff_inputs in CONFIG['ensembles']['diffs'].items():
            if not pcat.exists_in_cat(processing_level= diff_name):
                with (
                        Client(n_workers=4, threads_per_worker=5,
                               memory_limit="6GB", **daskkws),
                        measure_time(name=f'ensemble {diff_name}', logger=logger)
                ):
                    ens1 = pcat.search(**diff_inputs['first']).to_dask(**tdd)

                    ens2 = pcat.search(**diff_inputs['second']).to_dask(**tdd)


                    diff = (ens1 - ens2)/ens2
                    diff.attrs.update(ens1.attrs)
                    diff.attrs['method'] = '(ens1 - ens2)/ens2'
                    diff.attrs['cat:processing_level']= diff_name

                    # save and update
                    save_and_update(diff, CONFIG['paths']['ensembles'], pcat,
                                    rechunk={'lat':-1, 'lon':-1, 'season':1})

        # compute p-values
        for p_name, p_inputs in CONFIG['ensembles']['pvalues'].items():
            if not pcat.exists_in_cat(processing_level=p_name):
                with (
                        Client(n_workers=4, threads_per_worker=5,
                               memory_limit="6GB", **daskkws),
                        measure_time(name=f'ensemble {p_name}', logger=logger)
                ):
                    ens1 = pcat.search(**p_inputs['first']).to_dataset_dict(**tdd)

                    ens2 = pcat.search(**p_inputs['second']).to_dataset_dict(**tdd)

                    ens1 = xc.ensembles.create_ensemble(ens1)
                    ens2 = xc.ensembles.create_ensemble(ens2)


                    pvals = xr.apply_ufunc(
                        lambda f, r: scipy.stats.ttest_ind(f, r, axis=-1, nan_policy="omit",
                                                           equal_var=False).pvalue,
                        ens1,
                        ens2,
                        input_core_dims=[["realization"], ["realization"]],
                        output_core_dims=[[]],
                        vectorize=True,
                        dask="allowed",
                        output_dtypes=[bool],
                        join='outer',
                        keep_attrs=True
                    )
                    pvals.attrs['method']='pvals>=0.05'
                    pvals.attrs['cat:processing_level'] = p_name

                    # save and update
                    save_and_update(pvals, CONFIG['paths']['ensembles'], pcat,
                                    rechunk={'lat': -1, 'lon': -1, 'season': 1})

