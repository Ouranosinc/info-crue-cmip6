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
import cf_xarray as cfxr
import shutil as sh
from dask.distributed import Client, LocalCluster



import xclim as xc
from xclim import sdba
from xclim.core.calendar import convert_calendar, get_calendar
#from xclim.sdba import  construct_moving_yearly_window, unpack_moving_yearly_window


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


#fails on import
#from utils import move_then_delete, save_move_update, python_scp, save_and_update

path = 'configuration/paths_narval.yml'
config = 'configuration/config-MBCn-EMDNA.yml'

# Load configuration
load_config(path, config, verbose=(__name__ == '__main__'), reset=True)
server = CONFIG['server']
logger = logging.getLogger('xscen')

#workdir = Path(CONFIG['paths']['workdir'])
regriddir = Path(CONFIG['paths']['regriddir'])
refdir = Path(CONFIG['paths']['refdir'])




if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    #atexit.register(send_mail_on_exit, subject=CONFIG['scripting']['subject'])

    # defining variables
    ref_period = slice(*map(str, CONFIG['custom']['ref_period']))
    sim_period = slice(*map(str, CONFIG['custom']['sim_period']))
    ref_source = CONFIG['extraction']['ref_source']
    tdd = CONFIG['tdd']

    @contextmanager
    def context(client_kw=None, measure_time_kw=None, timeout_kw=None):
        """ Set up context for each task."""
        # set default
        client_kw = client_kw or {'n_workers': 4, 'threads_per_worker': 3, 'memory_limit': "7GB"}
        measure_time_kw = measure_time_kw or {'name': "undefined task"}
        timeout_kw = timeout_kw or {'seconds': int(1e5), 'task': "undefined task"}

        # call context
        with (Client(**client_kw, **daskkws, local_directory=os.environ['SLURM_TMPDIR']),
              measure_time(**measure_time_kw, logger=logger,),
              timeout(**timeout_kw)):
            yield


    def do_task(task, **kwargs):
        task_in_list = task in CONFIG["tasks"]
        not_already_done = not pcat.exists_in_cat(**kwargs)
        return task_in_list and not_already_done

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = ProjectCatalog.create(CONFIG['paths']['project_catalog'], project=CONFIG['project'])

    # load project catalog
    pcat = xs.ProjectCatalog(CONFIG['paths']['project_catalog'], create=True)

    # ---MAKEREF---
    for region_name, region_dict in CONFIG['custom']['regions'].items():
        if (
                "makeref" in CONFIG["tasks"]
                #and not pcat.exists_in_cat(domain=region_name, processing_level='diag-ref-prop', source=ref_source)
        ):
            # default
            if not pcat.exists_in_cat(domain=region_name,  source=ref_source):
                with context(**CONFIG['extraction']['reference']['context']):

                    # search
                    cat_ref = search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = extract_dataset(catalog=dc,
                                             region=region_dict,
                                             **CONFIG['extraction']['reference']['extract_dataset']
                                             )['D']

                    #standardize units
                    ds_ref = xs.clean_up(ds_ref, **CONFIG['extraction']['reference']['clean_up'])
                    ds_ref['pr'] = xc.core.units.convert_units_to(ds_ref['pr'],
                                                                  'kg m-2 s-1',
                                                                  context='hydro')


                    ds_ref = ds_ref.chunk(
                        {d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})

                    # stack
                    if CONFIG['custom']['stack_drop_nans']:


                        variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                             'variables_and_freqs'].keys())
                        ds_ref = stack_drop_nans(
                            ds_ref,
                            ds_ref[variables[0]].isel(time=130, drop=True).notnull().compute(),
                        )
                    ds_ref = ds_ref.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})
                    ds_ref.attrs['cat:calendar'] = 'default'

                    workdir = Path(f"{CONFIG['paths']['workdir']}/{ds_ref.attrs['cat:id']}_{region_name}/")

                    xs.save_and_update(ds=ds_ref,
                                     pcat=pcat,
                                     path=f"{refdir}/ref_{region_name}_default.zarr",
                                    update_kwargs={'info_dict': {"calendar": "default"}}
                                     )

            # noleap
            if not pcat.exists_in_cat(domain=region_name, calendar='noleap', source=ref_source):
                with context(**CONFIG['extraction']['reference']['context']):

                    ds_ref = pcat.search(source=ref_source,calendar='default',domain=region_name).to_dask()

                    # convert calendars
                    ds_refnl = convert_calendar(ds_ref, "noleap")
                    ds_refnl.attrs['cat:calendar'] = 'noleap'
                    xs.save_and_update(ds=ds_refnl,
                                     pcat=pcat,
                                     path=f"{refdir}/ref_{region_name}_noleap.zarr",
                                     update_kwargs={
                                           'info_dict': {"calendar": "noleap"}}
                                     )
            # 360_day
            if not pcat.exists_in_cat(domain=region_name, calendar='360_day', source=ref_source):
                #with (Client(n_workers=3, threads_per_worker=5, memory_limit="15GB", **daskkws)) :
                with context(**CONFIG['extraction']['reference']['context']):

                    ds_ref = pcat.search(source=ref_source,calendar='default',domain=region_name).to_dask()
                    ds_ref.attrs['cat:calendar'] = '360_day'
                    ds_ref360 = convert_calendar(ds_ref, "360_day", align_on="year")
                    xs.save_and_update(ds=ds_ref360,
                                     pcat=pcat,
                                     path=f"{refdir}/ref_{region_name}_360day.zarr",
                                       update_kwargs={
                                           'info_dict': {"calendar": "360_day"}}
                                     )

            # diag
            if (not pcat.exists_in_cat(domain=region_name, processing_level='diag-ref-prop',
                                       source=ref_source)) and ('diagnostics' in CONFIG['tasks']):
                #with context(**CONFIG['extraction']['reference']['context']):
                with context(client_kw={'n_workers': 2,'threads_per_worker': 5,'memory_limit': "30GB"}): # debug workers dying

                    # search
                    cat_ref = search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = extract_dataset(catalog=dc,
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

                    ds_ref_prop = ds_ref_prop.chunk(**CONFIG['extract']['ref_prop_chunk'])

                    path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=region_name,
                                                                           sim_id=ds_ref_prop.attrs['cat:id'],
                                                                           level=ds_ref_prop.attrs['cat:processing_level']))
                    path_diag_exec = f"{workdir}/{path_diag.name}"


                    xs.save_and_update(ds=ds_ref_prop,
                                     pcat=pcat,
                                     path=path_diag,
                                     )


    cat_sim = search_data_catalogs(
       **CONFIG['extraction']['simulation']['search_data_catalogs'])
    for sim_id, dc_id in cat_sim.items():
    #if False:
        for region_name, region_dict in CONFIG['custom']['regions'].items():
            #depending on the final tasks, check that the final file doesn't already exists
            final = {'final_zarr': dict(domain=region_name, processing_level='final', id=sim_id),
                     'diagnostics': dict(domain=region_name, processing_level='diag-improved', id=sim_id),
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

                workdir = Path(f"{CONFIG['paths']['workdir']}/{sim_id}_{region_name}/")



                # inside the loops, we have a default id and domain
                def do_task_loop(task,id=sim_id, domain=region_name, **kwargs):
                    return do_task(task,id=id, domain=domain, **kwargs)

                # ---EXTRACT---
                if do_task_loop(task="extract", processing_level='extracted'):
                    while True:  # if code bugs forever, it will be stopped by the timeout and then tried again
                        try:
                            with context(**CONFIG['extraction']['simulation']['context']):

                                # buffer is need to take a bit larger than actual domain, to avoid weird effect at the edge
                                # domain will be cut to the right shape during the regrid
                                region_dict['tile_buffer']=3
                                ds_sim = extract_dataset(catalog=dc_id,
                                                         region=region_dict,
                                                         **CONFIG['extraction']['simulation']['extract_dataset'],
                                                         )['D']
                                ds_sim['time'] = ds_sim.time.dt.floor('D') # probably this wont be need when data is cleaned

                                # need lat and lon -1 for the regrid
                                ds_sim = ds_sim.chunk(CONFIG['extract']['sim_chunks'])

                                # save to zarr
                                # path_cut = f"{workdir}/{sim_id}_{region_name}_extracted.zarr"
                                # save_to_zarr(ds=ds_sim,
                                #              filename=path_cut,
                                #              encoding=CONFIG['custom']['encoding'],
                                #              )
                                # pcat.update_from_ds(ds=ds_sim, path=path_cut)

                                xs.save_and_update(ds=ds_sim,
                                                    path=CONFIG['paths']['output'],
                                                    pcat=pcat,
                                                    save_kwargs=dict(encoding=CONFIG['custom']['encoding']))

                        except TimeoutException:
                            pass
                        else:
                            break
                # ---REGRID---
                # note: works well with xesmf 0.7.1. scheduler explodes with 0.8.2.
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
                        ds_regrid = ds_regrid.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_regrid.dims})

                        # save to zarr
                        # path_rg = f"{workdir}/{sim_id}_{region_name}_regridded.zarr"
                        # save_to_zarr(ds=ds_regrid,
                        #              filename=path_rg,
                        #              encoding=CONFIG['custom']['encoding'],
                        #              )
                        # pcat.update_from_ds(ds=ds_regrid, path=path_rg)

                        xs.save_and_update(
                            ds=ds_regrid,
                            path=CONFIG['paths']['output'],
                            pcat=pcat,
                            save_kwargs=dict(encoding=CONFIG['custom']['encoding']))
                        


                # ---BIAS ADJUST---

                from xclim.sdba import adjustment

                # ---MBCN narval---

                if (
                        "npdf-gpies-narval" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id,
                                                   processing_level='biasadjusted')
                ):
                    # load hist ds (simulation)
                    dsim = pcat.search(
                        domain=region_name,
                        variable=CONFIG['biasadjust_mbcn']['variable'],
                        id=sim_id, processing_level='regridded').to_dask(**tdd)
                    # because we took regridded from other domain
                    dsim.attrs['cat:domain'] = region_name

                    # choose right calendar and convert
                    refcal = minimum_calendar(get_calendar(dsim),
                                              CONFIG['custom']['maximal_calendar'])
                    dsim = convert_calendar(dsim, refcal,
                                            align_on=CONFIG['custom']['align_on'])

                    # load ref ds
                    dref = pcat.search(
                        source=ref_source, calendar=refcal,
                        processing_level='extracted',
                        variable=CONFIG['biasadjust_mbcn']['variable'],
                        domain=region_name, ).to_dask(**tdd)

                    # choose right ref period for hist
                    dhist = dsim.sel(time=ref_period)

                    dref, dhist, dsim = (sdba.stack_variables(da) for da in
                                         (dref, dhist, dsim))

                    # stack 30-year periods
                    dsim = xc.core.calendar.stack_periods(
                        dsim, window=30, stride=30).chunk({'time': -1, 'period': -1})


                    # create group
                    group = CONFIG['biasadjust_mbcn'].get('group')
                    if isinstance(group, dict):
                        group = sdba.Grouper.from_kwargs(**group)["group"]
                    elif isinstance(group, str):
                        group = sdba.Grouper(group)

                    # train
                    if not pcat.exists_in_cat(processing_level=f'training_mbcn',
                                              domain=region_name, id=sim_id, ):
                        #with context(**CONFIG['biasadjust_mbcn']['context']['train']): 
                        if True:
                            cluster = LocalCluster(n_workers=3, threads_per_worker=5, memory_limit="130GB",
                           **daskkws)
                            client = Client(cluster)
                            dtrain = sdba.MBCn.train(
                                ref=dref,#.chunk({'loc':5}), #TODO: try
                                hist=dhist,#.chunk({'loc':5}),
                                base_kws=dict(group=group, nquantiles=50, ), 
                                **CONFIG['biasadjust_mbcn']['train']
                            ).ds

                            # attrs
                            dtrain.attrs.update(dhist.attrs)
                            dtrain.attrs['cat:processing_level'] = f"training_mbcn"
                            

                            #dtrain=dtrain.chunk({'iterations':1})
                            print(dtrain)

                            #TODO: chunks iteration ?

                            # save
                            # cant write multi-index directly
                            #encoded = cfxr.encode_multi_index_as_compress(dtrain,
                            #encoded=dtrain                                            # "win_dim")

                            # b/c bug xscen with time coords
                            # path = CONFIG['paths']['mbcn'].format(
                            #     **xs.utils.get_cat_attrs(dtrain, var_as_str=True))
                            # xs.save_to_zarr(ds=dtrain, filename=path)
                            # print('saved')
                            # dtrain = dtrain.drop_vars('time')
                            # pcat.update_from_ds(dtrain, path)
                            
                            
                            #xs.save_and_update(ds=dtrain, path=CONFIG['paths']['mbcn'],
                            #                    pcat=pcat)
                            
                            #s_path=f"{os.environ['SLURM_TMPDIR']}/tmp_train2.zarr" 
                            #f_path = "/project/ctb-frigon/julavoie/info-crue-cmip6/tmp_train2.zarr"
                            f_path = Path(CONFIG['paths']['output'].format(
                                 **xs.utils.get_cat_attrs(dtrain, var_as_str=True)))
                            s_path=f"{os.environ['SLURM_TMPDIR']}/{f_path.name}"
                            f_path=str(f_path)
                            xs.save_to_zarr(ds=dtrain, filename=s_path)
                            print("saved2")
                            sh.make_archive(s_path, "zip", s_path)
                            print('zipped')
                            sh.move(f"{s_path}.zip",f_path.parent)
                            print('moved2')
                            pcat.update_from_ds(dtrain, f_path)
                            print('updated2')



                    # adjust
                    with context(**CONFIG['biasadjust_mbcn']['context']['adjust']):

                        dtrain = pcat.search(processing_level=f"training_mbcn",
                                             domain=region_name, id=sim_id,
                                             ).to_dataset(**tdd)

                        ADJ = sdba.adjustment.TrainAdjust.from_dataset(dtrain)

                        out = ADJ.adjust(
                            sim=dsim, ref=dref, hist=dhist, period_dim="period",
                            base=sdba.QuantileDeltaMapping,
                            **CONFIG['biasadjust_mbcn']['adjust'],
                        )

                        out = sdba.unstack_variables(out)
                        out = xc.core.calendar.unstack_periods(out)

                        # attrs
                        out.attrs.update(dsim.attrs)
                        out.attrs['cat:processing_level'] = f'biasadjusted'

                        # save
                        #xs.save_and_update(ds=out, path=CONFIG['paths']['mbcn'],
                        #                   pcat=pcat)
                        f_path = Path(CONFIG['paths']['mbcn'].format( #TODO: tmp
                                 **xs.utils.get_cat_attrs(out, var_as_str=True)))
                        s_path=f"{os.environ['SLURM_TMPDIR']}/{f_path.name}"
                        f_path=str(f_path)
                        xs.save_to_zarr(ds=out, filename=s_path)
                        print("saved")
                        sh.move(s_path,f_path)
                        print('moved')
                        pcat.update_from_ds(out, f_path)
                        print('updated')
                        
                        path_train=dtrain.attrs['cat:path']
                        sh.make_archive(path_train, "zip", path_train)
                        print('train zipped')


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
                                                     xr_combine_kwargs= {}, # bug xscen
                                                     periods= CONFIG['custom']['sim_period']
                                                          )['D']



                                ds = clean_up(ds = ds.chunk({'time':-1}),
                                              **CONFIG['clean_up']['xscen_clean_up'])

                                #save and update
                                # path_cu = f"{workdir}/{sim_id}_{region_name}_cleaned_up.zarr"
                                # save_to_zarr(ds=ds,
                                #              filename=path_cu,
                                #              mode='o')
                                # pcat.update_from_ds(ds=ds, path=path_cu)
                                xs.save_and_update(
                                    ds=ds,
                                    path=CONFIG['paths']['output'],
                                    pcat=pcat,
                                    )

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


                            final_regrid_path = f"{regriddir}/{sim_id}_{region_name}_regridded.zarr"
                            path_log = CONFIG['logging']['handlers']['file']['filename']
                            xs.move_and_delete(deleting=[workdir],
                                                moving=
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

                                xs.save_and_update(ds=ds, pcat=pcat, path=CONFIG['paths']['output'])
                                
                                # path_diag = Path(
                                #     CONFIG['paths']['diagnostics'].format(
                                #         region_name=region_name,
                                #         sim_id=sim_id,
                                #         level= ds.attrs['cat:processing_level']))
                                # path_diag_exec = f"{workdir}/{path_diag.name}"

                                # save_to_zarr(ds=ds,
                                #              filename=path_diag_exec,
                                #              mode='o',
                                #              itervar=True,
                                #              rechunk=CONFIG['extract']['ref_prop_chunk']
                                # )

                                # sh.move(path_diag_exec, path_diag)
                                # pcat.update_from_ds(ds=ds,
                                #                     path=str(path_diag))

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
                            # path_diag = Path(
                            #     CONFIG['paths']['diagnostics'].format(
                            #         region_name=ds.attrs['cat:domain'],
                            #         sim_id=ds.attrs['cat:id'],
                            #         level=ds.attrs['cat:processing_level']))
                            # if server == 'n':
                            #     path_diag = f"{workdir}/{path_diag.name}"
                            # save_to_zarr(ds=ds, filename=path_diag, mode='o',rechunk={'lat':100, 'lon':100})
                            # pcat.update_from_ds(ds=ds, path=path_diag)
                            xs.save_and_update(ds=ds, pcat=pcat, path=CONFIG['paths']['output'])


                        # # if delete workdir, but keep regridded and log
                        if CONFIG['custom']['delete_in_diag']:

                            logger.info('Move files and delete workdir.')

                           
                            final_regrid_path = f"{regriddir}/{sim_id}_{region_name}_regridded.zarr"
                            path_log = CONFIG['logging']['handlers']['file'][
                                'filename']
                            xs.move_and_delete(
                                deleting= [
                                    workdir
                                ],
                                moving =
                                [[f"{workdir}/{sim_id}_{region_name}_regridded.zarr",final_regrid_path],
                                    [f"{workdir}/{sim_id}_{region_name}_diag-sim-prop.zarr",
                                    f"{CONFIG['paths']['diagdir']}/{region_name}/{sim_id}/{sim_id}_{region_name}_diag-sim-prop.zarr"],
                                    [f"{workdir}/{sim_id}_{region_name}_diag-sim-meas.zarr",
                                    f"{CONFIG['paths']['diagdir']}/{region_name}/{sim_id}/{sim_id}_{region_name}_diag-sim-meas.zarr"],
                                    [f"{workdir}/{sim_id}_{region_name}_diag-scen-prop.zarr",
                                    f"{CONFIG['paths']['diagdir']}/{region_name}/{sim_id}/{sim_id}_{region_name}_diag-scen-prop.zarr"],
                                    [f"{workdir}/{sim_id}_{region_name}_diag-scen-meas.zarr",
                                    f"{CONFIG['paths']['diagdir']}/{region_name}/{sim_id}/{sim_id}_{region_name}_diag-scen-meas.zarr"],
                                    [f"{workdir}/{sim_id}_{region_name}_diag-improved.zarr", #TODO: check
                                    f"{CONFIG['paths']['diagdir']}/{region_name}/{sim_id}/{sim_id}_{region_name}_diag-improved.zarr"],
                                [path_log, CONFIG['paths']['logging'].format(**cur_dict)]
                                ],
                                pcat=pcat)

                        #send_mail(
                        #    subject=f"{sim_id}/{region_name} - Succès",
                        #    msg=f"Toutes les étapes demandées pour la simulation {sim_id}/{region_name} ont été accomplies.",
                        #)

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
                            xs.save_and_update(ds=ds_hor_wl, path=CONFIG['paths']['wl'],
                                               pcat=pcat)


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
                        xs.save_and_update(ds=ds_hor, path=CONFIG['paths']['horizons'],
                                           pcat=pcat)

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
                    xs.save_and_update(ds=ds_delta, path=CONFIG['paths']['deltas'],
                                       pcat=pcat,
                                       save_kwargs=dict(rechunk = {'lat': -1, 'lon':-1}))

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
                    xs.save_and_update(ds=ds_ens, path=CONFIG['paths']['ensembles'],
                                       pcat=pcat,
                                    save_kwargs=dict(rechunk={'lat':-1, 'lon':-1, 'season':1}))

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
                    xs.save_and_update(ds=diff, path=CONFIG['paths']['ensembles'],
                                       pcat=pcat,
                                    save_kwargs=dict(rechunk={'lat':-1, 'lon':-1, 'season':1}))

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
                    xs.save_and_update(ds=pvals, path=CONFIG['paths']['ensembles'],
                                       pcat=pcat,
                                    save_kwargs=dict(rechunk={'lat': -1, 'lon': -1, 'season': 1}))

