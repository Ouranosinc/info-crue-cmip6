from dask.distributed import Client
from dask import config as dskconf
from pathlib import Path
import xarray as xr
import logging
import os
import numpy as np
from datetime import timedelta
from dask.diagnostics import ProgressBar
from contextlib import contextmanager
import cf_xarray as cfxr
import shutil as sh
from dask.distributed import Client, LocalCluster
import sys
from xclim.sdba import adjustment
import xclim as xc
from xclim import sdba
from xclim.core.calendar import convert_calendar, get_calendar
from zipfile import ZipFile

if 'ESMFMKFILE' not in os.environ:
    os.environ['ESMFMKFILE'] = str(Path(os.__file__).parent.parent / 'esmf.mk')
import xscen as xs
from xscen.utils import minimum_calendar, stack_drop_nans
from xscen.io import rechunk
from xscen import CONFIG

# Load configuration
path = 'configuration/paths_narval.yml'
config = 'configuration/config-MBCn-EMDNA.yml'
xs.load_config(path, config, verbose=(__name__ == '__main__'), reset=True)

# useful
logger = logging.getLogger('xscen')
tdd = CONFIG['tdd']

if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    # start dask once for narval
    cluster = LocalCluster(n_workers=3, threads_per_worker=5, memory_limit="130GB", **daskkws)
    #cluster = LocalCluster(n_workers=5, threads_per_worker=5, memory_limit="80GB", **daskkws) # for test
    client = Client(cluster)


    # not very useful anymore on narval
    @contextmanager
    def context(client_kw=None, measure_time_kw=None, timeout_kw=None):
        """ Set up context for each task."""
        # set default
        client_kw = client_kw or {'n_workers': 4, 'threads_per_worker': 3, 'memory_limit': "7GB"}
        measure_time_kw = measure_time_kw or {'name': "undefined task"}
        timeout_kw = timeout_kw or {'seconds': int(1e5), 'task': "undefined task"}

        # call context
        with (
            # narval works better if only open client once
            #Client(**client_kw, **daskkws, local_directory=os.environ['SLURM_TMPDIR']),
              xs.measure_time(**measure_time_kw, logger=logger,),
            #timeout(**timeout_kw) # useless with narval and slurm
            ):
            yield


    def do_task(task, **kwargs):
        task_in_list = task in CONFIG["tasks"]
        not_already_done = not pcat.exists_in_cat(**kwargs)
        return task_in_list and not_already_done
    

    def zip_directory(root, zipfile, **zip_args):
        root = Path(root)

        def _add_to_zip(zf, path, root):
            zf.write(path, path.relative_to(root))
            if path.is_dir():
                for subpath in path.iterdir():
                    _add_to_zip(zf, subpath, root)

        with ZipFile(zipfile, "w", **zip_args) as zf:
            for file in root.iterdir():
                _add_to_zip(zf, file, root)

    def unzip_directory(zipfile, root):

        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        with ZipFile(zipfile, "r") as zf:
            zf.extractall(root)

    # load project catalog
    pcat = xs.ProjectCatalog(
        CONFIG['paths']['project_catalog'],
        create=True,
        project=CONFIG['project']
    )

    # ---MAKEREF---
    ref_source = CONFIG['extraction']['ref_source']
    for region_name, region_dict in CONFIG['custom']['regions'].items():
        if (
                "makeref" in CONFIG["tasks"]
                and not pcat.exists_in_cat(domain=region_name, processing_level='diag-ref-prop', source=ref_source)
        ):
            # default
            if not pcat.exists_in_cat(domain=region_name,  source=ref_source):
                with context(**CONFIG['extraction']['reference']['context']):

                    # search
                    cat_ref = xs.search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = xs.extract_dataset(catalog=dc,
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

                    xs.save_and_update(ds=ds_ref,
                                     pcat=pcat,
                                     path=CONFIG['paths']['reference'],
                                     #path=f"{refdir}/ref_{region_name}_default.zarr",
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
                                     #path=f"{refdir}/ref_{region_name}_noleap.zarr",
                                     path=CONFIG['paths']['reference'],
                                     update_kwargs={
                                           'info_dict': {"calendar": "noleap"}}
                                     )
            # 360_day
            if not pcat.exists_in_cat(domain=region_name, calendar='360_day', source=ref_source):
                with context(**CONFIG['extraction']['reference']['context']):

                    ds_ref = pcat.search(source=ref_source,calendar='default',domain=region_name).to_dask()
                    ds_ref.attrs['cat:calendar'] = '360_day'
                    ds_ref360 = convert_calendar(ds_ref, "360_day", align_on="year")
                    xs.save_and_update(ds=ds_ref360,
                                     pcat=pcat,
                                     #path=f"{refdir}/ref_{region_name}_360day.zarr",
                                     path=CONFIG['paths']['reference'],
                                       update_kwargs={
                                           'info_dict': {"calendar": "360_day"}}
                                     )

            # diag
            if (not pcat.exists_in_cat(domain=region_name, processing_level='diag-ref-prop',
                                       source=ref_source)) and ('diagnostics' in CONFIG['tasks']):
                with context(**CONFIG['extraction']['reference']['context']): 

                    # search
                    cat_ref = xs.search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                    # extract
                    dc = cat_ref.popitem()[1]
                    ds_ref = xs.extract_dataset(catalog=dc,
                                             region=region_dict,
                                             **CONFIG['extraction']['reference']['extract_dataset']
                                             )['D']

                    # drop to make faster
                    dref_ref = ds_ref.drop_vars('dtr')


                    dref_ref = dref_ref.chunk(CONFIG['custom']['ref_chunk'])

                    # diagnostics
                    ds_ref_prop, _ = xs.properties_and_measures(
                        ds=dref_ref,
                        **CONFIG['extraction']['reference'][
                            'properties_and_measures']
                    )

                    ds_ref_prop = ds_ref_prop.chunk(**CONFIG['custom']['ref_prop_chunk'])

                    path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=region_name,
                                                                           sim_id=ds_ref_prop.attrs['cat:id'],
                                                                           level=ds_ref_prop.attrs['cat:processing_level']))

                    xs.save_and_update(ds=ds_ref_prop,
                                     pcat=pcat,
                                     path=path_diag,
                                     )


    # use arg to define source, trick for narval, pre-snakemake
    sdc=CONFIG['extraction']['simulation']['search_data_catalogs'].copy()
    sdc['other_search_criteria']['source']=sys.argv[1]
    cat_sim = xs.search_data_catalogs(**sdc)

    for sim_id, dc_id in cat_sim.items():
        for region_name, region_dict in CONFIG['custom']['regions'].items():
            #depending on the final tasks, check that the final file doesn't already exists
            final = {'final_zarr': dict(domain=region_name, processing_level='final', id=sim_id),
                     'diagnostics': dict(domain=region_name, processing_level='diag-improved', id=sim_id),
                     }
            final_task = 'diagnostics' if 'diagnostics' in CONFIG[ "tasks"] else 'final_zarr'
            
            if not pcat.exists_in_cat(**final[final_task]):
                cur_dict = {'domain': region_name, 'id': sim_id}

                logger.info(cur_dict)
                workdir = Path(f"{CONFIG['paths']['workdir']}/{sim_id}_{region_name}/")

                # inside the loops, we have a default id and domain
                def do_task_loop(task,id=sim_id, domain=region_name, **kwargs):
                    return do_task(task,id=id, domain=domain, **kwargs)

                # ---EXTRACT---
                if do_task_loop(task="extract", processing_level='extracted'):

                    with context(**CONFIG['extraction']['simulation']['context']):

                        # buffer is need to take a bit larger than actual domain, to avoid weird effect at the edge
                        # domain will be cut to the right shape during the regrid
                        region_dict['tile_buffer']=3
                        ds_sim = xs.extract_dataset(catalog=dc_id,
                                                    region=region_dict,
                                                    **CONFIG['extraction']['simulation']['extract_dataset'],
                                                    )['D']
                        ds_sim['time'] = ds_sim.time.dt.floor('D') # probably this wont be need when data is cleaned

                        # need lat and lon -1 for the regrid
                        ds_sim = ds_sim.chunk(CONFIG['custom']['sim_chunks'])

                        xs.save_and_update(ds=ds_sim,
                                            path=CONFIG['paths']['output'],
                                            pcat=pcat,
                                            save_kwargs=dict(encoding=CONFIG['custom']['encoding']))


                # ---REGRID---
                if do_task_loop(task='regrid', processing_level='regridded'):
                    with context(**CONFIG['regrid']['context']):

                        ds_input = pcat.search(id=sim_id,
                                               processing_level='extracted',
                                               domain=region_name).to_dask()

                        ds_target = pcat.search(**CONFIG['regrid']['target'],
                                                domain=region_name).to_dask()

                        ds_regrid = xs.regrid_dataset(
                            ds=ds_input,
                            ds_grid=ds_target,
                            weights_location= f"{os.environ['SLURM_TMPDIR']}/weights/",
                            **CONFIG['regrid']['regrid_dataset']
                        )

                        # chunk time dim
                        ds_regrid = ds_regrid.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_regrid.dims})

                        xs.save_and_update(
                            ds=ds_regrid,
                            path=CONFIG['paths']['output'],
                            pcat=pcat,
                            save_kwargs=dict(encoding=CONFIG['custom']['encoding']))
                        


                # ---BIAS ADJUST---
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
                    dhist = dsim.sel(time=slice(*map(str, CONFIG['custom']['ref_period'])))

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
                        with context(**CONFIG['biasadjust_mbcn']['context']['train']): 
                            dtrain = sdba.MBCn.train(
                                ref=dref,
                                hist=dhist,
                                base_kws=dict(group=group, nquantiles=50, ), 
                                **CONFIG['biasadjust_mbcn']['train']
                            ).ds

                            # attrs
                            dtrain.attrs.update(dhist.attrs)
                            dtrain.attrs['cat:processing_level'] = f"training_mbcn"

                            # save, zip, move, update
                            f_path = Path(CONFIG['paths']['output_zip'].format(
                                 **xs.utils.get_cat_attrs(dtrain, var_as_str=True)))
                            s_path=f"{os.environ['SLURM_TMPDIR']}/{f_path.name[:-4]}"
                            xs.save_to_zarr(ds=dtrain, filename=s_path)
                            zip_directory(s_path,f_path)
                            pcat.update_from_ds(dtrain, f_path, info_dict={'format':'zarr'})



                    # adjust
                    with context(**CONFIG['biasadjust_mbcn']['context']['adjust']):

                        # dtrain = pcat.search(processing_level=f"training_mbcn",
                        #                      domain=region_name, id=sim_id,
                        #                      ).to_dataset(**tdd)

                        f_path = Path(CONFIG['paths']['output_zip'].format(**cur_dict, processing_level=f"training_mbcn"))
                        s_path=f"{os.environ['SLURM_TMPDIR']}/{f_path.name[:-4]}"
                        unzip_directory(f_path, s_path)
                        print(f_path)
                        print(s_path)
                        dtrain= xr.open_zarr(s_path, decode_timedelta=False)

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

                        f_path = Path(CONFIG['paths']['output'].format(
                                 **xs.utils.get_cat_attrs(out, var_as_str=True)))
                        s_path=f"{os.environ['SLURM_TMPDIR']}/{f_path.name}"
                        f_path=str(f_path)
                        xs.save_to_zarr(ds=out, filename=s_path)
                        sh.move(s_path,f_path)
                        pcat.update_from_ds(out, f_path)
                        



                # ---CLEAN UP ---
                if do_task_loop(task= "clean_up",processing_level='cleaned_up'):
                    with context(**CONFIG['clean_up']['context']):
                        #get all adjusted data
                        cat = xs.search_data_catalogs(**CONFIG['clean_up']['search_data_catalogs'],
                                                    other_search_criteria= { 'id': [sim_id],
                                                                            'processing_level':["biasadjusted"],
                                                                            'domain': region_name}
                                                    )
                        dc = cat.popitem()[1]
                        ds = xs.extract_dataset(catalog=dc,
                                                xr_combine_kwargs= {}, # bug xscen
                                                periods= CONFIG['custom']['sim_period']
                                                    )['D']



                        ds = xs.clean_up(ds = ds.chunk({'time':-1}),
                                        **CONFIG['clean_up']['xscen_clean_up'])

                        xs.save_and_update(
                            ds=ds,
                            path=CONFIG['paths']['output'],
                            pcat=pcat,
                            )


                # ---FINAL ZARR ---
                if do_task_loop(task='final_zarr', processing_level='final',format='zarr' ):
                    with context(**CONFIG['final_zarr']['context']):
                        #rechunk and move to final destination
                        fi_path = Path(f"{CONFIG['paths']['final']}".format(**cur_dict,))
                        fi_path.parent.mkdir(exist_ok=True, parents=True)

                        clean_up_path= f"{CONFIG['paths']['output']}".format(
                            **cur_dict, processing_level='cleaned_up')

                        rechunk(path_in=clean_up_path,
                                path_out=fi_path,
                                chunks_over_dim=CONFIG['custom']['out_chunks'],
                                **CONFIG['rechunk'],
                                overwrite=True)

                        # add final file to catalog
                        ds = xr.open_zarr(fi_path)
                        pcat.update_from_ds(ds=ds, path=str(fi_path), info_dict= {'processing_level': 'final'})


                        # if  delete workdir, but save log and regridded
                        if CONFIG['custom']['delete_in_final_zarr']:

                            regriddir=Path(CONFIG['paths']['regriddir'])
                            final_regrid_path = f"{regriddir}/{sim_id}_{region_name}_regridded.zarr"
                            xs.move_and_delete(deleting=[workdir],
                                                moving=[[f"{workdir}/{sim_id}_{region_name}_regridded.zarr", final_regrid_path],],
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
                        xs.save_and_update(ds=hc,path=CONFIG['paths']['checks'], pcat=pcat)


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
                            xs.save_and_update(ds=ds, pcat=pcat, path=CONFIG['paths']['output'])


                        # # if delete workdir, but keep regridded and log
                        if CONFIG['custom']['delete_in_diag']:

                            logger.info('Move files and delete workdir.')

                            regriddir=Path(CONFIG['paths']['regriddir'])
                            final_regrid_path = f"{regriddir}/{sim_id}_{region_name}_regridded.zarr"
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
                                    [f"{workdir}/{sim_id}_{region_name}_diag-improved.zarr",
                                    f"{CONFIG['paths']['diagdir']}/{region_name}/{sim_id}/{sim_id}_{region_name}_diag-improved.zarr"],
                                ],
                                pcat=pcat)
                            sh.rmtree(workdir) # don't to keep empty workdir in this case