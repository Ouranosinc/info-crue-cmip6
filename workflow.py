from dask.distributed import Client
from dask import config as dskconf
import atexit
from pathlib import Path
import xarray as xr
import shutil
import logging
import dask
import numpy as np
from matplotlib import pyplot as plt
import os
import xesmf
import scipy
from dask.diagnostics import ProgressBar
import SBCK

import xclim as xc
import xscen as xs
from xclim import sdba
from xclim.core.calendar import convert_calendar, get_calendar, date_range_like
from xclim.sdba import adjustment
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


from utils import (
    move_then_delete,
    save_move_update,
    python_scp,
    save_and_update,
    rotated_latlon)

server = 'neree_jarre' # not really but bc we can write on jarre from neree, no need to scp

# Load configuration
if server == 'neree': #TODO: put the right config
    load_config('paths_neree.yml', 'config-RSDS.yml', verbose=(__name__ == '__main__'), reset=True)
elif server =='neree_jarre':
    load_config('paths_neree_jarre.yml', 'config-RSDS.yml', verbose=(__name__ == '__main__'), reset=True)
else:
    load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
logger = logging.getLogger('xscen')

workdir = Path(CONFIG['paths']['workdir'])
regriddir = Path(CONFIG['paths']['regriddir'])
refdir = Path(CONFIG['paths']['refdir'])




if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    atexit.register(send_mail_on_exit, subject=CONFIG['scripting']['subject'])

    # defining variables
    ref_period = slice(*map(str, CONFIG['custom']['ref_period']))
    sim_period = slice(*map(str, CONFIG['custom']['sim_period']))
    ref_source = CONFIG['extraction']['ref_source']
    tdd = CONFIG['tdd']

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = ProjectCatalog.create(CONFIG['paths']['project_catalog'], project=CONFIG['project'])

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
                    ds_ref= ds_ref.chunk({'time':365, 'rlat':50, 'rlon':50})
                    #if rotated pole create a ref that is not stack for regrid.
                    # if 'rlat' in ds_ref:
                    #     save_move_update(ds=ds_ref,
                    #                      pcat=pcat,
                    #                      init_path=f"{workdir}/ref_{region_name}_nostack.zarr",
                    #                      final_path=f"{refdir}/ref_{region_name}_nostack.zarr",
                    #                      info_dict={'calendar': 'nostack'
                    #                                 },
                    #                      server=server)


                    # stack
                    if CONFIG['custom']['stack_drop_nans']:

                        # make it work for rlat/rlon
                        #save lat and lon and remove them
                        # if 'rlat' in ds_ref and 'lat' in ds_ref:
                        #     ds_ref.coords.to_dataset().to_netcdf(
                        #         CONFIG['save_rotated'].format(domain=ds_ref.attrs['cat:domain']))
                        #     ds_ref = ds_ref.drop_vars('lat')
                        #     ds_ref = ds_ref.drop_vars('lon')
                        #     ds_ref = ds_ref.drop_vars('rotated_pole')


                        variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                             'variables_and_freqs'].keys())
                        ds_ref = stack_drop_nans(
                            ds_ref,
                            ds_ref[variables[0]].isel(time=130, drop=True).notnull(),
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
            if not pcat.exists_in_cat(domain=region_name, processing_level='diag-ref-prop', source=ref_source):
                with (Client(n_workers=2, threads_per_worker=5, memory_limit="30GB", **daskkws)):

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


                    dref_ref = dref_ref.chunk(**CONFIG['extract']['ref_chunk'])



                    # diagnostics
                    if 'diagnostics' in CONFIG['tasks']:

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


                        save_move_update(ds=ds_ref_prop,
                                         pcat=pcat,
                                         init_path=path_diag_exec,
                                         final_path=path_diag,
                                         server=server
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

                        # stack_drop_nan after regrid only for rotated grid
                        # if 'rlat' in ds_target:
                        #     variable = list(ds_regrid.data_vars)[0]
                        #     # temporary domain to avoir saving to coords
                        #     ds_regrid.attrs['cat:domain']='no'
                        #     ds_regrid = ds_regrid.drop_vars('lat')
                        #     ds_regrid = ds_regrid.drop_vars('lon')
                        #     ds_regrid = ds_regrid.drop_vars('rotated_pole')
                        #     ds_regrid = stack_drop_nans(
                        #         ds_regrid,
                        #         ds_regrid[variable].isel(time=130,drop=True).notnull(),
                        #     )
                        #     ds_regrid.attrs['cat:domain'] = region_name


                        # chunk time dim
                        ds_regrid = ds_regrid.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_regrid.dims})

                        # save to zarr
                        path_rg = f"{workdir}/{sim_id}_{region_name}_regridded.zarr"
                        save_to_zarr(ds=ds_regrid,
                                     filename=path_rg,
                                     encoding=CONFIG['custom']['encoding'],
                                     )
                        pcat.update_from_ds(ds=ds_regrid, path=path_rg)


                # ---BIAS ADJUST---

                from xclim.sdba import adjustment

                # THIS WORKED on QC. not sure these are the best params, but it worked at least once.
                # ---BA-MBCn ---
                if (
                        "ba-MBCn" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id,
                                                   processing_level='biasadjusted')
                ):
                    with measure_time(name=f'MBCn', logger=logger):
                        # load hist ds (simulation)
                        sim = pcat.search(id=sim_id,
                                          processing_level='regridded',
                                          domain=region_name).to_dask(
                            # xarray_open_kwargs={'chunks':{'loc':5}}
                        ).drop_vars('dtr')
                        #sim = sim.chunk({'loc': 1})

                        # load ref ds
                        # choose right calendar
                        simcal = get_calendar(sim)
                        refcal = minimum_calendar(simcal,
                                                  CONFIG['custom'][
                                                      'maximal_calendar'])
                        dref = pcat.search(source=ref_source,
                                           calendar=refcal,
                                           domain=region_name).to_dask(
                        ).drop_vars('dtr')

                        # convert calendar if necessary
                        maximal_calendar = "noleap"
                        align_on = "year"
                        simcal = get_calendar(sim)
                        refcal = get_calendar(dref)
                        mincal = minimum_calendar(simcal, maximal_calendar)
                        if simcal != mincal:
                            sim = convert_calendar(sim, mincal, align_on=align_on)
                        if refcal != mincal:
                            dref = convert_calendar(dref, mincal, align_on=align_on)

                        dsim = sim.sel(time=sim_period)
                        dhist = sim.sel(time=ref_period)

                        nquantiles = 50
                        n_iter = 50 # worked with 20

                        # 1. initial univariate
                        # additive for tasmax
                        if not pcat.exists_in_cat(id=sim_id, domain=region_name, processing_level='ba1-tasmax' ):
                            with Client(n_workers=4, threads_per_worker=3,memory_limit="10GB", **daskkws) as C:
                                print('ba 1 - tasmax')
                                QDMtx = sdba.QuantileDeltaMapping.train(
                                    dref.tasmax, dhist.tasmax, nquantiles=nquantiles,
                                    kind="+",
                                    group="time"
                                )
                                # Adjust both hist and sim, we'll feed both to the Npdf transform.
                                scenh_tx = QDMtx.adjust(dhist.tasmax)

                                scenh_tx= scenh_tx.to_dataset()
                                scenh_tx.attrs.update(dsim.attrs)
                                save_to_zarr(ds=scenh_tx,
                                             filename=f"{workdir}/{sim_id}_{region_name}_scenh_tx_ba1.zarr",
                                             mode='o')

                                scens_tx = QDMtx.adjust(dsim.tasmax)
                                scens_tx = scens_tx.to_dataset()
                                scens_tx.attrs.update(dsim.attrs)
                                scens_tx.attrs['cat:processing_level']='ba1-tasmax'

                                path_adj = f"{workdir}/{sim_id}_{region_name}_scens_tx_ba1.zarr"
                                save_to_zarr(ds=scens_tx,
                                             filename=path_adj,
                                             mode='o')
                                pcat.update_from_ds(ds=scens_tx, path=path_adj)

                            # additive for tasmin
                        if not pcat.exists_in_cat(id=sim_id, domain=region_name,
                                                  processing_level='ba1-tasmin'):
                            with Client(n_workers=4, threads_per_worker=3,
                                        memory_limit="10GB", **daskkws) as C:
                                print('ba 1 - tasmin')
                                QDMtn = sdba.QuantileDeltaMapping.train(
                                    dref.tasmin, dhist.tasmin, nquantiles=nquantiles,
                                    kind="+",
                                    group="time"
                                )
                                # Adjust both hist and sim, we'll feed both to the Npdf transform.
                                scenh_tn = QDMtn.adjust(dhist.tasmin)
                                scens_tn = QDMtn.adjust(dsim.tasmin)

                                scenh_tn = scenh_tn.to_dataset()
                                scenh_tn.attrs.update(dsim.attrs)
                                scens_tn = scens_tn.to_dataset()
                                scens_tn.attrs.update(dsim.attrs)


                                save_to_zarr(ds=scenh_tn,
                                             filename=f"{workdir}/{sim_id}_{region_name}_scenh_tn_ba1.zarr",
                                             mode='o')

                                scens_tn.attrs['cat:processing_level'] = 'ba1-tasmin'

                                path_adj = f"{workdir}/{sim_id}_{region_name}_scens_tn_ba1.zarr"
                                save_to_zarr(ds=scens_tn,
                                             filename=path_adj,
                                             mode='o')
                                pcat.update_from_ds(ds=scens_tn, path=path_adj)

                        # multiplicative for pr
                        if not pcat.exists_in_cat(id=sim_id, domain=region_name,
                                                  processing_level='ba1-pr'):
                            with Client(n_workers=4, threads_per_worker=3,
                                        memory_limit="10GB", **daskkws) as C:
                                print('ba 1 - pr')
                                # remove == 0 values in pr:
                                dref["pr"] = sdba.processing.jitter_under_thresh(dref.pr,
                                                                                 "0.01 mm d-1")
                                dhist["pr"] = sdba.processing.jitter_under_thresh(dhist.pr,
                                                                                  "0.01 mm d-1")
                                dsim["pr"] = sdba.processing.jitter_under_thresh(dsim.pr,
                                                                                 "0.01 mm d-1")


                                QDMpr = sdba.QuantileDeltaMapping.train(
                                    dref.pr, dhist.pr, nquantiles=nquantiles, kind="*",
                                    group="time"
                                )
                                # Adjust both hist and sim, we'll feed both to the Npdf transform.
                                scenh_pr = QDMpr.adjust(dhist.pr)
                                scens_pr = QDMpr.adjust(dsim.pr)

                                scenh_pr = scenh_pr.to_dataset()
                                scenh_pr.attrs.update(dsim.attrs)
                                scens_pr = scens_pr.to_dataset()
                                scens_pr.attrs.update(dsim.attrs)

                                save_to_zarr(ds=scenh_pr,
                                             filename=f"{workdir}/{sim_id}_{region_name}_scenh_pr_ba1.zarr",
                                             mode='o')

                                scens_pr.attrs[
                                    'cat:processing_level'] = 'ba1-pr'

                                path_adj = f"{workdir}/{sim_id}_{region_name}_scens_pr_ba1.zarr"
                                save_to_zarr(ds=scens_pr,
                                             filename=path_adj,
                                             mode='o')
                                pcat.update_from_ds(ds=scens_pr, path=path_adj)


                        if not pcat.exists_in_cat(id=sim_id, domain=region_name,
                                                  processing_level='std'):
                            with Client(n_workers=4, threads_per_worker=3,
                                        memory_limit="10GB", **daskkws) as C:
                                print('std')
                                scenh_tx= xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scenh_tx_ba1.zarr")
                                scenh_tn = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scenh_tn_ba1.zarr")
                                scenh_pr = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scenh_pr_ba1.zarr")
                                scens_tx = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scens_tx_ba1.zarr")
                                scens_tn = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scens_tn_ba1.zarr")
                                scens_pr = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scens_pr_ba1.zarr")

                                scenh = xr.Dataset(
                                    dict(tasmax=scenh_tx.scen, pr=scenh_pr.scen, tasmin=scenh_tn.scen))
                                scens = xr.Dataset(
                                    dict(tasmax=scens_tx.scen, pr=scens_pr.scen, tasmin=scens_tn.scen))



                                # 2. stack and standardize
                                # Stack the variables (tasmax and pr)
                                ref = sdba.processing.stack_variables(dref)
                                scenh = sdba.processing.stack_variables(scenh)
                                scens = sdba.processing.stack_variables(scens)

                                # Standardize
                                ref, _, _ = sdba.processing.standardize(ref)

                                allsim_std, _, _ = sdba.processing.standardize(scens)
                                scenh_std = allsim_std.sel(time=scenh.time)
                                scens_std = allsim_std.sel(time=scens.time)

                                ref = ref.to_dataset()
                                ref.attrs.update(dref.attrs)
                                scenh_std = scenh_std.to_dataset()
                                scenh_std.attrs.update(scenh_tx.attrs)
                                scens_std = scens_std.to_dataset()
                                scens_std.attrs.update(scenh_tx.attrs)

                                save_to_zarr(ds=ref,
                                             filename=f"{workdir}/{sim_id}_{region_name}_ref_std.zarr",
                                             mode='o')
                                save_to_zarr(ds=scenh_std,
                                             filename=f"{workdir}/{sim_id}_{region_name}_scenh_std.zarr",
                                             mode='o')

                                scens_std.attrs['cat:processing_level'] = 'std'
                                path_adj = f"{workdir}/{sim_id}_{region_name}_scens_std.zarr"
                                save_to_zarr(ds=scens_std,
                                             filename=path_adj,
                                             mode='o')
                                pcat.update_from_ds(ds=scens_std, path=path_adj)

                        # 3. Perform the N-dimensional probability density function transform
                        if not pcat.exists_in_cat(id=sim_id, domain=region_name,
                                                      processing_level='npdf'):
                            with Client(n_workers=3, threads_per_worker=3,
                                    memory_limit="15GB", **daskkws) as C:
                                print('npdf')
                                ref = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_ref_std.zarr")
                                scens_std = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scens_std.zarr")
                                scenh_std = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scenh_std.zarr")

                                s_attrs= scenh_std.attrs

                                ref = ref.multivariate#.chunk({'loc':5})
                                scenh_std= scenh_std.multivariate#.chunk({'loc':5})
                                scens_std = scens_std.multivariate#.chunk({'loc':5})


                                # See the advanced notebook for details on how this option work
                                with xc.set_options(sdba_extra_output=True):
                                    out = sdba.adjustment.NpdfTransform.adjust(
                                        ref,
                                        scenh_std,
                                        scens_std,
                                        base=sdba.QuantileDeltaMapping,
                                        # Use QDM as the univariate adjustment.
                                        base_kws={"nquantiles": nquantiles,
                                                  "group": "time"},
                                        n_iter=n_iter,  # perform X iteration
                                        n_escore=1000,
                                        # only send 1000 points to the escore metric (it is realy slow)
                                    )


                                out.attrs.update(s_attrs)
                                out.attrs['cat:processing_level'] = 'npdf'
                                path_adj = f"{workdir}/{sim_id}_{region_name}_npdf.zarr"
                                save_to_zarr(ds=out,
                                             filename=path_adj,
                                             mode='o')
                                pcat.update_from_ds(ds=out, path=path_adj)


                        # 4. Restoring the trend
                        with Client(n_workers=2, threads_per_worker=3,
                                            memory_limit="30GB", **daskkws) as C:
                                print('restore')

                                out = xr.open_zarr(f"{workdir}/{sim_id}_{region_name}_npdf.zarr")

                                scenh_tx = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scenh_tx_ba1.zarr")
                                scenh_tn = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scenh_tn_ba1.zarr")
                                scenh_pr = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scenh_pr_ba1.zarr")
                                scens_tx = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scens_tx_ba1.zarr")
                                scens_tn = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scens_tn_ba1.zarr")
                                scens_pr = xr.open_zarr(
                                    f"{workdir}/{sim_id}_{region_name}_scens_pr_ba1.zarr")

                                scenh = xr.Dataset(
                                    dict(tasmax=scenh_tx.scen, pr=scenh_pr.scen,
                                         tasmin=scenh_tn.scen))
                                scens = xr.Dataset(
                                    dict(tasmax=scens_tx.scen, pr=scens_pr.scen,
                                         tasmin=scens_tn.scen))
                                scenh = sdba.processing.stack_variables(scenh)
                                scens = sdba.processing.stack_variables(scens)

                                scenh_npdft = out.scenh.rename(
                                    time_hist="time")  # Bias-adjusted historical period
                                scens_npdft = out.scen  # Bias-adjusted future period
                                extra = out.drop_vars(["scenh", "scen"])



                                scenh = sdba.processing.reordering(scenh_npdft, scenh,
                                                                   group="time")
                                scens = sdba.processing.reordering(scens_npdft, scens,
                                                                   group="time")

                                scenh = sdba.processing.unstack_variables(scenh)
                                scens = sdba.processing.unstack_variables(scens)


                                ds_scen, escores = dask.compute(scens, extra.escores)


                                ds_scen.attrs.update(sim.attrs)
                                for attrs_k, attrs_v in CONFIG['biasadjust_dOTC'][
                                    'attrs'].items():
                                    ds_scen.attrs[f"cat:{attrs_k}"] = attrs_v

                                ds_scen.attrs["cat:variable"] = \
                                    xs.catalog.parse_from_ds(ds_scen, ["variable"])[
                                        "variable"]

                                save_to_zarr(ds=escores.to_dataset(),
                                             filename=CONFIG['paths']['escores'].format(
                                                 sim_id=sim_id, region_name=region_name),
                                             mode='o')

                                # save and update
                                path_adj = f"{workdir}/{sim_id}_{region_name}_adjusted.zarr"
                                save_to_zarr(ds=ds_scen,
                                             filename=path_adj,
                                             mode='o')
                                pcat.update_from_ds(ds=ds_scen, path=path_adj)


                # ---BA-dOTC ---
                if (
                        "ba-dOTC" in CONFIG["tasks"]
                        and not pcat.exists_in_cat(domain=region_name, id=sim_id,
                                                   processing_level='biasadjusted')
                ):
                    with (
                            Client(n_workers=2, threads_per_worker=3,
                                   memory_limit="30GB", **daskkws),
                            measure_time(name=f'dOTC', logger=logger)
                    ):
                        # load hist ds (simulation)
                        sim = pcat.search(id=sim_id,
                                          processing_level='regridded',
                                          domain=region_name).to_dask().drop_vars('dtr')

                        # load ref ds
                        # choose right calendar
                        simcal = get_calendar(sim)
                        refcal = minimum_calendar(simcal,
                                                  CONFIG['custom']['maximal_calendar'])
                        ds_ref = pcat.search(source=ref_source,
                                             calendar=refcal,
                                             domain=region_name).to_dask().drop_vars('dtr')

                        # convert calendar if necessary
                        maximal_calendar = "noleap"
                        align_on = "year"
                        simcal = get_calendar(sim)
                        refcal = get_calendar(ds_ref)
                        mincal = minimum_calendar(simcal, maximal_calendar)
                        if simcal != mincal:
                            sim = convert_calendar(sim, mincal, align_on=align_on)
                        if refcal != mincal:
                            ds_ref = convert_calendar(ds_ref, mincal, align_on=align_on)

                        # TODO: maybe it would work to do it by season!
                        # TODO: careful with order of dimensions fed to sbck. wants (n_samples,n_features)
                        ds_sim = sim.sel(time=slice('2071', '2100')) #sim_period) # TODO: put back
                        ds_hist = sim.sel(time=ref_period)
                        # TODO: jitter???

                        # stack_var
                        ds_ref = xc.sdba.processing.stack_variables(ds_ref).transpose()
                        ds_hist = xc.sdba.processing.stack_variables(ds_hist).transpose()
                        ds_sim = xc.sdba.processing.stack_variables(ds_sim).transpose()

                        ds_ref = ds_ref.chunk({'loc':1})
                        ds_hist = ds_hist.chunk({'loc': 1})
                        ds_sim = ds_sim.chunk({'loc': 1})

                        ds_scen = adjustment.SBCK_dOTC.adjust(
                            ds_ref,
                            ds_hist,
                            ds_sim,
                            multi_dim="multivar",
                            **CONFIG['biasadjust_dOTC']['adjust'],
                        )
                        ds_scen = xc.sdba.processing.unstack_variables(ds_scen).load()

                        for k, v in CONFIG['biasadjust_dOTC']['attrs'].items():
                            ds_scen.attrs[f"cat_{k}"]= v

                        ds_scen.attrs["cat:variable"] = xs.catalog.parse_from_ds(ds_scen, ["variable"])["variable"]

                        # save and update
                        path_adj = f"{workdir}/{sim_id}_{region_name}_adjusted.zarr"
                        save_to_zarr(ds=ds_scen,
                                     filename=path_adj,
                                     mode='o')
                        pcat.update_from_ds(ds=ds_scen, path=path_adj)


                # --- UNIVARIATE ---
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

                            # make sure sim and ref have the same units

                            # training
                            ds_tr = train(dref=ds_ref,
                                          dhist=ds_hist,
                                          var=[var],
                                          **conf['training_args'])

                            #save and update
                            path_tr = f"{workdir}/{sim_id}_{region_name}_{var}_training_qm.zarr"
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
                            path_adj = f"{workdir}/{sim_id}_{region_name}_{var}_{plevel}.zarr"
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


                            # if var == 'pr' and 'rlat' in ds_ref.dims:
                            #     ds_ref=ds_ref.drop_vars('lat')
                            #     ds_ref = ds_ref.drop_vars('lon')
                            #     ds_ref = ds_ref.drop_vars('rotated_pole')
                            #     ds_hist = ds_hist.drop_vars('lat')
                            #     ds_hist = ds_hist.drop_vars('lon')
                            #     ds_hist = ds_hist.drop_vars('rotated_pole')
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
                            path_adj = f"{workdir}/{sim_id}_{region_name}_{var}_biasadjusted.zarr"
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


                                ds = clean_up(ds = ds.chunk(dict(time=-1)), # TODO: put in MBCn
                                              **CONFIG['clean_up']['xscen_clean_up'])

                                #if rotated pole put back, lat and lon
                                # if 'rlat' in ds:
                                #     ds= rotated_latlon(ds, CONFIG['save_rotated'].format(domain=ds_ref.attrs['cat:domain']))

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
                                path_log = CONFIG['logging']['handlers']['file']['filename']
                                move_then_delete(dirs_to_delete=[workdir],
                                                 moving_files=
                                                 [[f"{workdir}/{sim_id}_{region_name}_regridded.zarr", final_regrid_path],
                                                  [path_log, CONFIG['paths']['logging'].format(**fmtkws)]],
                                                 pcat=pcat)



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
                                             rechunk=CONFIG['extract']['ref_prop_chunk']
                                )
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
                            domain=region_name).to_dataset_dict(**tdd)

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
                                print(f"{workdir}/{sim_id}_{region_name}_regridded.zarr")
                                print(final_regrid_path)
                                path_log = CONFIG['logging']['handlers']['file'][
                                    'filename']
                                move_then_delete(
                                    dirs_to_delete= [
                                        workdir
                                    ],
                                                 moving_files =
                                                 [[f"{workdir}/{sim_id}_{region_name}_regridded.zarr",final_regrid_path],
                                                  [path_log, CONFIG['paths']['logging'].format(**fmtkws)]
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
                                          processing_level=f"+{wl}C"):
                    with (
                            Client(n_workers=6, threads_per_worker=5,
                                   memory_limit="4GB", **daskkws),
                            measure_time(name=f'individual_wl', logger=logger)
                    ):


                        # cut dataset on the wl window
                        ds_wl = xs.extract.subset_warming_level(ds_input, wl=wl)
                        if ds_wl:
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
                                          processing_level=f"horizon{period[0]}-{period[1]}"):
                    with (
                            Client(n_workers=2, threads_per_worker=5,
                                   memory_limit="20GB", **daskkws),
                            measure_time(name=f"horizon {period} for {ds_input.attrs['cat:id']}", logger=logger)
                    ):

                        # needed for some indicators (ideally would have been calculated in clean_up...)
                        ds_cut = ds_input.sel(time= slice(*map(str, period)))
                        ds_cut= ds_cut.chunk({'time':-1, 'lat':20, 'lon':20})
                        ds_cut = ds_cut.assign(tas=xc.atmos.tg(ds=ds_cut)).load()

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

