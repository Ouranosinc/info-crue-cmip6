from dask.distributed import Client
from dask import config as dskconf
import atexit
from pathlib import Path
import xarray as xr
import shutil
import logging
from matplotlib import pyplot as plt
import os
import xesmf
import numpy as np

from xclim.core.calendar import convert_calendar, get_calendar, date_range_like
from xclim.core.units import convert_units_to
from xclim.sdba import properties, measures, construct_moving_yearly_window, unpack_moving_yearly_window

from xscen.checkups import fig_compare_and_diff, fig_bias_compare_and_diff
from xscen.catalog import ProjectCatalog, parse_directory, parse_from_ds, DataCatalog
from xscen.extraction import search_data_catalogs, extract_dataset
from xscen.io import save_to_zarr, rechunk
from xscen.config import CONFIG, load_config
from xscen.common import minimum_calendar, translate_time_chunk, stack_drop_nans, unstack_fill_nan, maybe_unstack
from xscen.regridding import regrid
from xscen.biasadjust import train, adjust
from xscen.scr_utils import measure_time, send_mail, send_mail_on_exit, timeout, TimeoutException

from utils import calculate_properties, measures_and_heatmap,email_nan_count,move_then_delete

# Load configuration
load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
logger = logging.getLogger('xscen')

workdir = Path(CONFIG['paths']['workdir'])
refdir = Path(CONFIG['paths']['refdir'])
regriddir = Path(CONFIG['paths']['regriddir'])
mode = 'o'



if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    atexit.register(send_mail_on_exit, subject=CONFIG['scr_utils']['subject'])

    # defining variables
    ref_period = slice(*map(str, CONFIG['custom']['ref_period']))
    sim_period = slice(*map(str, CONFIG['custom']['sim_period']))
    calendar = CONFIG['custom']['calendar']
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
                and not pcat.exists_in_cat(domain=region_name, processing_level='extracted', source=ref_source)
        ):
            with (
                    Client(n_workers=3, threads_per_worker=5, memory_limit="15GB", **daskkws),
                    measure_time(name='makeref', logger=logger)
            ):
                # search
                cat_ref = search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

                # extract
                dc = cat_ref.popitem()[1]
                ds_ref = extract_dataset(catalog=dc,
                                         region=region_dict,
                                         **CONFIG['extraction']['reference']['extract_dataset']
                                         )['D']


                #diagnostics
                if 'diagnostics' in CONFIG['tasks'] :
                    # drop to make faster
                    dref_ref = ds_ref.drop_vars('dtr')
                    dref_ref = dref_ref.chunk({'time': -1})  # to help diag and nan_count

                    ds_ref_prop = calculate_properties(ds=dref_ref,
                                                       diag_dict=CONFIG['diagnostics']['properties'],
                                                       unit_conversion=CONFIG['clean_up']['units'])

                    path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=region_name,
                                                                           sim_id=ds_ref.attrs['cat/id'], # TODO CHANGE BACK
                                                                           #sim_id=ds_ref.attrs['intake_esm_attrs/id'],
                                                                           step='ref'))
                    path_diag_exec = f"{workdir}/{path_diag.name}"
                    save_to_zarr(ds=ds_ref_prop, filename=path_diag_exec, mode='o', itervar=True)
                    shutil.move(path_diag_exec, path_diag)
                    pcat.update_from_ds(ds=ds_ref_prop,
                                        info_dict={'processing_level': f'diag_ref'},
                                        path=str(path_diag))

                    # nan count
                    ds_ref_props_nan_count = dref_ref.to_array().isnull().sum('time').mean('variable').chunk(
                        {'lon': -1, 'lat': -1})
                    save_to_zarr(ds_ref_props_nan_count.to_dataset(name='nan_count'),
                                 f"{workdir}/ref_{region_name}_nancount.zarr",
                                 compute=True, mode=mode)
                    # plot nan_count and email
                    email_nan_count(path=f"{workdir}/ref_{region_name}_nancount.zarr", region_name=region_name)

                # drop tasmin, it was only needed for the diagnostics
                ds_ref = ds_ref.drop_vars('tasmin')

                # stack
                if CONFIG['custom']['stack_drop_nans']:
                    variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                         'variables_and_freqs'].keys())
                    ds_ref = stack_drop_nans(
                        ds_ref,
                        ds_ref[variables[0]].isel(time=130, drop=True).notnull(),
                        to_file=f'{refdir}/coords_{region_name}.nc'
                    )
                #chunk
                ds_ref = ds_ref.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})

                # convert calendars
                ds_refnl = convert_calendar(ds_ref, calendar)

                #save
                save_to_zarr(ds_refnl, f"{workdir}/ref_{region_name}_{calendar}.zarr",
                             compute=True, encoding=CONFIG['custom']['encoding'], mode=mode)
                shutil.move( f"{workdir}/ref_{region_name}_{calendar}.zarr",
                             f"{refdir}/ref_{region_name}_{calendar}.zarr"
                             )


                # update cat
                pcat.update_from_ds(ds=ds_refnl, path=f"{refdir}/ref_{region_name}_{calendar}.zarr",
                                        info_dict={'calendar': calendar})

    for sim_id in CONFIG['ids']:
        for exp in CONFIG['experiments']:
            sim_id = sim_id.replace('EXPERIMENT',exp)
            for region_name, region_dict in CONFIG['custom']['regions'].items():
                fmtkws = {'region_name': region_name, 'sim_id': sim_id}
                #depending on the final tasks, check that the final file doesn't already exists
                final = {'check_up': dict(domain=region_name, processing_level='final', id=sim_id),
                         'diagnostics': dict(domain=region_name, processing_level='diag_scen_meas', id=sim_id)}
                if not pcat.exists_in_cat(**final[CONFIG["tasks"][-1]]):

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
                                    # search the data that we need
                                    cat_sim = search_data_catalogs(**CONFIG['extraction']['simulations']['search_data_catalogs'],
                                                                    other_search_criteria={'id': sim_id})

                                    # extract
                                    dc = cat_sim[sim_id]
                                    # buffer is need to take a bit larger than actual domain, to avoid weird effect at the edge
                                    # domain will be cut to the right shape during the regrid
                                    region_dict['buffer']=1.5
                                    ds_sim = extract_dataset(catalog=dc,
                                                             region=region_dict,
                                                             **CONFIG['extraction']['simulations']['extract_dataset'],
                                                             )['D']
                                    ds_sim['time'] = ds_sim.time.dt.floor('D') # probably this wont be need when data is cleaned
                                    ds_sim = convert_calendar(ds_sim, calendar)

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
                            # iter over all regriddings
                            for reg_name, reg_dict in CONFIG['regrid'].items():
                                print(reg_dict)
                                # choose input
                                if reg_dict['input'] == 'cur_sim': # get current extracted simulation
                                    ds_in = pcat.search(id=sim_id,processing_level='extracted',domain=region_name).to_dask()
                                elif reg_dict['input'] == 'previous': # get results of previous regridding in the loop
                                    ds_in = ds_regrid

                                # choose target grid
                                if 'cf_grid_2d' in reg_dict['target']: # create a regular 2d grid
                                    ds_target = xesmf.util.cf_grid_2d(**reg_dict['target']['cf_grid_2d'])
                                    ds_target.attrs['cat/domain']=reg_name # need this in xscen regrid
                                elif 'search' in reg_dict['target']: # search a grid in the catalog
                                    ds_target = pcat.search(**reg_dict['target']['search'],domain=region_name).to_dask()

                                ds_regrid = regrid(
                                    ds=ds_in,
                                    ds_grid=ds_target,
                                    **reg_dict['xscen_regrid']
                                )


                            # chunk
                            ds_regrid = ds_regrid.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_regrid.dims})

                            # save to zarr
                            path_rg = f"{workdir}/{sim_id}_regridded.zarr"
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
                                refcal = minimum_calendar(simcal, CONFIG['custom']['calendar'])
                                ds_ref = pcat.search(source=ref_source,
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
                                                               'xrfreq': ds_hist.attrs['cat/xrfreq']
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
                                ds_ref = pcat.search(domain=region_name,
                                                     source=ref_source,
                                                     calendar=calendar
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
                                                               'xrfreq': ds_hist.attrs['cat/xrfreq']
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
                        with (
                                Client(n_workers=4, threads_per_worker=3, memory_limit="15GB", **daskkws),
                                measure_time(name=f'cleanup', logger=logger)
                        ):
                            #get all adjusted data
                            cat = search_data_catalogs(**CONFIG['clean_up']['search_data_catalogs'],
                                                       other_search_criteria= { 'id': [sim_id],
                                                                                'processing_level':["biasadjusted"],
                                                                                'domain': region_name}
                                                        )
                            dc = cat.popitem()[1]
                            ds = extract_dataset(catalog=dc,
                                                 to_level='cleaned_up',
                                                 periods= CONFIG['custom']['sim_period']
                                                      )['D']

                            # convert units
                            for var, unit in CONFIG['clean_up']['units'].items():
                                ds[var]=convert_units_to(ds[var], unit)


                            # put back feb 29th
                            with_missing = convert_calendar(ds, 'standard', missing=np.NaN)
                            ds = with_missing.interpolate_na('time', method='linear')


                            # unstack nans
                            if CONFIG['custom']['stack_drop_nans']:
                                ds = unstack_fill_nan(ds, coords=f"{refdir}/coords_{region_name}.nc")
                                ds = ds.chunk({d: CONFIG['custom']['chunks'][d] for d in ds.dims})

                            # remove all global attrs that don't come from the catalogue
                            for attr in list(ds.attrs.keys()):
                                if attr[:4] != 'cat/':
                                    del ds.attrs[attr]

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


                            #save and update
                            path_cu = f"{workdir}/{sim_id}_cleaned_up.zarr"
                            save_to_zarr(ds=ds,
                                         filename=path_cu,
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

                            # if this is last step, delete workdir, but save log and regridded
                            if CONFIG["tasks"][-1] == 'final_zarr':
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
                            and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='diag_scen_meas')
                    ):
                        with (
                                Client(n_workers=8, threads_per_worker=5, memory_limit="5GB", **daskkws),
                                measure_time(name=f'diagnostics', logger=logger)
                        ):
                            #load initial data
                            ds_scen = pcat.search(processing_level='final',
                                                  id=sim_id,
                                                  domain=region_name
                                                  ).to_dask().chunk({'time': -1}).sel(time=ref_period)

                            ds_sim = pcat.search(processing_level='regridded',
                                                 id=sim_id,
                                                 domain=region_name
                                                 ).to_dask().chunk({'time': -1}).sel(time=ref_period)

                            # properties
                            sim = calculate_properties(ds=ds_sim,
                                                       diag_dict=CONFIG['diagnostics']['properties'],
                                                       unstack=CONFIG['custom']['stack_drop_nans'],
                                                       path_coords=refdir / f'coords_{region_name}.nc',
                                                       unit_conversion=CONFIG['clean_up']['units'])
                            scen = calculate_properties(ds=ds_scen,
                                                        diag_dict=CONFIG['diagnostics']['properties'],
                                                        unit_conversion=CONFIG['clean_up']['units'])

                            #get ref properties calculated earlier in makeref
                            ref = pcat.search(source=ref_source,
                                               processing_level='diag_ref',
                                               domain=region_name).to_dask()

                            # calculate measures and diagnostic heat map
                            [meas_sim, meas_scen], hmap = measures_and_heatmap(ref=ref, sims=[sim, scen])

                            # save hmap
                            path_diag = Path(
                                CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat/domain'],
                                                                      sim_id=scen.attrs['cat/id'],
                                                                      step='hmap'))
                            path_diag = path_diag.with_suffix('.npy')  # replace zarr by npy
                            np.save(path_diag, hmap)

                            # save and update properties and biases/measures
                            for ds, step in zip([sim, scen, meas_sim, meas_scen],
                                                ["sim", "scen", 'sim_meas', 'scen_meas']):
                                path_diag = Path(
                                    CONFIG['paths']['diagnostics'].format(region_name=region_name,
                                                                          sim_id=sim_id,
                                                                          step=step))
                                path_diag_exec = f"{workdir}/{path_diag.name}"
                                save_to_zarr(ds=ds, filename=path_diag_exec, mode='o', itervar=True)
                                shutil.move(path_diag_exec, path_diag)
                                pcat.update_from_ds(ds=ds,
                                                    info_dict={'processing_level': f'diag_{step}'},
                                                    path=str(path_diag))


                                # if this is the last step, delete workdir, but keep regridded and log
                                if CONFIG["tasks"][-1] == 'diagnostics':
                                    final_regrid_path = f"{regriddir}/{sim_id}_{region_name}_regridded.zarr"
                                    path_log = CONFIG['logging']['handlers']['file']['filename']
                                    move_then_delete(dirs_to_delete= [workdir],
                                                     moving_files =
                                                     [[f"{workdir}/{sim_id}_regridded.zarr",final_regrid_path],
                                                      [path_log, CONFIG['paths']['logging'].format(**fmtkws)]],
                                                      pcat=pcat)

                            send_mail(
                                subject=f"{sim_id}/{region_name} - Succès",
                                msg=f"Toutes les étapes demandées pour la simulation {sim_id}/{region_name} ont été accomplies.",
                            )
