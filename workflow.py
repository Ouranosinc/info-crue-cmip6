from dask.distributed import Client
from dask import config as dskconf
import atexit
from pathlib import Path
import xarray as xr
import shutil
import numpy as np
import logging
from matplotlib import pyplot as plt
import os
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable

from xclim import atmos, sdba
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
from xscen.scr_utils import measure_time, send_mail, send_mail_on_exit, timeout

from utils import compute_properties,calculate_properties, plot_diagnotics, save_diagnotics

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
    fut_period = slice(*map(str, CONFIG['custom']['future_period']))
    ref_period = slice(*map(str, CONFIG['custom']['ref_period']))
    sim_period = slice(*map(str, CONFIG['custom']['sim_period']))
    check_period = slice(*map(str, CONFIG['custom']['check_period']))

    calendar = CONFIG['custom']['calendar']
    ref_project = CONFIG['extraction']['ref_project']

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = ProjectCatalog.create(CONFIG['paths']['project_catalog'], project=CONFIG['project'], overwrite=True)

    # load project catalog
    pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

    # ---MAKEREF---
    for region_name, region_dict in CONFIG['custom']['regions'].items():
        if (
                "makeref" in CONFIG["tasks"]
                and not pcat.exists_in_cat(domain=region_name, processing_level='extracted', project=ref_project)
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
                                         )
                dref_ref = ds_ref.chunk({'time': -1})  # time period already cut in extract

                # nan count
                ds_ref_props_nan_count= dref_ref.to_array().isnull().sum('time').mean('variable').chunk({'lon': -1, 'lat': -1})
                save_to_zarr(ds_ref_props_nan_count.to_dataset(name='nan_count'),
                             f"{workdir}/ref_{region_name}_nancount.zarr",
                             compute=True, mode=mode)
                                         
                #diagnostics
                if 'diagnostics' in CONFIG['tasks'] :
                    calculate_properties(ds=dref_ref, pcat=pcat, step='ref')

                # stack
                if CONFIG['custom']['stack_drop_nans']:
                    variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                         'variables_and_timedeltas'].keys())
                    ds_ref = stack_drop_nans(
                        ds_ref,
                        ds_ref[variables[0]].isel(time=130, drop=True).notnull(),
                        to_file=f'{refdir}/coords_{region_name}.nc'
                    )
                #chunk
                ds_ref = ds_ref.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_ref.dims})

                # convert calendars
                ds_refnl = convert_calendar(ds_ref, calendar, align_on='date')

                #save
                save_to_zarr(ds_refnl, f"{workdir}/ref_{region_name}_{calendar}.zarr",
                             compute=True, encoding=CONFIG['custom']['encoding'], mode=mode)
                shutil.move( f"{workdir}/ref_{region_name}_{calendar}.zarr",
                             f"{refdir}/ref_{region_name}_{calendar}.zarr"
                             )



                # plot nan_count and email
                ds_ref_props_nan_count = xr.open_zarr(f"{workdir}/ref_{region_name}_nancount.zarr", decode_timedelta=False).load()
                fig, ax = plt.subplots(figsize=(10, 10))
                cmap = plt.cm.winter.copy()
                cmap.set_under('white')
                ds_ref_props_nan_count.nan_count.plot(ax=ax, vmin=1, vmax=1000, cmap=cmap)
                ax.set_title(
                    f'Reference {region_name} - NaN count \nmax {ds_ref_props_nan_count.nan_count.max().item()} out of {dref_ref.time.size}')
                plt.close('all')
                send_mail(
                    subject=f'Reference for region {region_name} - Success',
                    msg=f"Action 'makeref' succeeded for region {region_name}.",
                    attachments=[fig]
                )

                # update cat
                pcat.update_from_ds(ds=ds_refnl, path=f"{refdir}/ref_{region_name}_{calendar}.zarr",
                                        info_dict={'calendar': calendar})

    for sim_id in CONFIG['ids']:
        for exp in CONFIG['experiments']:
            sim_id = sim_id.replace('EXPERIMENT',exp)
            for region_name, region_dict in CONFIG['custom']['regions'].items():
                #depending on the final tasks, check that the final file doesn't already exists
                final = {'check_up': dict(domain=region_name, processing_level='final', id=sim_id),
                         'diagnostics': dict(domain=region_name, processing_level='diag_scen', id=sim_id)}
                if not pcat.exists_in_cat(**final[CONFIG["tasks"][-1]]):

                    fmtkws = {'region_name': region_name,
                              'sim_id': sim_id}
                    print(fmtkws)

                    # ---CUT---
                    if (
                            "cut" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, processing_level='cut', id=sim_id)
                    ):
                        with (
                                Client(n_workers=2, threads_per_worker=5, memory_limit="25GB", **daskkws),
                                measure_time(name='cut', logger=logger)
                        ):
                            # search the data that we need
                            cat_sim = search_data_catalogs(**CONFIG['extraction']['simulations']['search_data_catalogs'],
                                                           other_search_criteria={'id': sim_id})

                            # extract
                            dc = cat_sim[sim_id]
                            ds_sim = extract_dataset(catalog=dc,
                                                     region=region_dict,
                                                     **CONFIG['extraction']['simulations']['extract_dataset'],
                                                     )
                            ds_sim['time'] = ds_sim.time.dt.floor('D')
                            ds_sim = convert_calendar(ds_sim, calendar)

                            # need lat and lon -1 for the regrid
                            ds_sim = ds_sim.chunk({'time': 365, 'lat':-1, 'lon':-1})

                            # save to zarr
                            path_cut = f"{workdir}/{sim_id}_cut.zarr"
                            save_to_zarr(ds=ds_sim,
                                         filename=path_cut,
                                         encoding=CONFIG['custom']['encoding'],
                                         mode=mode
                                         )
                            pcat.update_from_ds(ds=ds_sim, path=path_cut, info_dict={'processing_level':'cut'})

                    # ---REGRID---
                    if (
                            "regrid" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, processing_level='regridded', id=sim_id)
                    ):
                        with (
                                Client(n_workers=2, threads_per_worker=5, memory_limit="25GB", **daskkws),
                                measure_time(name='regrid', logger=logger)
                        ):
                            #get sim
                            ds_sim = pcat.search(id=sim_id,
                                                 processing_level='cut',
                                                 domain=region_name).to_dataset_dict().popitem()[1]

                            # get reference
                            ds_refnl = pcat.search(project=ref_project,
                                                   calendar=calendar,
                                                   domain=region_name).to_dataset_dict().popitem()[1]
                            # regrid
                            ds_sim_regrid = regrid(
                                ds=ds_sim,
                                ds_grid=ds_refnl,
                                **CONFIG['regrid']
                            )

                            # chunk
                            ds_sim_regrid = ds_sim_regrid.chunk({d: CONFIG['custom']['chunks'][d] for d in ds_sim_regrid.dims})

                            # save to zarr
                            path_rg = f"{workdir}/{sim_id}_regridded.zarr"
                            save_to_zarr(ds=ds_sim_regrid,
                                         filename=path_rg,
                                         encoding=CONFIG['custom']['encoding'],
                                         mode=mode
                                         )
                            pcat.update_from_ds(ds=ds_sim_regrid, path=path_rg)


                    # --- SIM PROPERTIES ---
                    if ("simproperties" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_simprops")
                    ):
                        with (
                                Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                                measure_time(name=f'simproperties', logger=logger),
                                timeout(3600, task='simproperties')
                        ):
                            #get sim, ref
                            ds_sim = pcat.search(id=sim_id,
                                                 processing_level='regridded',
                                                 domain=region_name).to_dataset_dict().popitem()[1]
                            simcal = get_calendar(ds_sim)
                            ds_ref = pcat.search(project=ref_project,
                                                 calendar= simcal,
                                                 domain=region_name).to_dataset_dict().popitem()[1]
                            #properties
                            ds_sim_props = compute_properties(ds_sim, ds_ref, check_period)
                            ds_sim_props.attrs.update(ds_sim.attrs)

                            #save
                            path_sim = Path(CONFIG['paths']['checkups'].format(region_name=region_name, sim_id=sim_id, step='sim'))
                            path_sim.parent.mkdir(exist_ok=True, parents=True)
                            path_sim_exec  = f"{workdir}/{path_sim.name}"
                            save_to_zarr(ds=ds_sim_props,
                                         filename=path_sim_exec,
                                         mode=mode,
                                         itervar=True
                                         )
                            shutil.move(path_sim_exec, path_sim)

                            logger.info('Sim properties computed, painting nan count and sending plot.')

                            ds_sim_props_unstack = unstack_fill_nan(ds_sim_props, coords=refdir / f'coords_{region_name}.nc')
                            nan_count = ds_sim_props_unstack.nan_count.load()

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

                            #update cat
                            pcat.update_from_ds(ds=ds_sim_props,
                                                info_dict={'id': f"{sim_id}_simprops",
                                                           'processing_level': 'properties'},
                                                path=str(path_sim))

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
                                ds_hist = pcat.search(id=sim_id,processing_level='regridded',domain=region_name).to_dataset_dict().popitem()[1]

                                # load ref ds
                                # choose right calendar
                                simcal = get_calendar(ds_hist)
                                refcal = minimum_calendar(simcal,
                                                          CONFIG['custom']['calendar'])
                                ds_ref = pcat.search(project = ref_project,
                                                     calendar=refcal,
                                                     domain=region_name).to_dataset_dict().popitem()[1]

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
                                # load sim ds and training dataset
                                ds_sim = pcat.search(id=sim_id,
                                                     processing_level = 'regridded',
                                                     domain=region_name).to_dataset_dict().popitem()[1]
                                ds_tr = pcat.search(id=f'{sim_id}_training_qm_{var}', domain=region_name).to_dataset_dict().popitem()[1]

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
                                                     processing_level='regridded').to_dataset_dict().popitem()[1]
                                ds_ref = pcat.search(domain=region_name,
                                                     project=ref_project,
                                                     calendar=calendar
                                                     ).to_dataset_dict().popitem()[1]

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
                                # load sim and extreme training dataset
                                ds_sim = pcat.search(id=sim_id, domain=region_name,
                                                     processing_level='regridded').to_dataset_dict().popitem()[1]

                                ds_tr = pcat.search(id=f'{sim_id}_training_ex_{var}', domain=region_name).to_dataset_dict().popitem()[1]


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
                                                      )


                            # convert units
                            ds['pr'] = convert_units_to(ds['pr'], 'mm/d')
                            ds['tasmax'] = convert_units_to(ds['tasmax'], 'degC')
                            ds['tasmin'] = convert_units_to(ds['tasmin'], 'degC')

                            # put back feb 29th
                            with_missing = convert_calendar(ds, 'standard', missing=np.NaN)
                            ds = with_missing.interpolate_na('time', method='linear')


                            # unstack nans
                            if CONFIG['custom']['stack_drop_nans']:
                                ds = unstack_fill_nan(ds, coords=f"{refdir}/coords_{region_name}.nc")
                                ds = ds.chunk({d: CONFIG['custom']['chunks'][d] for d in ds.dims})

                            # fix attrs
                            ds.attrs['cat/id'] = sim_id
                            #ds.attrs.pop('cat/path')
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

                            # remove attrs of pr because too long with ExtremeValues
                            # This was fix with a new version of xclim, but that version is incompatible with streamlit
                            # del ds.pr.attrs['bias_adjustment']
                            # del ds.pr.attrs['history']

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

                            #  move regridded to save it permantly
                            final_regrid_path =f"{regriddir}/{sim_id}_regridded.zarr"
                            shutil.move(f"{workdir}/{sim_id}_regridded.zarr", final_regrid_path )
                            ds_sim=xr.open_zarr(final_regrid_path)
                            pcat.update_from_ds(ds=ds_sim, path = final_regrid_path)

                            # move log to save it permantly
                            path_log = CONFIG['logging']['handlers']['file']['filename']
                            shutil.move(path_log, CONFIG['paths']['logging'].format(**fmtkws) )

                            # erase workdir content
                            if workdir.exists() and workdir.is_dir():
                                shutil.rmtree(workdir)
                                os.mkdir(workdir)

                            # add final file to catalog
                            ds = xr.open_zarr(fi_path)
                            pcat.update_from_ds(ds=ds, path=str(fi_path), info_dict= {'processing_level': 'final'})

                    # --- SCEN PROPS ---
                    if (
                            "scenproperties" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=f"{sim_id}_scenprops")
                    ):
                        with (
                                Client(n_workers=9, threads_per_worker=3, memory_limit="7GB", **daskkws),
                                measure_time(name=f'scenprops', logger=logger),
                                timeout(5400, task='scenproperties')
                        ):
                            ds_scen = pcat.search(id=sim_id,processing_level='final', domain=region_name).to_dataset_dict().popitem()[1]


                            scen_cal = get_calendar(ds_scen)
                            ds_ref = maybe_unstack(
                                pcat.search(project = ref_project,
                                            calendar=scen_cal,
                                            domain=region_name).to_dataset_dict().popitem()[1],
                                stack_drop_nans=CONFIG['custom']['stack_drop_nans'],
                                coords=refdir / f'coords_{region_name}.nc',
                                rechunk={d: CONFIG['custom']['out_chunks'][d] for d in ['lat', 'lon']}
                            )

                            ds_scen_props = compute_properties(ds_scen, ds_ref, check_period)
                            ds_scen_props.attrs.update(ds_scen.attrs)

                            path_scen = Path(CONFIG['paths']['checkups'].format(region_name=region_name, sim_id=sim_id, step='scen'))
                            path_scen_exec = f"{workdir}/{path_scen.name}"

                            save_to_zarr(ds=ds_scen_props,filename=path_scen_exec,mode=mode,itervar=True)
                            shutil.move(path_scen_exec,path_scen)

                            pcat.update_from_ds(ds=ds_scen_props,
                                                info_dict={'id': f"{sim_id}_scenprops",
                                                           'processing_level': 'properties'},
                                                path=str(path_scen))

                    # --- CHECK UP ---
                    if "check_up" in CONFIG["tasks"]:
                        with (
                                Client(n_workers=6, threads_per_worker=3, memory_limit="10GB", **daskkws),
                                measure_time(name=f'checkup', logger=logger)
                        ):

                            sim = maybe_unstack(
                                pcat.search(id=f'{sim_id}_simprops', domain=region_name).to_dataset_dict().popitem()[1],
                                coords=refdir / f'coords_{region_name}.nc',
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

                            for var in [ 'tx_mean_rmse', 'tn_mean_rmse', 'prcptot_rmse']:
                                fig_compare_and_diff(
                                    sim[var], scen[var], op = "improvement"
                                ).savefig(fig_dir / f'{var}_compare.png')
                                paths.append(fig_dir / f'{var}_compare.png')

                            send_mail(
                                subject=f"{sim_id}/{region_name} - Succès",
                                msg=f"Toutes les étapes demandées pour la simulation {sim_id}/{region_name} ont été accomplies.",
                                attachments=paths
                            )
                            plt.close('all')


                    # ---DIAGNOSTICS ---
                    if (
                            "diagnostics" in CONFIG["tasks"]
                            and not pcat.exists_in_cat(domain=region_name, id=sim_id, processing_level='scen_diag')
                    ):
                        with (
                                Client(n_workers=3, threads_per_worker=5, memory_limit="20GB", **daskkws),
                                measure_time(name=f'diagnostics', logger=logger)
                        ):
                            #load initial data
                            ds_scen = pcat.search(processing_level='final',
                                                  id=sim_id,
                                                  domain=region_name
                                                  ).to_dataset_dict().popitem()[1].chunk({'time': -1}).sel(time=ref_period)

                            ds_sim = pcat.search(processing_level='regridded',
                                                 id=sim_id,
                                                 domain=region_name
                                                 ).to_dataset_dict().popitem()[1].chunk({'time': -1}).sel(time=ref_period)

                            # properties
                            sim = calculate_properties(ds=ds_sim, pcat=pcat, step='sim', unstack=True)
                            scen = calculate_properties(ds=ds_scen, pcat=pcat, step='scen')

                            #get ref properties calculated earlier in makeref
                            ref = pcat.search(project=ref_project,
                                               processing_level='diag_ref',
                                               domain=region_name).to_dataset_dict().popitem()[1]

                            #create plots
                            #paths = plot_diagnotics(ref, sim, scen)

                            save_diagnotics(ref, sim, scen, pcat)

                            # send_mail(
                            # subject=f"{sim_id}/{region_name} - Diagnostics",
                            # msg=f"Diagnostiques pour la simulation {sim_id}/{region_name} ont été accomplies.",
                            # attachments=paths
                            # )
