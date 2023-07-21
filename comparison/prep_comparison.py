"""
Worflow to compute indicators for comparison of different versions
Use with info-crue-6-experimental env. (with new xscen)
"""

import xscen as xs
from xscen import CONFIG
from dask.distributed import Client
from dask import config as dskconf
from dask.diagnostics import ProgressBar
import atexit
import xclim as xc
import logging
import xarray as xr
import dask

import sys
sys.path.append("..")
from utils import comp_transfer

path = 'paths_comparison.yml'  #TODO: put the right path config
config = 'config_comparison.yml'#TODO: put the right config

# Load configuration
xs.load_config(path, config, verbose=(__name__ == '__main__'), reset=True)
logger = logging.getLogger('xscen')


if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})
    atexit.register(xs.send_mail_on_exit, subject=CONFIG['scripting']['subject'])
    tdd = CONFIG['tdd']

    # initialize Project Catalog
    if "initialize_pcat" in CONFIG["tasks"]:
        pcat = xs.ProjectCatalog.create(CONFIG['paths']['project_catalog'], project=CONFIG['project'])

    # load catalogs
    pcat = xs.ProjectCatalog(CONFIG['paths']['project_catalog'])
    pcat_extra = xs.ProjectCatalog(CONFIG['paths']['extra_catalog'])
    pcat_E5L = xs.ProjectCatalog(CONFIG['paths']['E5L_catalog'])
    pcat_cmip5 = xs.ProjectCatalog(CONFIG['paths']['cmip5_catalog'])


    # --- HORIZONS ---
    if 'horizons' in CONFIG['tasks']:
        # get inputs
        dict_input1 = pcat_extra.search(**CONFIG['horizons']['input']).to_dataset_dict()
        dict_input2 = pcat_E5L.search(**CONFIG['horizons']['input']).to_dataset_dict()
        dict_input3 = pcat_cmip5.search(**CONFIG['horizons']['input']).to_dataset_dict()
        dict_input= dict_input1|dict_input2|dict_input3
        for name_input, ds_input in dict_input.items():

            # fix domain names. in this project, I differentiate the different datasets by domain
            if ds_input.attrs['cat:activity'] == 'CMIP5':
                ds_input.attrs['cat:domain'] = 'QC-CM5'
                if '(m)' in ds_input.attrs['cat:id']: # these character don't play well with the workflow
                   ds_input.attrs['cat:id'] = ds_input.attrs['cat:id'].replace('(m)', '-M')
            elif ds_input.attrs['cat:domain'] == 'QC':
                ds_input.attrs['cat:domain'] = 'QC-E5L'

            for period in CONFIG['horizons']['periods']:
                mod = xs.indicators.load_xclim_module(CONFIG['paths']['indicators'])
                for name, ind in mod.iter_indicators():
                    if not pcat.exists_in_cat(id=ds_input.attrs['cat:id'],
                                              variable=name,
                                              domain=ds_input.attrs['cat:domain'],
                                              processing_level=f"comp-horizon{period[0]}-{period[1]}"):
                        with (
                                Client(n_workers=2, threads_per_worker=5,memory_limit="20GB", **daskkws),
                                xs.measure_time(name=f"horizon {period} for {ds_input.attrs['cat:id']}")
                        ):
                            # prepare input
                            # cmip5 doesn't always go to 2100 (INFO-Crue-CM5_CMIP5_NASA-GISS_GISS-E2-H_rcp45_r6i1p3_QC)
                            if period[1]=='2100' and str(ds_input.time[-1].dt.year.values) != '2100' and ds_input.attrs['cat:domain'] == 'QC-CM5':
                                period_cur =  [period[0], '2099']
                                logger.info('CMIP5 dataset does not go to 2100, cutting a year')
                            else:
                                period_cur=period
                            ds_cut = ds_input.assign(tas=xc.atmos.tg(ds=ds_input))

                            #produce horizon
                            ds_hor = xs.aggregate.produce_horizon(
                                ds_cut,
                                period=period_cur,
                                indicators=[(name, ind)],
                                to_level= "comp-horizon{period0}-{period1}".format(period0=period[0], period1=period[1]),
                            )
                            ds_hor.attrs['cat:variable']=name
                            # because of INFO-Crue-CM5_CMIP5_NASA-GISS_GISS-E2-H_rcp45_r6i1p3_QC
                            ds_hor['horizon']= ["{period0}-{period1}".format(period0=period[0], period1=period[1])]

                            # save and update
                            xs.save_and_update(ds_hor,
                                               pcat= pcat,
                                               path=CONFIG['paths']['work_comp'],
                                               )

        #scp to final destination
        comp_transfer(workdir=CONFIG['paths']['workdir'],
                      final_dest=CONFIG['paths']['final_dest'],
                      pcat=pcat,
                      scp_kwargs=CONFIG['scp'])




    # --- DELTAS ---
    if 'deltas' in CONFIG['tasks']:
        dict_input = pcat.search(**CONFIG['deltas']['input']).to_dataset_dict(**tdd)
        for name_input, ds_input in dict_input.items():
            id = ds_input.attrs['cat:id']
            level = f"delta-{ds_input.attrs['cat:processing_level']}"
            for var in ds_input.data_vars:
                if not pcat.exists_in_cat(id=id,
                                          variable=f"{var}_delta_1991_2020",
                                          domain=ds_input.attrs['cat:domain'],
                                          processing_level=level):
                    with (
                            Client(n_workers=2, threads_per_worker=5, memory_limit="12GB",
                                   **daskkws),
                            xs.measure_time(name=f"{name_input} {level}", logger=logger)
                    ):
                        # get ref dataset
                        ds_ref = pcat.search(id=id,
                                             variable=var,
                                             domain=ds_input.attrs['cat:domain'],
                                             **CONFIG['deltas']['reference']).to_dask(**tdd)

                        # concat past and future
                        ds_concat = xr.concat([ds_input[[var]], ds_ref], dim='horizon',
                                              combine_attrs='override')

                        # compute delta
                        ds_delta = xs.aggregate.compute_deltas(
                            ds=ds_concat,
                            reference_horizon=ds_ref.horizon.values[0],
                            to_level=level
                        )
                        ds_delta.attrs['cat:variable'] = var

                        # save and update
                        xs.save_and_update(ds_delta,
                                           pcat=pcat,
                                           path=CONFIG['paths']['work_comp'],
                                           )

                    # scp to final destination
        comp_transfer(workdir=CONFIG['paths']['workdir'],
                      final_dest=CONFIG['paths']['final_dest'],
                      pcat=pcat,
                      scp_kwargs=CONFIG['scp'])

    # --- REGRID ---

    if 'regrid' in CONFIG['tasks']:
        dict_input = pcat.search(**CONFIG['regrid']['input']).to_dataset_dict(**tdd)
        # regrid on rdrs grid
        ds_target = pcat.search(**CONFIG['regrid']['target']).to_dataset(**tdd)
        for name_input, ds_input in dict_input.items():
            id = ds_input.attrs['cat:id']
            level = ds_input.attrs['cat:processing_level']
            domain= f"{ds_input.attrs['cat:domain']}2rdrs"
            for var in ds_input.data_vars:
                if not pcat.exists_in_cat(id=id,
                                          variable=var,
                                          domain=domain,
                                          processing_level=level):
                    with (
                        Client(n_workers=2, threads_per_worker=5, memory_limit="12GB",**daskkws),
                        xs.measure_time(name=f" regrid {id} {level} {domain}", logger=logger)
                    ):

                        # regrid
                        ds_regrid = xs.regrid_dataset(
                            ds=ds_input[[var]].chunk({'lat':-1, 'lon':-1}),
                            ds_grid=ds_target,
                            to_level= level,
                            **CONFIG['regrid']['regrid_dataset']
                        )

                        # new domain name
                        ds_regrid.attrs['cat:domain']= domain
                        ds_regrid.attrs['cat:variable'] = var

                        # save and update
                        xs.save_and_update(ds_regrid,
                                           pcat=pcat,
                                           path=CONFIG['paths']['work_comp'],
                                           )

        # scp to final destination
        comp_transfer(workdir=CONFIG['paths']['workdir'],
                      final_dest=CONFIG['paths']['final_dest'],
                      pcat=pcat,
                      scp_kwargs=CONFIG['scp'])


    # --- ENSEMBLES ---
    if 'ensembles' in CONFIG['tasks']:
        # get each groups for which we want ensembles
        for group_name, group_inputs in CONFIG['ensembles']['groups'].items():
            group_cat = pcat.search(**group_inputs)
            rechunk = {'lat': -1, 'lon': -1} if group_name in ['EMDNA', 'E5L'] else {
                'rlat': -1, 'rlon': -1}

            for level in group_cat.df.processing_level.unique():
                if 'ensemble' not in level:
                    ens_name = f'ensemble-{level}-{group_name}'
                    for experiment in group_cat.df.experiment.unique():
                        sub_group_cat = group_cat.search(processing_level=level, experiment=experiment)
                        for var in sub_group_cat.df.variable.unique():

                            if not pcat.exists_in_cat(processing_level=ens_name,
                                                      variable=f"{var[0]}_p50",
                                                      experiment=experiment,):
                                with (
                                        ProgressBar(),
                                        xs.measure_time(name=f'ensemble {ens_name}', logger=logger)
                                ):
                                    # get datasets to create 1 ensemble/file
                                    datasets = sub_group_cat.search(variable=var).to_dataset_dict(**tdd)

                                    # weigths for ensemble
                                    weights = xs.ensembles.generate_weights(datasets=datasets)

                                    # calculate ensemble stats
                                    ds_ens = xs.ensemble_stats(datasets=datasets,
                                                               weights=weights,
                                                               to_level=ens_name)

                                    # change units, should have done it in indicators...
                                    for vv in ds_ens.data_vars:
                                        if var[0] in ['tx_mean', 'tn_mean']:
                                            ds_ens[vv] = xc.units.convert_units_to(ds_ens[vv], 'degC')

                                    if 'cat:domain' not in ds_ens.attrs:
                                        ds_ens.attrs['cat:domain']=f"QC-{group_name}2RDRS"

                                    # save and update
                                    xs.save_and_update(ds_ens,
                                                       pcat=pcat,
                                                       path=CONFIG['paths']['work_comp'],
                                                       save_kwargs = dict(rechunk= rechunk),
                                                       )

        # # scp to final destination
        comp_transfer(workdir=CONFIG['paths']['workdir'],
                      final_dest=CONFIG['paths']['final_dest'],
                      pcat=pcat,
                      scp_kwargs=CONFIG['scp'])


