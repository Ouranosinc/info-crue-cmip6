
import os
import xscen as xs
from xscen import CONFIG
from workflow.scripts.utils import dask_cluster, zip_directory, tmp_zarr_and_zip
import copy
import xarray as xr
from xclim.core.calendar import convert_calendar, get_calendar
from xscen.utils import minimum_calendar, stack_drop_nans
from xclim import sdba
import shutil as sh
import xclim as xc

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    client=dask_cluster(snakemake.params)

    ref_source = CONFIG['extraction']['ref_source']
    region_dict = CONFIG['custom']['regions'][snakemake.wildcards.region_name]

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
        {d: CONFIG['custom']['working_chunks'][d] for d in ds_ref.dims})

    # stack
    if CONFIG['custom']['stack_drop_nans']:


        variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                'variables_and_freqs'].keys())
        ds_ref = stack_drop_nans(
            ds_ref,
            ds_ref[variables[0]].isel(time=130, drop=True).notnull().compute(),
        )
    ds_ref = ds_ref.chunk({d: CONFIG['custom']['working_chunks'][d] for d in ds_ref.dims})
    ds_ref.attrs['cat:calendar'] = 'default'

    tmp_zarr_and_zip(ds_ref,snakemake.output.default)




    # noleap
    ds_refnl = convert_calendar(ds_ref, "noleap")
    ds_refnl.attrs['cat:calendar'] = 'noleap'
    tmp_zarr_and_zip(ds_refnl, snakemake.output.noleap)

    # 360_day

    ds_ref3 = convert_calendar(ds_ref, "360_day", align_on="year")
    ds_ref3.attrs['cat:calendar'] = '360_day'
    tmp_zarr_and_zip(ds_ref3, snakemake.output.day360)





    # # diagnostics
    # #

    # #extract on QC
    # ds_ref = xs.extract_dataset(catalog=dc,
    #                             region=CONFIG['custom']['qc_region'],
    #                             **CONFIG['extraction']['reference']['extract_dataset']
    #                             )['D']

    # #standardize units
    # ds_ref = xs.clean_up(ds_ref, **CONFIG['extraction']['reference']['clean_up'])
    # ds_ref['pr'] = xc.core.units.convert_units_to(ds_ref['pr'],
    #                                                 'kg m-2 s-1',
    #                                                 context='hydro')

    # dref_ref = ds_ref.drop_vars('dtr')


    # dref_ref = dref_ref.chunk(CONFIG['custom']['ref_chunk'])

    # # diagnostics
    # ds_ref_prop, _ = xs.properties_and_measures(
    #     ds=dref_ref,
    #     **CONFIG['extraction']['reference'][
    #         'properties_and_measures']
    # )

    # ds_ref_prop = ds_ref_prop.chunk(**CONFIG['custom']['ref_prop_chunk'])

    # tmp_zarr_and_zip(ds_ref_prop, snakemake.output.diag_ref_prop)

