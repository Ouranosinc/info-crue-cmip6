
import os
import xscen as xs
from xscen import CONFIG
from workflow.scripts.utils import dask_cluster, zip_directory, tmp_zarr_and_zip
import copy
import xarray as xr
from xscen.utils import stack_drop_nans
from xclim import sdba
import shutil as sh
import xclim as xc

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    client=dask_cluster(snakemake.params)

    # search
    cat_ref = xs.search_data_catalogs(**CONFIG['extraction']['reference']['search_data_catalogs'])

    # extract
    dc = cat_ref.popitem()[1]
    ds_ref = xs.extract_dataset(catalog=dc,
                                region=CONFIG['custom']['full_region'],
                                **CONFIG['extraction']['reference']['extract_dataset']
                                )['D']
    

    #standardize units
    ds_ref = xs.clean_up(ds_ref, **CONFIG['extraction']['reference']['clean_up'])
    #FIXME: when xscen/xsda can handle units correctly
    ds_ref['pr'] = xc.core.units.convert_units_to(ds_ref['pr'],
                                                  'kg m-2 s-1',
                                                  context='hydro')


    # stack
    if CONFIG['custom']['stack_drop_nans']:

        variables = list(CONFIG['extraction']['reference']['search_data_catalogs'][
                                'variables_and_freqs'].keys())
        ds_ref = stack_drop_nans(
            ds_ref,
            ds_ref[variables[0]].isel(time=130, drop=True).notnull().compute(),
        )

    # cut region
    n=CONFIG['subregions']['n']
    r=int(snakemake.wildcards.region_name.replace(f"{CONFIG['subregions']['code']}-",''))
    ds_ref=ds_ref.sel(loc=slice(n*r, n*(r+1)))

    ds_ref = ds_ref.chunk({d: CONFIG['chunks']['working'][d] for d in ds_ref.dims})
    ds_ref.attrs['cat:calendar'] = 'default'

    tmp_zarr_and_zip(ds_ref,snakemake.output.default)

    # noleap
    ds_refnl =ds_ref.convert_calendar('noleap')
    ds_refnl.attrs['cat:calendar'] = 'noleap'
    tmp_zarr_and_zip(ds_refnl, snakemake.output.noleap)

    # 360_day
    ds_ref3 = ds_ref.convert_calendar('360_day', align_on="year")
    ds_ref3.attrs['cat:calendar'] = '360_day'
    tmp_zarr_and_zip(ds_ref3, snakemake.output.day360)