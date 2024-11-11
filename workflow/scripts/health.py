
import os
import xscen as xs
from xscen import CONFIG
from workflow.scripts.utils import dask_cluster, zip_directory, unzip_directory, tmp_zarr_and_zip
import copy
import xarray as xr
from xclim.core.calendar import convert_calendar, get_calendar
from xscen.utils import minimum_calendar, stack_drop_nans
from xclim import sdba
import shutil as sh

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    client=dask_cluster(snakemake.params)


    ds_input = xr.open_mfdataset([snakemake.input.pr, snakemake.input.tasmax, snakemake.input.tasmin],
                                  engine='zarr',decode_timedelta=False)
    hc = xs.diagnostics.health_checks(
        ds=ds_input,
        **CONFIG['diagnostics']['health_checks'])

    hc.attrs.update(ds_input.attrs)
    hc.attrs['cat:processing_level'] = 'health_checks'
    xs.save_to_zarr(ds=hc, filename=snakemake.output[0])
