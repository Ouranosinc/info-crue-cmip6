
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
import xclim as xc

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':
    params=snakemake.params

    if 'sim_id_slash' in params:
        params.pop('sim_id_slash')

    client=dask_cluster(params)


    list_dsR = []
    for file in snakemake.input:
        dsR = xr.open_zarr(file, decode_timedelta=False)
        dsR.lat.encoding.pop('chunks', None)
        dsR.lon.encoding.pop('chunks', None)
        list_dsR.append(dsR)

    if 'rlat' in dsR:
        dsC = xr.concat(list_dsR, 'rlat')
    else:
        dsC = xr.concat(list_dsR, 'lat')

    dsC.attrs['cat:domain'] = 'QC'
    dsC.attrs.pop('cat:path', None)

    for var in dsC.data_vars:
        out=dsC[[var]]
        out[var].encoding={}
        out = out.chunk(
        xs.utils.translate_time_chunk(
            {'time': '4year'},
            xc.core.calendar.get_calendar(out),
            out.time.size)| CONFIG['custom']['concat_chunks']
                               )

        tmp_zarr_and_zip(out, snakemake.output[var])
