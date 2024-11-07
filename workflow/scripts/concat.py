
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
    params=snakemake.params
    params.pop('sim_id_slash',None)

    client=dask_cluster(params)


    list_dsR = []
    for files in range(len(snakemake.input.final)):
        dsR = xr.open_zarr(snakemake.input.final[files], decode_timedelta=False)
        dsR.lat.encoding.pop('chunks', None)
        dsR.lon.encoding.pop('chunks', None)
        list_dsR.append(dsR)

    if 'rlat' in dsR:
        dsC = xr.concat(list_dsR, 'rlat')
    else:
        dsC = xr.concat(list_dsR, 'lat')

    dsC.attrs['cat:domain'] = CONFIG['custom']['amno_region']['name']
    dsC.attrs['cat:processing_level']= 'final'
    dsC.attrs.pop('intake_esm_dataset_key')
    dsC.attrs.pop('cat:path')

    dsC = dsC.chunk(
        xs.utils.translate_time_chunk(
            {'time': '4year'},
            xc.core.calendar.get_calendar(dsC),
            dsC.time.size)| CONFIG['custom']['final_chunks']
                               )
    
    # f_path = snakemake.output[0]
    # s_path=f"{os.environ['SLURM_TMPDIR']}/{f_path.name[:-4]}"

    # xs.save_to_zarr(
    #     ds=dsC,
    #     filename=s_path,
    #     )
    
    # zip_directory(s_path, f_path)
    for var in dsC.data_vars:
        if snakemake.output[var] == xs.catutils.build_path(dsC[[var]], root=COMFIG['paths']['finaldir'])+'.zarr.zip':
            tmp_zarr_and_zip(dsC[[var]],snakemake.output[var])
        else:
            raise ValueError(f"Output path for {var} is not as expected. 
                             from build_path: {xs.catutils.build_path(dsC[[var]], root=COMFIG['paths']['finaldir'])}.
                               from snakemake: {snakemake.output[var]}")