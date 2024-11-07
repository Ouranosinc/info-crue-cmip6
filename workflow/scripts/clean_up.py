
import os
import xscen as xs
from xscen import CONFIG
from workflow.scripts.utils import dask_cluster, zip_directory, unzip_directory
import copy
import xarray as xr
from xclim.core.calendar import convert_calendar, get_calendar
from xscen.utils import minimum_calendar, stack_drop_nans
from xclim import sdba
import shutil as sh
from xscen.xclim_modules import conversions


xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    client=dask_cluster(snakemake.params)


    all_per=[xr.open_zarr(f,decode_timedelta=False) for f in snakemake.input]

    ds=xr.concat(all_per,dim='time')
    ds.attrs['cat:processing_level'] = f'biasadjusted'

    ds = ds.assign(tasmin=conversions.tasmin_from_dtr(dtr=ds.dtr, tasmax=ds.tasmax))
    ds = ds.drop_vars('dtr')    

    print(ds)




    ds = xs.clean_up(ds = ds.chunk({'time':-1}),
                    **CONFIG['clean_up']['xscen_clean_up'])

    clean_path=f"{os.environ['SLURM_TMPDIR']}/{sim_id}_{region_name}_cleaned.zarr"
    xs.save_to_zarr(ds, clean_path)


    xs.io.rechunk(path_in=clean_path,
        path_out=tmp_path(snakemake.output[0]),
        chunks_over_dim=CONFIG['custom']['final_zarr_chunks'],
        **CONFIG['rechunk'],
        overwrite=True)
    

    zip_directory(tmp_path(snakemake.output[0]),snakemake.output[0])
