
import os
import xscen as xs
from xscen import CONFIG
from workflow.scripts.utils import dask_cluster, zip_directory, unzip_directory, tmp_zarr_and_zip
import copy
import xarray as xr
from xclim.core.calendar import  get_calendar # convert_calendar
from xscen.utils import minimum_calendar, stack_drop_nans
import shutil as sh

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    client=dask_cluster(snakemake.params)

    
    unzip_directory(snakemake.input.sim,f"{os.environ['SLURM_TMPDIR']}/dsim.zarr" )
    dsim= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dsim.zarr",decode_timedelta=False)
    # because we took regridded from other domain
    #dsim.attrs['cat:domain'] = snakemake.wildcards.region_name # should not be needed if rest works

    # load ref ds
    refcal = minimum_calendar(get_calendar(dsim),CONFIG['custom']['maximal_calendar'])
    unzip_directory(snakemake.input[f'ref_{refcal}'],f"{os.environ['SLURM_TMPDIR']}/dref.zarr")
    dref= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dref.zarr",decode_timedelta=False)


    dtrain=xs.train(dref,dsim,**CONFIG['biasadjust_mbcn']['train'])


    tmp_zarr_and_zip(dtrain,snakemake.output[0])
