
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

    
    sh.copytree(snakemake.input.sim,f"{os.environ['SLURM_TMPDIR']}/dsim.zarr" )
    dsim= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dsim.zarr",decode_timedelta=False)
    # because we took regridded from other domain
    dsim.attrs['cat:domain'] = snakemake.wildcards.region_name

    # load ref ds
        # choose right calendar and convert
    refcal = minimum_calendar(get_calendar(dsim),CONFIG['custom']['maximal_calendar'])
    unzip_directory(snakemake.input[f'ref_{refcal}'],f"{os.environ['SLURM_TMPDIR']}/dref.zarr")
    dref= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dref.zarr",decode_timedelta=False)


#   train:
#     n_iter: 20
#     adj_kws:
#       interp: nearest
#       extrapolation: constant
#       n_escore: -1

    dtrain=xs.train(dref,
                     dsim,
                     var=CONFIG['biasadjust_mbcn']['variables'],
                     method='MBCn',
                     period=CONFIG['custom']['ref_period'],
                     group=CONFIG['biasadjust_mbcn']['group'],
                     xsdba_train_args = {'base_kws': {'nquantiles': 50},}| CONFIG['biasadjust_mbcn']['train'],
                     )


    tmp_zarr_and_zip(dtrain,snakemake.output[0])
