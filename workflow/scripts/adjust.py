
import os
import xscen as xs
from xscen import CONFIG
from workflow.scripts.utils import dask_cluster, zip_directory, unzip_directory, tmp_zarr_and_zip

import copy
import xarray as xr
from xclim.core.calendar import  get_calendar #convert_calendar,
from xscen.utils import minimum_calendar, stack_drop_nans
import shutil as sh
from pathlib import Path
import xclim as xc

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    dsim= xr.open_zarr(snakemake.input.sim,decode_timedelta=False).load()

    refcal = minimum_calendar(get_calendar(dsim),CONFIG['custom']['maximal_calendar'])
    dref= xr.open_zarr(snakemake.input[f'ref_{refcal}'],decode_timedelta=False).load()
   
    dtrain= xr.open_zarr(snakemake.input.train,
                        decode_timedelta=False, 
                        drop_variables=['escores'],
                        ).load()

    ##FIXME: when xscen/xsda can handle units correctly
    dref['pr']=xc.units.convert_units_to(dref['pr'], 'kg m^-2 s^-1', context='hydro')
    dsim['pr']=xc.units.convert_units_to(dsim['pr'], 'kg m^-2 s^-1', context='hydro')

    out = xs.adjust(
        dtrain = dtrain, 
        dsim = dsim,
        dref = dref,
        **CONFIG['biasadjust_mbcn']['adjust'],
    )

    tmp_zarr_and_zip(out,snakemake.output[0])
