
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

    
    sh.copytree(snakemake.input.sim,f"{os.environ['SLURM_TMPDIR']}/dsim.zarr" )
    dsim= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dsim.zarr",decode_timedelta=False)
    print(dsim)
    print(dsim)
    
    # because we took regridded from other domain
    dsim.attrs['cat:domain'] = snakemake.wildcards.region_name

    # choose right calendar and convert
    refcal = minimum_calendar(get_calendar(dsim),
                            CONFIG['custom']['maximal_calendar'])
    dsim = convert_calendar(dsim, refcal,
                            align_on=CONFIG['custom']['align_on'])

    # load ref ds
    unzip_directory(snakemake.input[f'ref_{refcal}'],f"{os.environ['SLURM_TMPDIR']}/dref.zarr")
    dref= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dref.zarr",decode_timedelta=False)
    dref = convert_calendar(dref, refcal,
                            align_on=CONFIG['custom']['align_on'])
    


    # choose right ref period for hist
    dhist = dsim.sel(time=slice(*map(str, CONFIG['custom']['ref_period'])))

    dref,  dhist = (sdba.stack_variables(da) for da in
                        (dref, dhist))
    print('dref')
    print(dref)
    print('dhist')
    print(dhist)                   


    # create group
    group = CONFIG['biasadjust_mbcn'].get('group')
    if isinstance(group, dict):
        group = sdba.Grouper.from_kwargs(**group)["group"]
    elif isinstance(group, str):
        group = sdba.Grouper(group)
    
    # train
 
    dtrain = sdba.MBCn.train(
        ref=dref,
        hist=dhist,
        base_kws=dict(group=group, nquantiles=50, ), 
        **CONFIG['biasadjust_mbcn']['train']
    ).ds

    # attrs
    dtrain.attrs.update(dhist.attrs)
    dtrain.attrs['cat:processing_level'] = f"training_mbcn"


    tmp_zarr_and_zip(dtrain,snakemake.output[0])
