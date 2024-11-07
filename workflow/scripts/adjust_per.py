
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
from pathlib import Path

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    client=dask_cluster(snakemake.params)

    
    #dsim_disk = xr.open_zarr(snakemake.input.sim, decode_timedelta=False)
    #xs.save_to_zarr(ds=dsim_disk, filename=f"{os.environ['SLURM_TMPDIR']}/dsim.zarr")
    sh.copytree(snakemake.input.sim,f"{os.environ['SLURM_TMPDIR']}/dsim.zarr" )
    dsim= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dsim.zarr",decode_timedelta=False)
    print(dsim)
    
    # because we took regridded from other domain
    region_name = snakemake.wildcards.region_name
    sim_id = snakemake.wildcards.sim_id

    # choose right calendar and convert
    refcal = minimum_calendar(get_calendar(dsim),
                            CONFIG['custom']['maximal_calendar'])
    dsim = convert_calendar(dsim, refcal,
                            align_on=CONFIG['custom']['align_on'])

    # load ref ds
    #dref_disk = xr.open_zarr(snakemake.input.ref, decode_timedelta=False)
    #xs.save_to_zarr(ds=dref_disk, filename=f"{os.environ['SLURM_TMPDIR']}/dref.zarr")
    unzip_directory(snakemake.input[f'ref_{refcal}'],f"{os.environ['SLURM_TMPDIR']}/dref.zarr")
    dref= xr.open_zarr(f"{os.environ['SLURM_TMPDIR']}/dref.zarr",decode_timedelta=False)
    print(refcal)
    dref = convert_calendar(dref, refcal,
                            align_on=CONFIG['custom']['align_on'])
    


    # choose right ref period for hist
    dhist = dsim.sel(time=slice(*map(str, CONFIG['custom']['ref_period'])))

    dref,  dhist = (sdba.stack_variables(da) for da in
                        (dref, dhist))

    print(dref)
    print(dhist)                   


    # create group
    group = CONFIG['biasadjust_mbcn'].get('group')
    if isinstance(group, dict):
        group = sdba.Grouper.from_kwargs(**group)["group"]
    elif isinstance(group, str):
        group = sdba.Grouper(group)
    


    f_path = Path(snakemake.input.train)
    s_path=f"{os.environ['SLURM_TMPDIR']}/{f_path.name[:-4]}"
    unzip_directory(f_path, s_path)

    dtrain= xr.open_zarr(s_path,
                        decode_timedelta=False, 
                        drop_variables=['escores'],
                        )
    print(dtrain)
    ADJ = sdba.adjustment.TrainAdjust.from_dataset(dtrain)

    per=[snakemake.wildcards.period.split('-')[0],snakemake.wildcards.period.split('-')[1]]
    dsim_cur=dsim.sel(time=slice(*per))
    dsim_cur = sdba.stack_variables(dsim_cur)
    
    print(dsim_cur)
    out = ADJ.adjust(
        sim=dsim_cur,
        ref=dref,
        hist=dhist,
        base=sdba.QuantileDeltaMapping,
        **CONFIG['biasadjust_mbcn']['adjust'],
    )

    out = sdba.unstack_variables(out)

    # attrs
    out.attrs.update(dsim.attrs)

    tmp_path=f"{os.environ['SLURM_TMPDIR']}/{sim_id}_{region_name}_{snakemake.wildcards.period}_biasadjusted.zarr"
    save_path=snakemake.output[0]
    xs.save_to_zarr(ds=out, filename=tmp_path)
    sh.move(tmp_path, save_path)
