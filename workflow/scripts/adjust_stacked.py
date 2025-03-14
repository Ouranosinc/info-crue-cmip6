
import os
import xscen as xs
print(xs.__version__)
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
    dsim= dsim[CONFIG['biasadjust_mbcn']['variables']]

    # because we took regridded from other domain
    region_name = snakemake.wildcards.region_name
    sim_id = snakemake.wildcards.sim_id

    # choose right calendar and convert
    refcal = minimum_calendar(get_calendar(dsim),
                            CONFIG['custom']['maximal_calendar'])
    dsim = dsim.convert_calendar(refcal,align_on=CONFIG['custom']['align_on'])
    #dsim = convert_calendar(dsim, refcal, align_on=CONFIG['custom']['align_on'])

    # load ref ds
    dref= xr.open_zarr(snakemake.input[f'ref_{refcal}'],decode_timedelta=False).load()
    # dref = convert_calendar(dref, refcal,
    #                         align_on=CONFIG['custom']['align_on'])
    # dref= dref[CONFIG['biasadjust_mbcn']['variables']]


    # # choose right ref period for hist
    # dhist = dsim.sel(time=slice(*map(str, CONFIG['custom']['ref_period'])))

    # dref,  dhist = (sdba.stack_variables(da) for da in
    #                     (dref, dhist))
                
    


    # f_path = Path(snakemake.input.train)

    dtrain= xr.open_zarr(snakemake.input.train,
                        decode_timedelta=False, 
                        drop_variables=['escores'],
                        ).load()

    params_stacked = {
        "periods" : ["1950", "2099"], # juste mettre ta période complet d'intérêt, un multiple de périodes de 30 ans, 30*5 = 150 ans
        "xsdba_adjust_args" : {"period_dim":"period"}| CONFIG['biasadjust_mbcn']['adjust'] # c'est cette option qui va dire à xscen de faire le stacking
    }

    #TODO: include in xscen
    dref['pr']=xc.units.convert_units_to(dref['pr'], 'kg m^-2 s^-1', context='hydro')
    dsim['pr']=xc.units.convert_units_to(dsim['pr'], 'kg m^-2 s^-1', context='hydro')

    print(dref.pr.attrs['units'])
    print(dsim.pr.attrs['units'])
    print(dtrain.multivar.attrs['_units'])


    out = xs.adjust(
        dtrain = dtrain, 
        dsim = dsim,
        dref = dref,
        **params_stacked
    )


    # ADJ = sdba.adjustment.TrainAdjust.from_dataset(dtrain)

    
    # out = ADJ.adjust(
    #     sim=dsim,
    #     ref=dref,
    #     hist=dhist,
    #     base=sdba.QuantileDeltaMapping,
    #     **CONFIG['biasadjust_mbcn']['adjust'],
    # )

    # attrs
    # out.attrs.update(dsim.attrs)
    # for a in CONFIG['biasadjust_mbcn']['attrs']:
    #     out.attrs[f"cat:"+a] = CONFIG['biasadjust_mbcn']['attrs'][a]


    # tmp_path=f"{os.environ['SLURM_TMPDIR']}/{sim_id}_{region_name}_biasadjusted.zarr"
    # save_path=snakemake.output[0]
    # xs.save_to_zarr(ds=out, filename=tmp_path)
    # sh.move(tmp_path, save_path)

    tmp_zarr_and_zip(out,snakemake.output[0])
