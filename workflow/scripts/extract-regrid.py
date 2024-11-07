import os
import xscen as xs
from xscen import CONFIG
from workflow.scripts.utils import dask_cluster
import copy
import xarray as xr

xs.load_config("config/config.yml","config/paths.yml")

if __name__ == '__main__':

    client=dask_cluster(snakemake.params)

    args=copy.deepcopy(CONFIG['extraction']['simulation']['search_data_catalogs'])
    args['other_search_criteria'] = {'id': snakemake.wildcards.sim_id +'_global'}
    # search cat
    cat_sim_id = xs.search_data_catalogs(**args,)

    # extract
    dc_id = cat_sim_id.popitem()[1]
    # buffer is need to take a bit larger than actual domain, to avoid weird effect at the edge
    # domain will be cut to the right shape during the regrid
    region_dict=CONFIG['custom']['regions'][snakemake.wildcards.region_name]
    region_dict['tile_buffer']=5
    ds_sim = xs.extract_dataset(catalog=dc_id,
                                region=region_dict,
                                **CONFIG['extraction']['simulation']['extract_dataset'],
                                )['D']
    ds_sim['time'] = ds_sim.time.dt.floor('D') # probably this wont be need when data is cleaned

    # need lat and lon -1 for the regrid
    ds_sim = ds_sim.chunk(CONFIG['custom']['sim_chunks'])


    ds_input = ds_sim

    ds_target = xr.open_zarr(snakemake.input.noleap, decode_timedelta=False)


    ds_regrid = xs.regrid_dataset(
        ds=ds_input,
        ds_grid=ds_target,
        weights_location= f"{os.environ['SLURM_TMPDIR']}/weights/",
        **CONFIG['regrid']['regrid_dataset']
    )

    # chunk time dim
    ds_regrid = ds_regrid.chunk({d: CONFIG['custom']['working_chunks'][d] for d in ds_regrid.dims})

    xs.save_to_zarr(ds_regrid, str(snakemake.output[0]))