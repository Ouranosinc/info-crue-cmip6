""" compute indicators for partition figures"""
import os
from pathlib import Path
if 'ESMFMKFILE' not in os.environ:
    os.environ['ESMFMKFILE'] = str(Path(os.__file__).parent.parent / 'esmf.mk')
import xscen as xs
from dask.distributed import Client
from dask import config as dskconf
from xscen import CONFIG
import logging
path = '../paths_l.yml'
config = '../config-MBCn-RDRS.yml'
xs.load_config(path, config, verbose=(__name__ == '__main__'), reset=True)
logger = logging.getLogger('xscen')

if __name__ == '__main__':
    daskkws = CONFIG['dask'].get('client', {})
    dskconf.set(**{k: v for k, v in CONFIG['dask'].items() if k != 'client'})

    cat_extra = xs.ProjectCatalog(f"../{CONFIG['paths']['project_catalog']}")
    cat_e5l = xs.ProjectCatalog(f"../{CONFIG['paths']['project_catalog_e5l']}")

    with Client(n_workers=8, threads_per_worker=5, memory_limit="6GB",**daskkws):

        for pcat in [cat_extra, cat_e5l]:
            ds_dict = pcat.search(processing_level="final",
                                  id='CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp245_r2i1p1f1_global', #TODO: remove
                                  domain=['QC'],
                                  #'QC-RDRS','QC-EMDNA']
                                ).to_dataset_dict()

            for ds in ds_dict.values():
                if not cat_extra.exists_in_cat(id=ds.attrs['cat:id'],
                                          processing_level='indicators',
                                          domain=ds.attrs['cat:domain']):
                    _, ds_ind = xs.compute_indicators(
                        ds=ds,
                        indicators="indicators-partition.yml",
                    ).popitem()

                    xs.save_and_update(ds=ds_ind.chunk({'time': -1}),
                                       pcat=cat_extra,
                                       path=CONFIG['paths']['indicators_wd'])

