import xscen as xs
import glob
import xarray as xr

if __name__ == '__main__':

    pcat= xs.ProjectCatalog('/project/ctb-frigon/julavoie/info-crue-cmip6/info-crue-6.json')

    for f in glob.glob('/project/ctb-frigon/julavoie/info-crue-cmip6/final_regions/*/*.zip'):
        print(f)
        ds=xr.open_zarr(f)
        pcat.update_from_ds(ds,f, info_dict={'processing_level': 'final','format': 'zarr'})
    for f in glob.glob('/project/ctb-frigon/julavoie/info-crue-cmip6/diagnostics/*/*.zip'):
        print(f)
        ds=xr.open_zarr(f)
        pcat.update_from_ds(ds,f, info_dict={'format': 'zarr'})

    for f in glob.glob('/scratch/julavoie/info-crue-cmip6_workdir/*/*'):
        print(f)
        ds=xr.open_zarr(f)
        pcat.update_from_ds(ds,f, info_dict={'format': 'zarr'})
    