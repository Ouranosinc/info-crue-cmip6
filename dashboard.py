import streamlit as st
import holoviews as hv
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

useCat=False



st.set_page_config(layout="wide")
st.title('Diagnostiques de Info-Crue CMIP6')

if useCat:
    from xscen.config import CONFIG, load_config
    from xscen.catalog import ProjectCatalog
    load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
    pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

    # choose id
    option_id = st.selectbox('id',pcat.search(type='simulations').df.id.unique())

    #load all properties from ref, sim, scen
    ref = pcat.search( processing_level='diag_ref').to_dataset_dict().popitem()[1]
    sim = pcat.search(id= option_id, processing_level='diag_sim').to_dataset_dict().popitem()[1]
    scen = pcat.search(id= option_id, processing_level='diag_scen').to_dataset_dict().popitem()[1]
    #get bias
    bias_sim = pcat.search(id=option_id, processing_level='diag_sim_bias').to_dataset_dict().popitem()[1]
    bias_scen = pcat.search(id=option_id, processing_level='diag_scen_bias').to_dataset_dict().popitem()[1]

    # load hmap
    path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat/domain'],
                                                           sim_id=scen.attrs['cat/id'],
                                                           step='hmap'))
    # replace .zarr by .npy
    path_diag = path_diag.with_suffix('.npy')
    hmap = np.load(path_diag)
else:
    option_id = st.selectbox('id',[x[30:-5] for x in glob.glob('dashboard_data/diag_scen_bias_*')])
    ref=xr.open_dataset(f'dashboard_data/diag_ref_ERA_ecmwf_ERA5_era5-land_NAM_qc.zarr.zarr')
    sim = xr.open_dataset(f'dashboard_data/diag_sim_{option_id}.zarr')
    scen = xr.open_dataset(f'dashboard_data/diag_scen_{option_id}.zarr')
    bias_sim = xr.open_dataset(f'dashboard_data/diag_sim_bias_{option_id}.zarr')
    bias_scen = xr.open_dataset(f'dashboard_data/diag_scen_bias_{option_id}.zarr')


# choose properties
option_var = st.selectbox('Properties',scen.data_vars)
prop_sim = sim[option_var]
prop_ref = ref[option_var]
prop_scen = scen[option_var]
bias_scen_prop = bias_scen[option_var]
bias_sim_prop = bias_sim[option_var]

#colormap
maxi_prop = max(prop_ref.max().values, prop_scen.max().values, prop_sim.max().values)
mini_prop = min(prop_ref.min().values, prop_scen.min().values, prop_sim.min().values)
maxi_bias = max(abs(bias_scen_prop).max().values, abs(bias_sim_prop).max().values)
cmap='viridis_r' if prop_sim.attrs['standard_name']== 'precipitation_flux' else 'plasma'
cmap_bias ='BrBG' if prop_sim.attrs['standard_name']== 'precipitation_flux' else 'coolwarm'

long_name=prop_sim.attrs['long_name']

col1, col2, col3 = st.columns([2,1,1])
w, h = 350, 300
col1.write(hv.render(prop_ref.hvplot(title=f'REF\n{long_name}',width=650, height=600, cmap=cmap, clim=(mini_prop,maxi_prop))))
col2.write(hv.render(prop_sim.hvplot(width=w, height=h, title=f'SIM', cmap=cmap, clim=(mini_prop,maxi_prop))))
col3.write(hv.render(bias_sim_prop.hvplot(width=w, height=h, title=f'SIM BIAS', cmap=cmap_bias, clim=(-maxi_bias,maxi_bias))))
col2.write(hv.render(prop_scen.hvplot(width=w, height=h, title=f'SCEN', cmap=cmap, clim=(mini_prop,maxi_prop))))
col3.write(hv.render(bias_scen_prop.hvplot(width=w, height=h, title=f'SCEN BIAS', cmap=cmap_bias, clim=(-maxi_bias,maxi_bias))))





#plot hmap
dict_prop = sim.data_vars
labels_row = ['sim', 'scen']
fig_hmap, ax = plt.subplots(figsize=(1 * len(dict_prop), 1 * len(labels_row)))
im = ax.imshow(hmap, cmap='RdYlGn_r')
ax.set_xticks(ticks=np.arange(len(dict_prop)), labels=dict_prop.keys(), rotation=45,
              ha='right')
ax.set_yticks(ticks=np.arange(len(labels_row)), labels=labels_row)

divider = make_axes_locatable(ax)
cax = divider.new_vertical(size='15%', pad=0.4)
fig_hmap.add_axes(cax)
cbar = fig_hmap.colorbar(im, cax=cax, ticks=[0, 1], orientation='horizontal')
cbar.ax.set_xticklabels(['best', 'worst'])
plt.title('Normalised mean bias of properties')
fig_hmap.tight_layout()

st.write(fig_hmap)




#plot 5 maps
fig, axs = plt.subplots(2, 3, figsize=(15, 7))
maxi_prop = max(prop_ref.max().values, prop_scen.max().values, prop_sim.max().values)
mini_prop = min(prop_ref.min().values, prop_scen.min().values, prop_sim.min().values)
maxi_bias = max(abs(bias_scen_prop).max().values, abs(bias_sim_prop).max().values)

prop_ref.plot(ax=axs[0, 0], vmax=maxi_prop, vmin=mini_prop)
axs[0, 0].set_title('REF')
prop_scen.plot(ax=axs[0, 1], vmax=maxi_prop, vmin=mini_prop)
axs[0, 1].set_title('SCEN')
bias_scen_prop.plot(ax=axs[0, 2], vmax=maxi_bias, vmin=-maxi_bias, cmap='bwr')
axs[0, 2].set_title('bias scen')

prop_sim.plot(ax=axs[1, 1], vmax=maxi_prop, vmin=mini_prop)
axs[1, 1].set_title('SIM')
bias_sim_prop.plot(ax=axs[1, 2], vmax=maxi_bias, vmin=-maxi_bias, cmap='bwr')
axs[1, 2].set_title('bias sim')
fig.delaxes(axs[1][0])
fig.suptitle(option_var, fontsize=20)
fig.tight_layout()
st.write(fig)