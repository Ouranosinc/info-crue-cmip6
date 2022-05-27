import streamlit as st
import holoviews as hv
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import hvplot.xarray
import panel as pn
from mpl_toolkits.axes_grid1 import make_axes_locatable


from xclim import sdba

from xscen.config import CONFIG, load_config
from xscen.catalog import ProjectCatalog

load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])


st.title('Testing')

st.write('Proving that hvplot works')
choose_data = st.radio(
     'data type',
    ('sim', 'scen','ref'))

ds = pcat.search(processing_level=f'diag_{choose_data}').to_dataset_dict().popitem()[1]

vars= ds.data_vars
option = st.selectbox('Properties',vars)

st.write(hv.render(ds[option].hvplot()))

st.write('show diagnotics, eventually show this with hvplot and make pretty')
# choose id
option_id = st.selectbox('id',['CMIP6_ScenarioMIP_NOAA-GFDL_GFDL-ESM4_EXPERIMENT_r1i1p1f1_global'])

#load all properties from ref, sim, scen
ref = pcat.search( processing_level='diag_ref').to_dataset_dict().popitem()[1]
sim = pcat.search(id= option_id, processing_level='diag_sim').to_dataset_dict().popitem()[1]
scen = pcat.search(id= option_id, processing_level='diag_scen').to_dataset_dict().popitem()[1]

#choose property to show
option_var = st.selectbox('Properties',vars)
prop_sim = sim[option_var]
prop_ref = ref[option_var]
prop_scen = scen[option_var]
bias_scen_prop = pcat.search(id=option_id, processing_level='diag_scen_prop').to_dataset_dict().popitem()[1][option_var]
bias_sim_prop = pcat.search(id=option_id, processing_level='diag_sim_prop').to_dataset_dict().popitem()[1][option_var]

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


#load hmap
path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat/domain'],
                                                       sim_id=scen.attrs['cat/id'],
                                                       step='hmap'))
# replace zarr by npy
path_diag = path_diag.with_suffix('.npy')
hmap = np.load(path_diag)

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
