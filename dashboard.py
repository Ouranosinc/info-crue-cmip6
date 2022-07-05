# app is here: https://share.streamlit.io/ouranosinc/info-crue-cmip6/main/dashboard.py
import streamlit as st
import holoviews as hv
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import hvplot.xarray
from matplotlib import colors


useCat=False



st.set_page_config(layout="wide")
st.title('Diagnostiques de Info-Crue CMIP6')

cols = st.columns(2)

if useCat:
    from xscen.config import CONFIG, load_config
    from xscen.catalog import ProjectCatalog
    load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
    pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

    # choose id
    option_id = cols[0].selectbox('Id',pcat.search(type=['simulations', 'simulation']).df.id.unique())

    option_domain = cols[1].selectbox('Domain', pcat.search(type=['simulations', 'simulation']).df.domain.unique())

    #load all properties from ref, sim, scen
    ref = pcat.search( processing_level='diag_ref', domain=option_domain).to_dataset_dict().popitem()[1]
    sim = pcat.search(id= option_id, processing_level='diag_sim', domain=option_domain).to_dataset_dict().popitem()[1]
    scen = pcat.search(id= option_id, processing_level='diag_scen', domain=option_domain).to_dataset_dict().popitem()[1]
    #get bias
    bias_sim = pcat.search(id=option_id, processing_level='diag_sim_bias', domain=option_domain).to_dataset_dict().popitem()[1]
    bias_scen = pcat.search(id=option_id, processing_level='diag_scen_bias', domain=option_domain).to_dataset_dict().popitem()[1]

    # load hmap
    path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat/domain'],
                                                           sim_id=scen.attrs['cat/id'],
                                                           step='hmap'))
    # replace .zarr by .npy
    path_diag = path_diag.with_suffix('.npy')
    hmap = np.load(path_diag)
else:

    #option_id = st.selectbox('id',[x[30:-5] for x in glob.glob('dashboard_data/diag_scen_bias_*')])
    ids = [x[30:-5] for x in glob.glob('dashboard_data/diag_scen_bias_*')]
    models = [y.split('_')[3] for y in ids ]
    option_model = cols[0].selectbox('Models',models)
    option_ssp = cols[1].selectbox('Experiments',['ssp370'])

    option_id = [x for x in ids if option_model in x and option_ssp in x ][0]


    ref = xr.open_zarr(f'dashboard_data/diag_ref_ERA_ecmwf_ERA5_era5-land_NAM_qc.zarr')
    sim = xr.open_zarr(f'dashboard_data/diag_sim_{option_id}.zarr')
    scen = xr.open_zarr(f'dashboard_data/diag_scen_{option_id}.zarr')
    bias_sim = xr.open_zarr(f'dashboard_data/diag_sim_bias_{option_id}.zarr')
    bias_scen = xr.open_zarr(f'dashboard_data/diag_scen_bias_{option_id}.zarr')
    hmap = np.load(f'dashboard_data/diag_hmap_{option_id}.npy')


cols2=st.columns(2)
# choose properties
option_var = cols2[0].selectbox('Input Variables',['Minimum Temperature', 'Maximum Temperature', 'Precipitation'])
varlong2short = {'Minimum Temperature':'tasmin', 'Maximum Temperature':'tasmax', 'Precipitation':'pr' }

props_of_var= [x for x in scen.data_vars if varlong2short[option_var] in scen[x].attrs['history'] ]

def show_long_name(name):
    return f"{scen[name].attrs['long_name']} ({name})"

option_prop = cols2[1].selectbox('Properties of the variable',props_of_var, format_func = show_long_name)
prop_sim = sim[option_prop]
prop_ref = ref[option_prop]
prop_scen = scen[option_prop]
bias_scen_prop = bias_scen[option_prop]
bias_sim_prop = bias_sim[option_prop]

#colormap
maxi_prop = max(prop_ref.max().values, prop_scen.max().values, prop_sim.max().values)
mini_prop = min(prop_ref.min().values, prop_scen.min().values, prop_sim.min().values)
maxi_bias = max(abs(bias_scen_prop).max().values, abs(bias_sim_prop).max().values)
cmap='viridis_r' if option_var == 'Precipitation' else 'plasma'
cmap_bias ='BrBG' if option_var == 'Precipitation' else 'coolwarm'

long_name=prop_sim.attrs['long_name']

col1, col2, col3 = st.columns([6,3,4])
w, h = 300, 300
wb, hb = 400, 300

measure_name = bias_sim_prop.attrs['long_name']
# fix range of colorbar
if measure_name =='Ratio': #center around 1 for ratio
    maxi_rat = maxi_bias
    mini_rat = max(abs(bias_scen_prop).min().values, abs(bias_sim_prop).min().values)
    max_deviation_from_1 = max(abs(1-maxi_rat),abs(1-mini_rat))
    mini, maxi = 1-max_deviation_from_1, 1+max_deviation_from_1

else: # center around 0 for bias
    mini, maxi = -maxi_bias, maxi_bias

col1.write(hv.render(prop_ref.hvplot(title=f'REF\n{long_name}',width=600, height=616, cmap=cmap, clim=(mini_prop,maxi_prop))))
col2.write(hv.render(prop_scen.hvplot(width=w, height=h, title=f'SCEN', cmap=cmap, clim=(mini_prop,maxi_prop)).opts(colorbar=False)))
col2.write(hv.render(prop_sim.hvplot(width=w, height=h, title=f'SIM', cmap=cmap, clim=(mini_prop,maxi_prop)).opts(colorbar=False)))
col3.write(hv.render(bias_sim_prop.hvplot(width=wb, height=hb, title=f'SIM {measure_name}', cmap=cmap_bias,clim=(mini, maxi))))
col3.write(hv.render(bias_scen_prop.hvplot(width=wb, height=hb, title=f'SCEN {measure_name}', cmap=cmap_bias,clim=(mini,maxi))))


# TODO: fix hmap before putting it back
#plot hmap
dict_prop = sorted(sim.data_vars)
labels_row = ['sim', 'scen']
fig_hmap, ax = plt.subplots(figsize=(1 * len(dict_prop), 1 * len(labels_row)))
cmap=plt.cm.RdYlGn_r
norm = colors.BoundaryNorm(np.linspace(0,1,len(labels_row)+2), cmap.N)
im = ax.imshow(hmap, cmap=cmap, norm=norm)
ax.set_xticks(ticks=np.arange(len(dict_prop)), labels=dict_prop, rotation=45,
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
# fig, axs = plt.subplots(2, 3, figsize=(15, 7))
# maxi_prop = max(prop_ref.max().values, prop_scen.max().values, prop_sim.max().values)
# mini_prop = min(prop_ref.min().values, prop_scen.min().values, prop_sim.min().values)
# maxi_bias = max(abs(bias_scen_prop).max().values, abs(bias_sim_prop).max().values)
#
# prop_ref.plot(ax=axs[0, 0], vmax=maxi_prop, vmin=mini_prop)
# axs[0, 0].set_title('REF')
# prop_scen.plot(ax=axs[0, 1], vmax=maxi_prop, vmin=mini_prop)
# axs[0, 1].set_title('SCEN')
# bias_scen_prop.plot(ax=axs[0, 2], vmax=maxi_bias, vmin=-maxi_bias, cmap='bwr')
# axs[0, 2].set_title('bias scen')
#
# prop_sim.plot(ax=axs[1, 1], vmax=maxi_prop, vmin=mini_prop)
# axs[1, 1].set_title('SIM')
# bias_sim_prop.plot(ax=axs[1, 2], vmax=maxi_bias, vmin=-maxi_bias, cmap='bwr')
# axs[1, 2].set_title('bias sim')
# fig.delaxes(axs[1][0])
# fig.suptitle(option_prop, fontsize=20)
# fig.tight_layout()
# st.write(fig)