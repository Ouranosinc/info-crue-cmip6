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






st.set_page_config(layout="wide")
st.title('Diagnostiques de Info-Crue CMIP6')

#useCat=True
useCat= st.checkbox("use catalog (only for local version)")

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
    bias_sim = pcat.search(id=option_id, processing_level='diag_sim_meas', domain=option_domain).to_dataset_dict().popitem()[1]
    bias_scen = pcat.search(id=option_id, processing_level='diag_scen_meas', domain=option_domain).to_dataset_dict().popitem()[1]

    # load hmap
    if 'cat/domain' in scen.attrs:
        path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat/domain'],
                                                                sim_id=scen.attrs['cat/id'],
                                                                step='hmap'))
    else:
        path_diag = Path(
            CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat:domain'],
                                                  sim_id=scen.attrs['cat:id'],
                                                  step='hmap'))
    # replace .zarr by .npy
    path_diag = path_diag.with_suffix('.npy')
    hmap = np.load(path_diag)
else:

    #option_id = st.selectbox('id',[x[30:-5] for x in glob.glob('dashboard_data/diag_scen_bias_*')])
    ids = [x[30:-5] for x in glob.glob('dashboard_data/diag_scen_meas_*')]
    models = sorted(set([y.split('_')[3] for y in ids ]))
    exps = sorted(set([y.split('_')[4] for y in ids]))
    option_model = cols[0].selectbox('Models',models)
    option_ssp = cols[1].selectbox('Experiments',exps)

    option_id = [x for x in ids if option_model in x and option_ssp in x ][0]


    ref = xr.open_zarr(f'dashboard_data/diag_ref_ECMWF_ERA5-Land_NAM_qc.zarr')
    sim = xr.open_zarr(f'dashboard_data/diag_sim_{option_id}.zarr')
    scen = xr.open_zarr(f'dashboard_data/diag_scen_{option_id}.zarr')
    bias_sim = xr.open_zarr(f'dashboard_data/diag_sim_meas_{option_id}.zarr')
    bias_scen = xr.open_zarr(f'dashboard_data/diag_scen_meas_{option_id}.zarr')
    hmap = np.load(f'dashboard_data/diag_hmap_{option_id}.npy')

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
plt.title('Normalised mean measure of properties')
fig_hmap.tight_layout()

#percentage of grid point that improved
# percent_better=[]
# for var in sorted(bias_scen.data_vars):
#     if bias_sim[var].attrs['measure']=='ratio':
#         diff_bias = abs(bias_sim[var]-1) - abs(bias_scen[var]-1)
#     else:
#         diff_bias = abs(bias_sim[var]) - abs(bias_scen[var])
#     diff_bias=diff_bias.values.ravel()
#     diff_bias= diff_bias[~ np.isnan(diff_bias)]
#
#     total= bias_scen[var].values.ravel()
#     total = total[~ np.isnan(total)]
#
#     improved = diff_bias>=0
#     percent_better.append( np.sum(improved)/len(total)) # we count nan
#
# percent_better=np.reshape(np.array(percent_better), (1, len(bias_scen.data_vars)))
#
# fig_per, ax = plt.subplots(figsize=(1 * len(dict_prop), 1))
# cmap=plt.cm.RdYlGn
# norm = colors.BoundaryNorm(np.linspace(0,1,100), cmap.N)
# im = ax.imshow(percent_better, cmap=cmap, norm=norm)
# ax.set_xticks(ticks=np.arange(len(dict_prop)), labels=dict_prop, rotation=45,
#               ha='right')
# ax.set_yticks(ticks=np.arange(1), labels=[''])
#
# divider = make_axes_locatable(ax)
# cax = divider.new_vertical(size='15%', pad=0.4)
# fig_per.add_axes(cax)
# cbar = fig_per.colorbar(im, cax=cax, ticks=[0, 0.1, 0.2, 0.3, 0.4,0.5, 0.6, 0.7,0.8,0.9, 1], orientation='horizontal')
# plt.title('Percentage of grid cells that improved or stayed the same')
# fig_per.tight_layout()


cols2=st.columns(2)
# choose properties
option_var = cols2[0].selectbox('Input Variables',[ 'Maximum Temperature','Minimum Temperature', 'Precipitation'])
varlong2short = {'Minimum Temperature':'tasmin', 'Maximum Temperature':'tasmax', 'Precipitation':'pr' }

props_of_var= [x for x in scen.data_vars if f'(da={varlong2short[option_var]}' in scen[x].attrs['history'] ]

def show_long_name(name):
    return f"{scen[name].attrs['long_name']} ({name})"

option_prop = cols2[1].selectbox('Properties of the variable',sorted(props_of_var), format_func = show_long_name)
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
col2.write(hv.render(prop_sim.hvplot(width=w, height=h, title=f'SIM', cmap=cmap, clim=(mini_prop,maxi_prop)).opts(colorbar=False)))
col2.write(hv.render(prop_scen.hvplot(width=w, height=h, title=f'SCEN', cmap=cmap, clim=(mini_prop,maxi_prop)).opts(colorbar=False)))
col3.write(hv.render(bias_sim_prop.hvplot(width=wb, height=hb, title=f'SIM {measure_name}', cmap=cmap_bias,clim=(mini, maxi))))
col3.write(hv.render(bias_scen_prop.hvplot(width=wb, height=hb, title=f'SCEN {measure_name}', cmap=cmap_bias,clim=(mini,maxi))))


# test panel
# https://github.com/holoviz/panel/issues/1074
# https://github.com/streamlit/streamlit/issues/927
# import panel as pn
# import hvplot.xarray
#
# hvplot_plot = scen['mean-pr'].hvplot()
# hvplot_pane = pn.pane.HoloViews(hvplot_plot, name="Holoviews Plot")
# tabs = pn.Tabs(hvplot_pane)
# st.bokeh_chart(tabs.get_root())
#
# def plot_var(variable):
#     return scen[variable].hvplot()
#
# pan = pn.interact(plot_var, variable=list(scen.data_vars))
# st.write(hv.render(pan).get_root(), backend='bokeh')




st.write(fig_hmap)
#st.write(fig_per)
