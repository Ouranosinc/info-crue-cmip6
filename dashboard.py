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

st.title('Info-Crue CMIP6')
useCat = st.checkbox("use catalog (only for local version)")

tab1, tab2 = st.tabs(["Diagnostiques", "Niveaux de réchauffement global"])
with tab1:
    #useCat=True
    cols = st.columns(2)

    #load data
    if useCat:
        from xscen.config import CONFIG, load_config
        from xscen.catalog import ProjectCatalog
        load_config('paths_neree.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
        pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

        # choose id
        option_id = cols[0].selectbox('Id',pcat.search(type=['simulations', 'simulation'], processing_level = 'diag-improved').df.id.unique())

        option_domain = cols[1].selectbox('Domain', pcat.search(type=['simulations', 'simulation']).df.domain.unique())

        #load all properties from ref, sim, scen
        ref = pcat.search( processing_level='diag-ref-prop', domain=option_domain).to_dask(xarray_open_kwargs={'decode_timedelta':False})
        sim = pcat.search(id= option_id, processing_level='diag-sim-prop', domain=option_domain).to_dask(xarray_open_kwargs={'decode_timedelta':False})
        scen = pcat.search(id= option_id, processing_level='diag-scen-prop', domain=option_domain).to_dask(xarray_open_kwargs={'decode_timedelta':False})
        #get bias
        bias_sim = pcat.search(id=option_id, processing_level='diag-sim-meas', domain=option_domain).to_dask(xarray_open_kwargs={'decode_timedelta':False})
        bias_scen = pcat.search(id=option_id, processing_level='diag-scen-meas', domain=option_domain).to_dask(xarray_open_kwargs={'decode_timedelta':False})

        #get measures summary
        hm = pcat.search( id=option_id,processing_level='diag-heatmap').to_dask()
        imp = pcat.search(id=option_id,processing_level='diag-improved').to_dask()


    else:

        #option_id = st.selectbox('id',[x[30:-5] for x in glob.glob('dashboard_data/diag_scen_bias_*')])
        ids = [x[30:-5] for x in glob.glob('dashboard_data/diag_scen_meas_*')]
        models = sorted(set([y.split('_')[3] for y in ids ]))
        exps = sorted(set([y.split('_')[4] for y in ids]))
        option_model = cols[0].selectbox('Models',models)
        option_ssp = cols[1].selectbox('Experiments',exps)

        option_id = [x for x in ids if option_model in x and option_ssp in x ][0]

        ref = xr.open_dataset(f'dashboard_data/diag_ref_ECMWF_ERA5-Land_NAM_qc.nc',
                              decode_timedelta=False)
        sim = xr.open_dataset(f'dashboard_data/diag-sim-prop_{option_id}.nc',
                              decode_timedelta=False)
        scen = xr.open_dataset(f'dashboard_data/diag-scen-prop_{option_id}.nc',
                               decode_timedelta=False)
        bias_sim = xr.open_dataset(f'dashboard_data/diag-sim-meas_{option_id}.nc',
                                   decode_timedelta=False)
        bias_scen = xr.open_dataset(f'dashboard_data/diagscen-meas_{option_id}.nc',
                                    decode_timedelta=False)

        hm = xr.open_zarr(f'dashboard_data/diag-heatmap_{option_id}.zarr',
                          decode_timedelta=False)
        imp = xr.open_zarr(f'dashboard_data/diag-improved_{option_id}.zarr',
                           decode_timedelta=False)

    cols2=st.columns(2)
    # choose properties
    option_var = cols2[0].selectbox('Input Variables',[ 'Maximum Temperature','Minimum Temperature', 'Precipitation'])
    varlong2short = {'Minimum Temperature':'tasmin', 'Maximum Temperature':'tasmax', 'Precipitation':'pr' }
    props_of_var= [x for x in scen.data_vars if f'{varlong2short[option_var]},' in scen[x].attrs['history'] ]


    def show_long_name(name):
        return f"{scen[name].attrs['long_name']} ({name})"

    option_prop = cols2[1].selectbox('Properties of the variable',sorted(props_of_var), format_func = show_long_name)
    prop_sim = sim[option_prop]
    prop_ref = ref[option_prop]
    prop_scen = scen[option_prop]
    bias_scen_prop = bias_scen[option_prop]
    bias_sim_prop = bias_sim[option_prop]

    if 'season' in prop_sim.coords:
        option_season =  cols2[1].selectbox('Season',prop_sim.season.values)
        prop_sim = prop_sim.sel(season=option_season)
        prop_ref = prop_ref.sel(season=option_season)
        prop_scen = prop_scen.sel(season=option_season)
        bias_scen_prop = bias_scen_prop.sel(season=option_season)
        bias_sim_prop = bias_sim_prop.sel(season=option_season)

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


    #plot the heat map
    fig_hmap, ax = plt.subplots(figsize=(7,3))
    cmap=plt.cm.RdYlGn_r
    norm = colors.BoundaryNorm(np.linspace(0,1,4), cmap.N)
    im = ax.imshow(hm.heatmap.values, cmap=cmap, norm=norm)
    ax.set_xticks(ticks = np.arange(len(hm.properties)), labels=hm.properties.values, rotation=45,ha='right')
    ax.set_yticks(ticks = np.arange(len(hm.datasets)), labels=[x.split('.')[2].split('-')[1] for x in hm.datasets.values])
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='15%', pad=0.4)
    fig_hmap.add_axes(cax)
    cbar = fig_hmap.colorbar(im, cax=cax, ticks=[0, 1], orientation='horizontal')
    cbar.ax.set_xticklabels(['best', 'worst'])
    plt.title('Normalised mean measure of properties')
    fig_hmap.tight_layout()

    #plot improved
    percent_better= imp.improved_grid_points.values
    percent_better=np.reshape(np.array(percent_better), (1, len(hm.properties)))
    fig_per, ax = plt.subplots(figsize=(7, 3))
    cmap=plt.cm.RdYlGn
    norm = colors.BoundaryNorm(np.linspace(0,1,100), cmap.N)
    im = ax.imshow(percent_better, cmap=cmap, norm=norm)
    ax.set_xticks(ticks=np.arange(len(hm.properties)), labels= imp.properties.values, rotation=45,ha='right')
    ax.set_yticks(ticks=np.arange(1), labels=[''])

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size='15%', pad=0.4)
    fig_per.add_axes(cax)
    cbar = fig_per.colorbar(im, cax=cax, ticks=np.arange(0,1.1,0.1), orientation='horizontal')
    plt.title('Fraction of grid cells of scen that improved or stayed the same compared to sim')
    fig_per.tight_layout()


    col1, col2 = st.columns([1,1])

    col1.write(fig_hmap)
    col2.write(fig_per)

with tab2:
    # load data
    if useCat:
        from xscen.config import CONFIG, load_config
        from xscen.catalog import ProjectCatalog

        load_config('paths_neree.yml', 'config.yml', verbose=(__name__ == '__main__'),
                    reset=True)
        pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

        # get warminglevel
        wl = pcat.search(processing_level='ensemble-warminglevels').to_dask()
    else:
        print('not ready yet')

    #choose data
    cols = st.columns(3)
    option_stats =  cols[0].selectbox('Statistiques', ['max', 'mean','min', 'stdev'])
    def show_long_name(name):
        return f"{wl[name].attrs['long_name']} ({name})"

    option_ind = cols[1].selectbox('Indicateurs',[x for x in wl.data_vars if option_stats in x], format_func = show_long_name)
    option_season = cols[2].selectbox('Saisons', wl.season.values)

    #plot data
    cols2 = st.columns(len(wl.warminglevel.values))
    cmap = 'viridis_r' if wl[option_ind].attrs['standard_name'] == 'precipitation_flux' else 'plasma'
    vmin = wl[option_ind].min().values
    vmax = wl[option_ind].max().values
    select_wl = wl[option_ind].sel(season=option_season)
    #col1.write(hv.render(prop_ref.hvplot(title=f'REF\n{long_name}',width=600, height=616, cmap=cmap, clim=(mini_prop,maxi_prop))))


    # cols2[0].write(hv.render(select_wl.hvplot( cmap=cmap, clim=(vmin,vmax),by='warminglevel',
    #                          subplots=True)))

    fig_wl, axs = plt.subplots(1, len(wl.warminglevel.values), figsize=(15, 5))

    for i,w in enumerate(wl.warminglevel.values):
        wl[option_ind].sel(season=option_season, warminglevel=w).plot( vmin=vmin, vmax=vmax)
    st.write(fig_wl)



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




#st.write(fig_hmap)
#st.write(fig_per)
