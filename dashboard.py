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

@st.cache(hash_funcs={xr.core.dataset.Dataset: id},ttl=60)
def load_zarr(path):
    return xr.open_zarr(path,decode_timedelta= False)

@st.cache(hash_funcs={xr.core.dataset.Dataset: id},ttl=60)
def load_nc(path):
    return xr.open_dataset(path,decode_timedelta= False)


useCat = st.checkbox("use catalog (only for local version)")

tab1, tab2, tab3, tab4 = st.tabs(["Diagnostiques", "Niveaux de réchauffement global", "Horizons Temporels", "Sélection vs tout"])
with tab1:
    #useCat=True


    #load data
    if useCat:
        cols = st.columns(2)
        from xscen.config import CONFIG, load_config
        from xscen.catalog import ProjectCatalog
        load_config('paths_neree.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
        #pcat = ProjectCatalog(CONFIG['paths']['dashboard_catalog'])
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
        hm = pcat.search( id=option_id,processing_level='diag-heatmap', domain=option_domain).to_dask()
        imp = pcat.search(id=option_id,processing_level='diag-improved', domain=option_domain).to_dask()


    else:
        cols = st.columns(3)
        #option_id = st.selectbox('id',[x[30:-5] for x in glob.glob('dashboard_data/diag_scen_bias_*')])
        ids = [x[30:-6] for x in glob.glob('dashboard_data/diag-scen-meas_*')]

        models = sorted(set([y.split('_')[3] for y in ids ]))
        option_model = cols[0].selectbox('Models', models)
        ids_of_model = [x for x in ids if option_model in x]
        exps = sorted(set([y.split('_')[4] for y in ids_of_model]))
        option_ssp = cols[1].selectbox('Experiments',exps)

        ids_of_model_exps = [x for x in ids if option_model in x and option_ssp in x]
        members = sorted(set([y.split('_')[5] for y in ids_of_model_exps]))
        option_member = cols[2].selectbox('Members', members)

        option_id = [x for x in ids if option_model in x and option_ssp in x and option_member in x][0]

        ref = load_nc(f'dashboard_data/diag_ref_ECMWF_ERA5-Land_NAM_qc.nc')
        sim = load_nc(f'dashboard_data/diag-sim-prop_{option_id}_qc.nc')
        scen = load_nc(f'dashboard_data/diag-scen-prop_{option_id}_qc.nc')
        bias_sim = load_nc(f'dashboard_data/diag-sim-meas_{option_id}_qc.nc')
        bias_scen = load_nc(f'dashboard_data/diag-scen-meas_{option_id}_qc.nc')

        hm = load_nc(f'dashboard_data/diag-heatmap_{option_id}_qc.nc')
        imp = load_nc(f'dashboard_data/diag-improved_{option_id}_qc.nc')

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
    if 'ratio' in measure_name : #center around 1 for ratio
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
    cols = st.columns([3,1,1])
    #option_type=  cols[0].selectbox('Type',['delta', 'absolute'])

    # load data
    if useCat:
        #from xscen.config import CONFIG, load_config
        #from xscen.catalog import ProjectCatalog

        #load_config('paths_neree.yml', 'config.yml', verbose=(__name__ == '__main__'),reset=True)
        #pcat = ProjectCatalog(CONFIG['paths']['project_catalog'])

        # get warminglevel
        wl15 = pcat.search(processing_level='delta-+1.5C-selection').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        wl2 = pcat.search(processing_level='delta-+2C-selection').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        wl3 = pcat.search(processing_level='delta-+3C-selection').to_dask(xarray_open_kwargs={'decode_timedelta':False})


        #ensemble_sizes= {x.horizon.values[0]:x.horizon.attrs['ensemble_size'] for x in wls.values()}
        #wl = xr.concat(wls.values(), dim='horizon')
    else:
        ensemble_sizes={}


        wl15 = load_zarr(f'dashboard_data/ensemble-deltas-1.5_CMIP6_ScenarioMIP_qc.zarr')
        wl2 = load_zarr(f'dashboard_data/ensemble-deltas-2_CMIP6_ScenarioMIP_qc.zarr')
        wl3 = load_zarr(f'dashboard_data/ensemble-deltas-3_CMIP6_ScenarioMIP_qc.zarr')


        #ensemble_sizes[f"+{h}C"]=cur_wl.horizon.attrs['ensemble_size']
            #wls.append(cur_wl)
        #wl = xr.concat(wls, dim='horizon')
    #choose data
    def show_long_name(name):
        for var in wl2.data_vars:
            if f"{name}_delta" in var:
                return f"{wl2[var].attrs['long_name']} ({name})"


    option_ind_wl = cols[0].selectbox('Indicateurs',set([x.split('_delta')[0] for x in wl2.data_vars]), format_func = show_long_name)
    option_stats_wl =  cols[1].selectbox('Statistiques', ['mean', 'max','min', 'stdev','p10', 'p50', 'p190'])
    option_season_wl = cols[2].selectbox('Saisons', wl2.season.values)
    complete_var = f"{option_ind_wl}_delta_1991_2020_{option_stats_wl}"


    #plot data
    cmap = 'viridis_r' if  'precipitation' in wl2[complete_var].attrs['standard_name'] else 'plasma'
    vmin = np.min([cur_wl[complete_var].sel(season=option_season_wl).min().values for cur_wl in [wl15, wl2, wl3]])
    vmax = np.max([cur_wl[complete_var].sel(season=option_season_wl).max().values for cur_wl in [wl15, wl2, wl3]])


    col3 = st.columns(3)
    for i, cur_wl in enumerate([wl15, wl2, wl3]):
        select_wl = cur_wl[complete_var].sel(season=option_season_wl).isel(horizon=0)
        col3[i].write(
            hv.render(select_wl.hvplot(cmap=cmap,
                                                      width=450,
                                                      height=350,
                                                      clim=(vmin, vmax))))
        col3[i].write(f"Nombre de réalisations de l'ensemble: {cur_wl.attrs['ensemble_size']}")
    st.write( f"Warming levels are calculated from the {select_wl.horizon.attrs['baseline']} baseline.")

with tab3:
    cols = st.columns([3,1,1])

    # load data
    if useCat:

        # get warminglevel
        s1 = pcat.search(processing_level='delta-ssp126-2081-2100-selection').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        s2 = pcat.search(processing_level='delta-ssp245-2081-2100-selection').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        s3 = pcat.search(processing_level='delta-ssp370-2081-2100-selection').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        s4 = pcat.search(processing_level='delta-ssp585-2081-2100-selection').to_dask(xarray_open_kwargs={'decode_timedelta':False})

    else:
        ensemble_sizes={}

        # wl15 = load_zarr(f'dashboard_data/ensemble-deltas-1.5_CMIP6_ScenarioMIP_qc.zarr')
        # wl2 = load_zarr(f'dashboard_data/ensemble-deltas-2_CMIP6_ScenarioMIP_qc.zarr')
        # wl3 = load_zarr(f'dashboard_data/ensemble-deltas-3_CMIP6_ScenarioMIP_qc.zarr')

        #ensemble_sizes[f"+{h}C"]=cur_wl.horizon.attrs['ensemble_size']
            #wls.append(cur_wl)
        #wl = xr.concat(wls, dim='horizon')

    #choose data
    def show_long_name(name):
        for var in s2.data_vars:
            if name in var:
                return f"{s2[var].attrs['long_name']} ({name})"


    option_ind_s = cols[0].selectbox('Indicateurs ',set([x.split('_delta')[0] for x in s2.data_vars]), format_func = show_long_name)
    option_stats_s =  cols[1].selectbox('Statistiques ', [ 'mean', 'max', 'min', 'stdev','p10', 'p50', 'p90'])
    option_season_s = cols[2].selectbox('Saisons ', s2.season.values)
    complete_var = f"{option_ind_s}_delta_1991_2020_{option_stats_s}"


    #plot data
    cmap = 'viridis_r' if  'precipitation' in s2[complete_var].attrs['standard_name']  else 'plasma'
    vmin = np.min([cur_wl[complete_var].sel(season=option_season_s).min().values for cur_wl in [s1,s2,s3,s4]])
    vmax = np.max([cur_wl[complete_var].sel(season=option_season_s).max().values for cur_wl in [s1,s2,s3,s4]])


    col3 = st.columns(4)
    names = ['ssp1-2.6', 'ssp2-4.5', 'ssp3-7.0', 'ssp5-8.5']
    for i, cur_wl in enumerate([s1,s2,s3,s4]):
        select_wl = cur_wl[complete_var].sel(season=option_season_s).isel(horizon=0)
        col3[i].write(names[i])
        col3[i].write(
            hv.render(select_wl.hvplot(cmap=cmap,width=400,
                                                      height=350,
                                                      clim=(vmin, vmax))))
        col3[i].write(f"Nombre de réalisations de l'ensemble: {cur_wl.attrs['ensemble_size']}")

with tab4:
    col = st.columns([1,5,1,1])
    if useCat:
        option_ens = col[0].selectbox('ensemble', ['+1.5C', '+2C','+3C' ,'ssp126-2081-2100','ssp245-2081-2100','ssp370-2081-2100','ssp585-2081-2100'])
        option_can = 'include'
        if option_ens in ['ssp245-2081-2100','ssp370-2081-2100']:
            option_can = col[0].selectbox('CanESM5', ['include', 'exclude'])
        can=''
        r=''
        if option_can== 'exclude':
            can = 'NoCanESM5'
            option_r = col[0].selectbox('members', ['all members', 'only r1'])
            if option_r == 'only r1':
                r = 'r1'
        selection = pcat.search(processing_level=f'delta-{option_ens}-selection{can}{r}').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        all = pcat.search(processing_level=f'delta-{option_ens}-all{r}').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        diff = pcat.search(processing_level=f'{option_ens}-selection{can}{r}VSall{r}').to_dask(xarray_open_kwargs={'decode_timedelta':False})
        pvalues=None
        if len(pcat.search(processing_level=f'p-{option_ens}-selection{can}{r}VSall{r}').df)>0:
            pvalues = pcat.search(processing_level=f'p-{option_ens}-selection{can}{r}VSall{r}').to_dask(xarray_open_kwargs={'decode_timedelta':False})

    if 'baseline' in selection.horizon.attrs:
        st.write(f"Warming levels are calculated  from the {selection.horizon.attrs['baseline']} baseline")
    def show_long_name_ens(name):
        return f"{all[f'{name}_mean'].attrs['long_name']} ({name})"

    option_ind_ens = col[1].selectbox('Indicateur',
                                   set(['_'.join(x.split('_')[:-1]) for x in
                                        all.data_vars]),
                                   format_func=show_long_name_ens)
    option_stats_ens = col[2].selectbox('Statistique',[ 'mean', 'max', 'min', 'stdev','p10', 'p50', 'p90'])
    option_season_ens = col[3].selectbox('Saison', all.season.values)
    complete_var = f"{option_ind_ens}_{option_stats_ens}"



    col_fig = st.columns(3)


    select_haus = selection[complete_var].sel(season=option_season_ens).isel(horizon=0)
    select_all = all[complete_var].sel(season=option_season_ens).isel(horizon=0)
    select_diff = diff[complete_var].sel(season=option_season_ens).isel(horizon=0)

    #plot data
    cmap = 'viridis_r' if  'precipitation' in select_all.attrs['standard_name'] else 'plasma'
    cmap_bias = 'BrBG' if 'precipitation' in select_all.attrs[
        'standard_name'] else 'coolwarm'
    vmin = np.min([cur.min().values for cur in [select_all, select_haus]])
    vmax = np.max([cur.max().values for cur in [select_all, select_haus]])





    col_fig[0].write('Selection')
    col_fig[0].write(hv.render(select_haus.hvplot(cmap=cmap,width=450,height=350,clim=(vmin, vmax))))
    col_fig[0].write(f"Nombre de réalisations de l'ensemble: {selection.attrs['ensemble_size']}")

    col_fig[1].write('All')
    col_fig[1].write( hv.render(select_all.hvplot(cmap=cmap, width=450,height=350, clim=(vmin, vmax))))
    col_fig[1].write(f"Nombre de réalisations de l'ensemble: {all.attrs['ensemble_size']}")

    col_fig[2].write('(Selection - All)/All')
    diff_plot= select_diff.hvplot(cmap=cmap_bias,clim =(- abs(select_diff).max(),abs(select_diff).max()),width=450,height=350)
    col_fig[2].write( hv.render(diff_plot))

    if pvalues:
        select_p = pvalues[option_ind_ens].sel(season=option_season_ens).isel(horizon=0)
        levels = [0, 0.05, 10]
        colors = ['#43A047', '#E53935']
        col_fig[2].write( hv.render(select_p.hvplot(width=450,height=350).options(color_levels=levels, cmap=colors)))





# # test panel
# # https://github.com/holoviz/panel/issues/1074
# # https://github.com/streamlit/streamlit/issues/927
# # import panel as pn
# # import hvplot.xarray
# #
# # hvplot_plot = scen['mean-pr'].hvplot()
# # hvplot_pane = pn.pane.HoloViews(hvplot_plot, name="Holoviews Plot")
# # tabs = pn.Tabs(hvplot_pane)
# # st.bokeh_chart(tabs.get_root())
# #
# # def plot_var(variable):
# #     return scen[variable].hvplot()
# #
# # pan = pn.interact(plot_var, variable=list(scen.data_vars))
# # st.write(hv.render(pan).get_root(), backend='bokeh')
# #
# #
# #
# #
# # st.write(fig_hmap)
# # st.write(fig_per)
