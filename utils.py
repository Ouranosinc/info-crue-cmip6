import xarray as xr
import logging
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


from xclim.sdba.measures import rmse
from xclim import atmos, sdba

from xscen.config import CONFIG, load_config
from xscen.io import save_to_zarr
from xscen.common import  maybe_unstack


load_config('paths.yml', 'config.yml', verbose=(__name__ == '__main__'), reset=True)
workdir = Path(CONFIG['paths']['workdir'])
refdir = Path(CONFIG['paths']['refdir'])
logger = logging.getLogger('xscen')

def compute_properties(sim, ref, period):
    # TODO add more diagnostics, xclim.sdba and from Yannick (R2?)
    nan_count = sim.to_array().isnull().sum('time').mean('variable')
    hist = sim.sel(time=period)
    ref = ref.sel(time=period)

    # Je load deux des variables pour essayer d'éviter les KilledWorker et Timeout
    out = xr.Dataset(data_vars={

        'tx_mean_rmse': rmse(atmos.tx_mean(hist.tasmax, freq='MS').chunk({'time': -1}),
                             atmos.tx_mean(ref.tasmax, freq='MS').chunk({'time': -1})),
        'tn_mean_rmse': rmse(atmos.tn_mean(tasmin=hist.tasmin, freq='MS').chunk({'time': -1}),
                             atmos.tn_mean(tasmin=ref.tasmin, freq='MS').chunk({'time': -1})),
        'prcptot_rmse': rmse(atmos.precip_accumulation(hist.pr, freq='MS').chunk({'time': -1}),
                             atmos.precip_accumulation(ref.pr, freq='MS').chunk({'time': -1})),
        'nan_count': nan_count,
    })

    return out


def calculate_properties(ds, pcat, step, unstack=False, diag_dict=CONFIG['diagnostics']['properties']):
    """
    Calculate diagnostic in the dictionary, save them all in one zarr and updates the catalog.
    The function verifies that a catalog entry doesn't exist already.
    The initial calculations are made in the workdir, but move to a permanent location afterwards.

    If the property is monthly or seasonal, we only keep the first month/season.

    :param ds: Input dataset (with tasmin, tasmax, pr) and the attrs we want to be passed to the final dataset
    :param pcat: Project Catalogue to update
    :param step: Type of input (ref, sim or scen)
    :param diag_dict: Dictionnary of properties to calculate. needs key func, var and args
    :return: A dataset with all properties
    """
    region_name = ds.attrs['cat/domain']
    if not pcat.exists_in_cat(domain=region_name, processing_level=f'diag_{step}', id=ds.attrs['cat/id']):
        for i, (name, prop) in enumerate(diag_dict.items()):
            logger.info(f"Calculating diagnostic {name}")
            prop = eval(prop['func'])(da=ds[prop['var']], **prop['args']).load()

            if unstack:
                prop = maybe_unstack(
                    prop,
                    stack_drop_nans=CONFIG['custom']['stack_drop_nans'],
                    coords=refdir / f'coords_{region_name}.nc',
                    rechunk={d: CONFIG['custom']['out_chunks'][d] for d in ['lat', 'lon']}
                )

            if ("season" or "month") in prop.coords:
                prop = prop[0]

            # put all properties in one dataset
            if i == 0:
                all_prop = prop.to_dataset(name=name)
            else:
                all_prop[name] = prop
        all_prop.attrs.update(ds.attrs)

        path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=region_name,
                                                               sim_id=ds.attrs['cat/id'],
                                                               step=step))
        path_diag_exec = f"{workdir}/{path_diag.name}"
        save_to_zarr(ds=all_prop, filename=path_diag_exec, mode='o', itervar=True)
        shutil.move(path_diag_exec, path_diag)
        pcat.update_from_ds(ds=all_prop,
                            info_dict={'processing_level': f'diag_{step}'},
                            path=str(path_diag))
        return all_prop

def plot_diagnotics(ref, sim, scen):
    """
    Creates plots of diagnostics.
    If show_maps is True, creates one figure per property with 3 property map (ref, sim, scen) and 2 bias maps(sim-ref, scen-ref).
    Always creates a heat map, showing which one of sim or scen is the best for each properties.
    :param ref: reference dataset
    :param sim: simulation dataset
    :param scen: scenario dataset
    :return: list of paths to figures
    """
    sim_id = sim.attrs['sim_id']
    hmap = []
    paths=[]
    diag_dir = Path(CONFIG['paths']['diagfigs'].format(sim_id=sim_id))
    diag_dir.mkdir(exist_ok=True, parents=True)

    #iterate through all available properties
    for i, var_name in enumerate(sim.data_vars):
        # get property
        prop_sim = sim[var_name]
        prop_ref = ref[var_name]
        prop_scen = scen[var_name]

        #calculate bias
        bias_scen_prop = sdba.measures.bias(sim=prop_scen, ref=prop_ref).load()
        bias_sim_prop = sdba.measures.bias(sim=prop_sim, ref=prop_ref).load()


        #mean the absolute value of the bias over all positions and add to heat map
        hmap.append([abs(bias_sim_prop).mean().values, abs(bias_scen_prop).mean().values])

        #individual figure for all properties with 5 maps
        if CONFIG['diagnostics']['show_maps']:
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
            fig.suptitle(var_name, fontsize=20)
            fig.tight_layout()

            fig.savefig(diag_dir / f"{prop_sim.name}.png")
            paths.append(diag_dir / f"{prop_sim.name}.png")

    # plot heat map of biases ( 1 column per properties, 1 column for sim , 1 column for scen)
    hmap = np.array(hmap).T
    hmap = np.array(
        [(c - min(c)) / (max(c) - min(c)) if max(c) != min(c) else [0.5] * len(c) for c in
         hmap.T]).T
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
    fig_hmap.savefig(diag_dir / f"hmap.png", bbox_inches='tight')
    paths.append(diag_dir / f"hmap.png")
    return paths



def save_diagnotics(ref, sim, scen, pcat):
    """
    calculate the biases amd the heat map and save
    Saves files of biases and heat map.
    :param ref: reference dataset
    :param sim: simulation dataset
    :param scen: scenario dataset
    :param pcat: project catalog to update
    """
    hmap = []
    #iterate through all available properties
    for i, var_name in enumerate(sim.data_vars):
        # get property
        prop_sim = sim[var_name]
        prop_ref = ref[var_name]
        prop_scen = scen[var_name]

        #calculate bias
        bias_scen_prop = sdba.measures.bias(sim=prop_scen, ref=prop_ref).load()
        bias_sim_prop = sdba.measures.bias(sim=prop_sim, ref=prop_ref).load()

        # put all bias in one dataset
        if i == 0:
            all_bias_scen_prop = bias_scen_prop.to_dataset(name=var_name)
            all_bias_sim_prop = bias_sim_prop.to_dataset(name=var_name)
        else:
            all_bias_scen_prop[var_name] = bias_scen_prop
            all_bias_sim_prop[var_name] = bias_sim_prop


        #mean the absolute value of the bias over all positions and add to heat map
        hmap.append([abs(bias_sim_prop).mean().values, abs(bias_scen_prop).mean().values])

    # plot heat map of biases ( 1 column per properties, 1 column for sim , 1 column for scen)
    hmap = np.array(hmap).T
    hmap = np.array(
        [(c - min(c)) / (max(c) - min(c)) if max(c) != min(c) else [0.5] * len(c) for c in
         hmap.T]).T

    path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat/domain'],
                                                           sim_id=scen.attrs['cat/id'],
                                                           step='hmap'))
    #replace zarr by npy
    path_diag = path_diag.with_suffix('.npy')
    np.save(path_diag, hmap)

    for ds, step in zip([all_bias_scen_prop,all_bias_sim_prop],['scen_bias','sim_bias']):
        path_diag = Path(CONFIG['paths']['diagnostics'].format(region_name=scen.attrs['cat/domain'],
                                                               sim_id=scen.attrs['cat/id'],
                                                               step=step))
        save_to_zarr(ds=ds, filename=path_diag, mode='o', itervar=True)
        pcat.update_from_ds(ds=ds,
                            info_dict={'processing_level': f'diag_{step}'},
                            path=str(path_diag))

