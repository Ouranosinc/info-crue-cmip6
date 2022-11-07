import xarray as xr
import logging
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import logging
from datetime import datetime as dt
from getpass import getpass
from pathlib import Path

from paramiko import SSHClient
from scp import SCPClient


from xclim.sdba.measures import rmse
from xclim import atmos, sdba
from xclim.core.units import convert_units_to

from xscen.io import save_to_zarr
from xscen.utils import  maybe_unstack,unstack_fill_nan
from xscen.scripting import measure_time, send_mail
import xscen as xs



logger = logging.getLogger('xscen')




def calculate_properties(ds, diag_dict, unstack=False, path_coords=None, unit_conversion={}):
    """
    Calculate properties in the dictionary.
    If the property is monthly or seasonal, we only keep the first month/season.

    :param ds: Input dataset (with tasmin, tasmax, pr) and the attrs we want to be passed to the final dataset
    :param diag_dict: Dictionary of properties to calculate. needs key func, var and args
    :param unit_conversion: Dictionary {variable: units to convert to}
    :return: A dataset with all properties
    """

    # we will need to have same units to be able to do measures properly
    # it won't always be able to convert after properties (eg. kg2 m-4 s-2 -> mm2 d-2 doesn't work)
    # so we have to do it now
    for var, unit in unit_conversion.items():
        ds[var] = convert_units_to(ds[var], unit)

    region_name=ds.attrs["cat:domain"]
    for i, (name, prop_dict) in enumerate(diag_dict.items()):
        logger.info(f"Calculating diagnostic {name}")
        prop = eval(prop_dict['func'])(da=ds[prop_dict['var']], **prop_dict['args']).load()

        # TODO: create something more general that keeps all months/seasons
        if "season" in prop.coords:
            prop = prop.isel(season=0)
            prop=prop.drop('season')
        if "month" in prop.coords:
            prop = prop.isel(month=0)
            prop=prop.drop('month')

        if unstack:
            prop = unstack_fill_nan(
                prop,
                coords=path_coords,
            )
            prop=prop.transpose("lat", "lon")

        prop.attrs['measure']=prop_dict['measure'] if 'measure' in prop_dict else 'bias'


        # put all properties in one dataset
        if i == 0:
            all_prop = prop.to_dataset(name=name)
        else:
            all_prop[name] = prop
    all_prop.attrs.update(ds.attrs)


    return all_prop








def measures_and_heatmap(ref, sims):
    """
    calculate the measures of the difference with the properties of ref and the properties of each sim and create the heat map
    :param ref: reference dataset
    :param sims: list of datasets to compare with ref. Each will be a row on the heatmap.
    """
    hmap = []
    all_measures=[]
    for sim in sims:
        row =[]
        # iterate through all available properties
        for i, var_name in enumerate(sorted(sim.data_vars)):
            # get property
            prop_sim = sim[var_name]
            prop_ref = ref[var_name]

            #choose right measure
            measure_name= prop_sim.attrs['measure'] if 'measure' in prop_sim.attrs else 'bias'
            measure= getattr(sdba.measures,measure_name)

            #calculate bias
            bias_sim_prop = measure(sim=prop_sim, ref=prop_ref).load()

            # put all bias in one dataset
            if i == 0:
                all_bias_sim_prop = bias_sim_prop.to_dataset(name=var_name)
            else:
                all_bias_sim_prop[var_name] = bias_sim_prop


            #mean the absolute value of the bias over all positions and add to heat map
            if measure_name == 'ratio': #if ratio, best is 1, this moves "best to 0 to compare with bias
                row.append(abs(bias_sim_prop -1).mean().values)
            else:
                row.append(abs(bias_sim_prop).mean().values)
        all_bias_sim_prop.attrs.update(sim.attrs)
        all_measures.append(all_bias_sim_prop)
        # append all properties
        hmap.append(row)



    # plot heat map of biases ( 1 column per properties, 1 column for sim , 1 column for scen)
    hmap = np.array(hmap)
    # normalize to 0-1 -> best-worst
    hmap = np.array(
        [(c - min(c)) / (max(c) - min(c)) if max(c) != min(c) else [0.5] * len(c) for c in
         hmap.T]).T

    return all_measures, hmap




def email_nan_count(path, region_name):
    ds_ref_props_nan_count = xr.open_zarr(path, decode_timedelta=False).load()
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.winter.copy()
    cmap.set_under('white')
    ds_ref_props_nan_count.nan_count.plot(ax=ax, vmin=1, vmax=1000, cmap=cmap)
    ax.set_title(
        f'Reference {region_name} - NaN count \nmax {ds_ref_props_nan_count.nan_count.max().item()}')
    plt.close('all')
    send_mail(
        subject=f'Reference for region {region_name} - Success',
        msg=f"Action 'makeref' succeeded for region {region_name}.",
        attachments=[fig]
    )

def move_then_delete(dirs_to_delete, moving_files, pcat):
    """
    First, move the moving_files. If they are zarr, update catalog
    with new path.
    Then, delete everything in dir_to_delete
    :param dirs_to_delete: list of directory where all content will be deleted
    :param moving_files: list of lists of path of files to move with format: [[source 1, destination1], [source 2, destination2],...]
    :param pcat: project catalog to update
    """

    for files in moving_files:
        source, dest = files[0], files[1]
        if Path(source).exists():
            shutil.move(source, dest)
            if dest[-5:] =='.zarr':
                ds = xr.open_zarr(dest)
                pcat.update_from_ds(ds=ds, path=dest)

    # erase workdir content if this is the last step
    for dir_to_delete in dirs_to_delete:
        if Path(dir_to_delete).exists() and Path(dir_to_delete).is_dir():
            shutil.rmtree(dir_to_delete)
            os.mkdir(dir_to_delete)


def save_move_update(ds,pcat, init_path, final_path,info_dict=None,
                     encoding=None, mode='o', itervar=False, server='doris'):
    encoding = encoding or {var: {'dtype':'float32'} for var in ds.data_vars}
    save_to_zarr(ds, init_path, encoding=encoding, mode=mode,itervar=itervar)
    if server == 'neree':
        python_scp(init_path, Path(final_path).parent, 'doris.ouranos.ca')
    else:
        shutil.move(init_path,final_path)
    pcat.update_from_ds(ds=ds, path=str(final_path),info_dict=info_dict)


def python_scp(source_path, destination_path, server_address):
    """
    scp with python
    based on https://gist.github.com/Zeitsperre/448f8d6d7bf907c9e9976b4bf2069fb1


    Parameters
    ----------
    source_path: file to transfer
    destination_path: destination to transfer the file to
    server_address: server to scp to

    Returns
    -------
    None

    Notes
    _____
    Password argument is not neccessary because I have set up a ssh key between neree and doris.
    On neree:
    ssh-keygen -t ed25519
    ssh-copy-id [usager]@doris.ouranos.ca
    """
    logging.basicConfig(
        filename=f"{dt.strftime(dt.now(), '%Y-%m-%d')}_{Path(__file__).stem}.log",
        level=logging.INFO,
    )  # can't go wrong making a logfile

    user = "jlavoie"  # username on the server being accessed.
    my_folder = Path(source_path)  # folder being transferred

    if my_folder.exists():
        with SSHClient() as ssh:
            ssh.load_system_host_keys()  # loads any SSH keys. If keys are loaded, password not needed.

            #server_address = f"doris.ouranos.ca"
            logging.info(f"Connecting to {server_address}")

            ssh.connect(server_address, username=user)  # opens a shell to create a connection.
            logging.info(f"Connected!")

            with SCPClient(ssh.get_transport(), socket_timeout=30.0) as scp:
                scp.put(my_folder, recursive=True, remote_path=destination_path)
            logging.info(f"{my_folder} transfered to {destination_path}.")

    else:
        logging.info(f"{my_folder} doesn't exist.")

def save_and_update(ds, path, pcat):
    path = path.format(**xs.utils.get_cat_attrs(ds))
    save_to_zarr(ds=ds, filename=path)
    pcat.update_from_ds(ds=ds, path=path)