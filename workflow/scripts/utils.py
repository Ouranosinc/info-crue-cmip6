from pathlib import Path
from dask.distributed import Client, LocalCluster
import os
import xscen as xs
from xscen import CONFIG
from zipfile import ZipFile

xs.load_config("config/config.yml","config/paths.yml")


def dask_cluster(params):
    cluster = LocalCluster(
        n_workers=params.n_workers,
        threads_per_worker=1, #params.cpus_per_task/params.n_workers,
        memory_limit="200G", #f"{int(int(params.mem.replace('GB',''))/params.n_workers)}GB",
        local_directory=os.environ['SLURM_TMPDIR'], **CONFIG['dask'].get('client', {}))
    client = Client(cluster)
    return client



# eventually take this from xscen
def zip_directory(
    root: str | os.PathLike,
    zipfile: str | os.PathLike,
    delete: bool = False,
    **zip_args,
):
    r"""Make a zip archive of the content of a directory.

    Parameters
    ----------
    root : path
        The directory with the content to archive.
    zipfile : path
        The zip file to create.
    delete : bool
        If True, the original directory is deleted after zipping.
    \*\*zip_args
        Any other arguments to pass to :py:mod:`zipfile.ZipFile`, such as "compression".
        The default is to make no compression (``compression=ZIP_STORED``).
    """
    root = Path(root)

    def _add_to_zip(zf, path, root):
        zf.write(path, path.relative_to(root))
        if path.is_dir():
            for subpath in path.iterdir():
                _add_to_zip(zf, subpath, root)

    with ZipFile(zipfile, "w", **zip_args) as zf:
        for file in root.iterdir():
            _add_to_zip(zf, file, root)

    if delete:
        sh.rmtree(root)


def unzip_directory(zipfile: str | os.PathLike, root: str | os.PathLike):
    r"""Unzip an archive to a directory.

    This function is the exact opposite of :py:func:`xscen.io.zip_directory`.

    Parameters
    ----------
    zipfile : path
        The zip file to read.
    root : path
        The directory where to put the content to archive.
        If doesn't exist, it will be created (and all its parents).
        If it exists, should be empty.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    with ZipFile(zipfile, "r") as zf:
        zf.extractall(root)


def create_tmp_path(path):
    return f"{os.environ['SLURM_TMPDIR']}/{Path(path).name.replace('.zip','')}"



def tmp_zarr_and_zip(ds, p):
    tmp_path=create_tmp_path(p)
    xs.save_to_zarr(ds, tmp_path)
    zip_directory(tmp_path, p)