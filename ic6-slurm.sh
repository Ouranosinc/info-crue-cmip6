#!/bin/bash
#SBATCH --time=00:10:00 #modifier pour vos besoins
#SBATCH --account=ctb-frigon
#SBATCH --constraint=genoa # pour accéder à bébé narval
#SBATCH --partition=c-frigon # pour avoir la priorité Ouranos
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G #modifier selon vos besoins
#SBATCH --output=/home/julavoie/code/info-crue-cmip6/slurm-outputs/%j.out
#SBATCH --mail-type=ALL # optionel
#SBATCH --mail-user=lavoie.juliette@ouranos.ca # optionel

module load StdEnv/2023 gcc openmpi python/3.11 arrow/16.1.0 openmpi netcdf proj esmf geos mpi4py

bash /project/ctb-frigon/julavoie/ouranos_commun/config_xscen0.9.0_env_slurm.sh # fourni par Ouranos

yes| pip uninstall xclim
pip install /project/ctb-frigon/julavoie/wheels/xclim-0.50.0-py3-none-any.whl

python /home/julavoie/code/info-crue-cmip6/workflow.py