#!/bin/bash
#SBATCH --time=168:00:00 #modifier pour vos besoins
#SBATCH --account=ctb-frigon
#SBATCH --constraint=genoa # pour accéder à bébé narval
#SBATCH --partition=c-frigon # pour avoir la priorité Ouranos
#SBATCH --cpus-per-task=20
#SBATCH --mem=500G #modifier selon vos besoins
#SBATCH --output=/home/julavoie/code/test-vs/info-crue-cmip6/slurm_outputs/%x_%j.out
#SBATCH --mail-type=ALL # optionel
#SBATCH --mail-user=lavoie.juliette@ouranos.ca # optionel


module load StdEnv/2023 gcc openmpi python/3.11 arrow/16.1.0 openmpi netcdf proj esmf geos mpi4py/3.1.4 ipykernel/2023b scipy-stack/2023b

bash /project/ctb-frigon/scenario/environnements/config_xscen0.9.0_env_slurm.sh # fourni par Ouranos

pip uninstall --yes xclim
#pip install --no-index /project/ctb-frigon/julavoie/wheels/xclim-0.50.1.dev1-py3-none-any.whl

#pip install --no-index /project/ctb-frigon/julavoie/wheels/xclim-0.51.1.dev3-py3-none-any.whl # master on July 25th (after merge of mbcn PR)
pip install --no-index /project/ctb-frigon/julavoie/wheels/xclim-0.51.1.dev10-py3-none-any.whl # branch fix-jitter august 4

python /home/julavoie/code/test-vs/info-crue-cmip6/workflow.py CMIP6_ScenarioMIP_MIROC_MIROC6_ssp370_r2i1p1f1_global

#python /home/julavoie/code/test-vs/info-crue-cmip6/workflow.py $SLURM_JOB_NAME
## --job-name ACCESS-ESM1-5