id=[
    'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp245_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp370_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp245_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp370_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp245_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp370_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp245_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp370_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp245_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp370_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp245_r1i1p1f1_global',
    'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp370_r1i1p1f1_global'
]


rule all:
    input: expand("/project/ctb-frigon/julavoie/info-crue-cmip6/final/day_{id}_QC-EMDNA.zarr",id=id)

rule wf:
    output: "/project/ctb-frigon/julavoie/info-crue-cmip6/final/day_{id}_QC-EMDNA.zarr"
    retries: 2
    shell:"""
            module load StdEnv/2023 gcc openmpi python/3.11 arrow/16.1.0 openmpi netcdf proj esmf geos mpi4py/3.1.4 ipykernel/2023b scipy-stack/2023b
            bash /project/ctb-frigon/scenario/environnements/config_xscen0.9.0_env_slurm.sh # fourni par Ouranos
            pip uninstall --yes xclim
            pip install --no-index /project/ctb-frigon/julavoie/wheels/xclim-0.51.1.dev10-py3-none-any.whl # branch fix-jitter august 4
            python /home/julavoie/code/test-vs/info-crue-cmip6/workflow.py {id}
         """
# use squeue --format="%.18i %.9P %.8j %.8k %.8u %.2t %.10M %.6D %.20R %Q" -u julavoie to see name of rule in comment