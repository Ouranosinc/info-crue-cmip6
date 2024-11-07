from pathlib import Path
import xscen as xs


# Load configuration
configfile: "config/config.yml"
configfile: "config/paths.yml"

sim_id=[
    #'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp245_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp370_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp245_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp370_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp245_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp370_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp245_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp370_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp245_r1i1p1f1',
     'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp370_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp245_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp370_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_NIMS-KMA_KACE-1-0-G_ssp245_r1i1p1f1',
    # 'CMIP6_ScenarioMIP_NIMS-KMA_KACE-1-0-G_ssp370_r1i1p1f1',
]



regions= list(config['custom']['regions'].keys())

wdir= Path(config['paths']['workdir'])
finaldir= Path(config['paths']['finaldir'])


rule all:
    input: 
        expand(finaldir/"health/{sim_id}_health.zarr.zip",sim_id=sim_id),
        #input: expand(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_training.zarr.zip", sim_id=sim_id, region_name=list(config['custom']['regions'].keys()))
        expand(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_adjusted20w_{period}.zarr", sim_id=sim_id, region_name=['a-EMDNA'], period=['1951-1980']),
        expand(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_adjusted-dwi_{period}.zarr", sim_id=sim_id, region_name=['a-EMDNA'], period=['1951-1980']),

rule makeref:
    output: 
        default=finaldir/ "reference/{region_name}_default.zarr.zip",
        noleap=finaldir/ "reference/{region_name}_noleap.zarr.zip",
        day360=finaldir/ "reference/{region_name}_360_day.zarr.zip",
    params:
        n_workers=2,
        mem="250GB",
        cpus_per_task=4,
        time="00:30:00",
    script: "workflow/scripts/makeref.py"

rule extractregrid:
    input: 
        noleap=finaldir/ "reference/{region_name}_noleap.zarr.zip",
    output: directory(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr")
    params:
        n_workers=2,
        mem="250GB",
        cpus_per_task=4,
        time="00:10:00",
    script:
        "workflow/scripts/extract-regrid.py"

rule train:
    input:
        sim= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr",
        ref_noleap= finaldir/"reference/{region_name}_noleap.zarr.zip",
        ref_360_day= finaldir/"reference/{region_name}_360_day.zarr.zip",
    output: wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_training.zarr.zip",
    params:
        n_workers=2,
        mem="250GB",
        cpus_per_task=4,
        time="03:00:00",
    script:
        "workflow/scripts/train.py"



rule adjust_per:
    input:
        sim= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr",
        ref_noleap= finaldir/"reference/{region_name}_noleap.zarr.zip",
        ref_360_day= finaldir/"reference/{region_name}_360_day.zarr.zip",
        train= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_training.zarr.zip",
    output: directory(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_adjusted_{period}.zarr"),
    params:
        n_workers=10,
        mem="150GB",
        cpus_per_task=12,
        time="6:00:00",
    script:
        "workflow/scripts/adjust_per.py"

rule adjust_per_dontwriteinput:
    input:
        sim= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr",
        ref_noleap= finaldir/"reference/{region_name}_noleap.zarr.zip",
        ref_360_day= finaldir/"reference/{region_name}_360_day.zarr.zip",
        train= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_training.zarr.zip",
    output: directory(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_adjusted-dwi_{period}.zarr"),
    params:
        n_workers=10,
        mem="150GB",
        cpus_per_task=12,
        time="6:00:00",
    script:
        "workflow/scripts/adjust_per_dwi.py"


rule adjust_per_20w:
    input:
        sim= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr",
        ref_noleap= finaldir/"reference/{region_name}_noleap.zarr.zip",
        ref_360_day= finaldir/"reference/{region_name}_360_day.zarr.zip",
        train= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_training.zarr.zip",
    output: directory(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_adjusted20w_{period}.zarr"),
    params:
        n_workers=20,
        mem="150GB",
        cpus_per_task=22,
        time="6:00:00",
    script:
        "workflow/scripts/adjust_per.py"

rule clean_up:
    input: expand(wdir/"{{sim_id}}_{{region_name}}/{{sim_id}}_{{region_name}}_adjusted_{period}.zarr",period=['1951-1980','1981-2010','2011-2040','2041-2070','2071-2100'])
    output: finaldir/"final_regions/{region_name}/day_{sim_id}_{region_name}.zarr.zip"
    params:
        n_workers=2,
        mem="50GB",
        cpus_per_task=4,
        time="00:30:00",
    script:
        "workflow/scripts/clean_up.py"

rule concat_scen:
    input: expand(finaldir/"final_regions/{region_name}/day_{{sim_id}}_{region_name}.zarr.zip",region_name=regions)
    output: 
        pr=finaldir/"staging/simulation/biasadjusted/IC6-EM-MBCn_v10/{sim_id_slash}/day/pr/pr_day_IC6-EM-MBCn_v10_{sim_id}_1950-2100.zarr.zip",
        tasmax=finaldir/"staging/simulation/biasadjusted/IC6-EM-MBCn_v10/{sim_id_slash}/day/tasmax/tasmax_day_IC6-EM-MBCn_v10_{sim_id}_1950-2100.zarr.zip",
        tasmin=finaldir/"staging/simulation/biasadjusted/IC6-EM-MBCn_v10/{sim_id_slash}/day/tasmin/tasmin_day_IC6-EM-MBCn_v10_{sim_id}_1950-2100.zarr.zip",
    params:
        sim_id_slash=lambda wildcards: wildcards.sim_id.replace('_','/'),
        n_workers=2,
        mem="50GB",
        cpus_per_task=4,
        time="00:30:00",
    script:
        "workflow/scripts/concat.py"

# we dont actually use bc no diag now
rule concat_sim:
    input: expand(wdir/"{{sim_id}}_{region_name}/{{sim_id}}_{region_name}_regridded.zarr",region_name=regions)
    output: 
        pr=  directory(finaldir /"regridded/{sim_id}/pr_{sim_id}_regridded.zarr"),
        tasmax= directory(finaldir /"regridded/{sim_id}/tasmax_{sim_id}_regridded.zarr"),
        tasmin=  directory(finaldir /"regridded/{sim_id}/tasmin_{sim_id}_regridded.zarr"),
    params:
        n_workers=2,
        mem="50GB",
        cpus_per_task=4,
        time="00:30:00",
    script:
        "workflow/scripts/concat.py"

rule health:
    input:
        pr=lambda wildcards: finaldir/f"staging/simulation/biasadjusted/IC6-EM-MBCn_v10/{wildcards.sim_id.replace('_','/')}/day/pr/pr_day_IC6-EM-MBCn_v10_{wildcards.sim_id}_1950-2100.zarr.zip",
        tasmax=lambda wildcards: finaldir/f"staging/simulation/biasadjusted/IC6-EM-MBCn_v10/{wildcards.sim_id.replace('_','/')}/day/tasmax/tasmax_day_IC6-EM-MBCn_v10_{wildcards.sim_id}_1950-2100.zarr.zip",
        tasmin=lambda wildcards: finaldir/f"staging/simulation/biasadjusted/IC6-EM-MBCn_v10/{wildcards.sim_id.replace('_','/')}/day/tasmin/tasmin_day_IC6-EM-MBCn_v10_{wildcards.sim_id}_1950-2100.zarr.zip",
    output: 
        finaldir/"health/{sim_id}_health.zarr.zip"
    params:
        n_workers=2,
        mem="50GB",
        cpus_per_task=4,
        time="00:30:00",
    script:
        "workflow/scripts/health.py"


    

