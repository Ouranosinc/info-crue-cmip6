from pathlib import Path
import xscen as xs
import pandas as pd
import os


# Load configuration
configfile: "config/config.yml"
configfile: "config/paths.yml"


#TODO: put the right sim, careful with PCIC members vs Ouranos members
sim_ids=[
    'CMIP6_ScenarioMIP_CMCC_CMCC-ESM2_ssp370_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp245_r1i1p1f1', #7442
    #  'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp370_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp245_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp370_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp245_r4i1p1f1', #  PCIC member
    #  'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp370_r4i1p1f1',# PCIC member
    #  'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp245_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp370_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp245_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp370_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp245_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp370_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_NIMS-KMA_KACE-1-0-G_ssp245_r2i1p1f1',#  PCIC member
    #  'CMIP6_ScenarioMIP_NIMS-KMA_KACE-1-0-G_ssp370_r2i1p1f1', #PCIC member
    # 'CMIP6_ScenarioMIP_CCCma_CanESM5_ssp126_r1i1p1f1', # start 8444
    # 'CMIP6_ScenarioMIP_CCCma_CanESM5_ssp245_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CCCma_CanESM5_ssp370_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CCCma_CanESM5_ssp585_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp126_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CAS_FGOALS-g3_ssp585_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp126_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_CSIRO_ACCESS-ESM1-5_ssp585_r1i1p1f1',
    #  'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp126_r4i1p1f1', #  PCIC member
    #  'CMIP6_ScenarioMIP_EC-Earth-Consortium_EC-Earth3_ssp585_r4i1p1f1',# PCIC member
    #  'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp126_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_IPSL_IPSL-CM6A-LR_ssp585_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp126_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MIROC_MIROC6_ssp585_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp126_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_MRI_MRI-ESM2-0_ssp585_r1i1p1f1', 
    #  'CMIP6_ScenarioMIP_NIMS-KMA_KACE-1-0-G_ssp126_r2i1p1f1',#  PCIC member
    #  'CMIP6_ScenarioMIP_NIMS-KMA_KACE-1-0-G_ssp585_r2i1p1f1', #  PCIC member
]



regions= list(config['custom']['regions'].keys())

wdir= Path(config['paths']['workdir'])
finaldir= Path(config['paths']['finaldir'])

#TODO: replace the QC if other region


rule all:
    input: 
        expand(finaldir/"health/{sim_id}_health.zarr.zip",sim_id=sim_ids),
        expand(finaldir/"diagnostics/QC/{sim_id}/{sim_id}_QC_imp.zarr.zip",sim_id=sim_ids)

rule makeref:
    output: 
        default=finaldir/ "reference/{region_name}_default.zarr.zip",
        noleap=finaldir/ "reference/{region_name}_noleap.zarr.zip",
        day360=finaldir/ "reference/{region_name}_360_day.zarr.zip",
    params:
        n_workers=2,
        mem="250GB",
        cpus_per_task=4,
        time="00:10:00",
    script: "workflow/scripts/makeref.py"

rule extractregrid:
    input: 
        noleap=finaldir/ "reference/{region_name}_noleap.zarr.zip",
    output: temp(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr.zip")
    params:
        n_workers=2,
        mem="10GB",
        cpus_per_task=4,
        time="00:10:00",
    script:
        "workflow/scripts/extract-regrid.py"

rule train:
    input:
        sim= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr.zip",
        ref_noleap= finaldir/"reference/{region_name}_noleap.zarr.zip",
        ref_360_day= finaldir/"reference/{region_name}_360_day.zarr.zip",
    output: temp(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_training.zarr.zip"),
    params:
        n_workers=10,
        mem="50GB",
        cpus_per_task=12,
        time="00:50:00",
    script:
        "workflow/scripts/train.py"


rule adjust:
    input:
        sim= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_regridded.zarr.zip",
        ref_noleap= finaldir/"reference/{region_name}_noleap.zarr.zip",
        ref_360_day= finaldir/"reference/{region_name}_360_day.zarr.zip",
        train= wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_training.zarr.zip",
    output: temp(wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_adjusted.zarr.zip"),
    params:
        mem="40GB",
        cpus_per_task=1,
        time="6:00:00",
    script:
        "workflow/scripts/adjust.py"


rule clean_up:
    input: wdir/"{sim_id}_{region_name}/{sim_id}_{region_name}_adjusted.zarr.zip"
    output: temp(finaldir/"final_regions/{region_name}/day_{sim_id}_{region_name}.zarr.zip")
    params:
        mem="5GB",
        cpus_per_task=1,
        time="00:30:00",
    script:
        "workflow/scripts/clean_up.py"



def final_path(id):
    path='test'
    path= xs.build_path(
        data=pd.Series(
            dict(zip(['mip_era','activity','institution','source', 'experiment','member'],id.split('_'))
     )|dict(
        domain=config['custom']['full_region']['name'],
        format='zarr.zip',
         variable='foo',
         type='simulation',
         processing_level='biasadjusted',
         bias_adjust_project=config['custom']['bias_adjust_project'],
         bias_adjust_institution=config['custom']['bias_adjust_institution'],
         version=config['custom']['version'],
         frequency='day',
         xrfreq='D',
         date_start=config['custom']['sim_period'][0],
         date_end=config['custom']['sim_period'][1])))
    return str(os.path.dirname(os.path.dirname(path)))

#sim_id HAS to be in output, so can't use only params
rule concat_scen:
    input: expand(finaldir/"final_regions/{region_name}/day_{{sim_id}}_{region_name}.zarr.zip",region_name=regions)
    output: 
        pr=finaldir/"{path}/pr/pr_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip", 
        tasmax=finaldir/"{path}/tasmax/tasmax_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip",
        tasmin=finaldir/"{path}/tasmin/tasmin_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip",
        dtr=finaldir/"{path}/dtr/dtr_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip", 
    params:
        path=lambda wildcards: final_path(wildcards.sim_id),
        n_workers=2,
        mem="60GB", 
        cpus_per_task=4,
        time="00:15:00",
    script:
        "workflow/scripts/concat.py"


rule health:
    input:
        pr=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/pr/pr_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
        tasmax=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/tasmax/tasmax_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
        tasmin=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/tasmin/tasmin_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
        dtr=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/dtr/dtr_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
    output: 
        finaldir/"health/{sim_id}_health.zarr.zip"
    params:
        n_workers=2,
        mem="50GB",
        cpus_per_task=4,
        time="00:30:00",
    script:
        "workflow/scripts/health.py"



rule diag_ref:
    output: 
        ref=finaldir/ "reference/QC_default.zarr.zip",
        prop=finaldir/"diagnostics/QC/prop_ref.zarr.zip"
    params:
        n_workers=2,
        mem="50GB",
        cpus_per_task=4,
        time="00:10:00",
    script:
        "workflow/scripts/diag_ref.py"

rule diag:
    input:
        ref=finaldir/ "reference/QC_default.zarr.zip",
        ref_prop=finaldir/"diagnostics/QC/prop_ref.zarr.zip",
        scen_pr=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/pr/pr_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
        scen_tasmax=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/tasmax/tasmax_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
        scen_tasmin=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/tasmin/tasmin_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
        scen_dtr=lambda wildcards: finaldir/(f"{final_path(wildcards.sim_id)}"+"/dtr/dtr_day_MBCn-EM_v10_{sim_id}_QC-EMDNA_1951-2100.zarr.zip"),
    output: 
        sim_prop=finaldir/"diagnostics/QC/{sim_id}/{sim_id}_QC_sim-prop.zarr.zip",
        sim_meas=finaldir/"diagnostics/QC/{sim_id}/{sim_id}_QC_sim-meas.zarr.zip",
        scen_prop=finaldir/"diagnostics/QC/{sim_id}/{sim_id}_QC_scen-prop.zarr.zip",
        scen_meas=finaldir/"diagnostics/QC/{sim_id}/{sim_id}_QC_scen-meas.zarr.zip",
        imp=finaldir/"diagnostics/QC/{sim_id}/{sim_id}_QC_imp.zarr.zip",
    params:
        n_workers=2,
        mem="50GB",
        cpus_per_task=4,
        time="02:00:00",
    script:
        "workflow/scripts/diag.py"


    

