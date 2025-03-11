# info-crue-cmip6

Instructions: 
1) On narval, build a virtual env:

```bash
$ module load StdEnv/2023 gcc openmpi python/3.11 arrow/16.1.0 openmpi netcdf proj esmf/8.6.0 geos mpi4py/3.1.4 ipykernel/2023b scipy-stack/2023b
$ cd <PATH_ENV_DIR>
$ virtualenv --no-download ic6-mbcn-sm
$ source ic6-mbcn-sm/bin/activate
$ pip install --no-index --upgrade pip
$ pip install --no-index -r requirements.txt
$ echo "module load StdEnv/2023 gcc openmpi python/3.11 arrow/16.1.0 openmpi netcdf proj esmf/8.6.0 geos mpi4py/3.1.4 ipykernel/2023b scipy-stack/2023b" > ic6-mbcn-sm/bin/modules
```
 or just activate it: `pyact ic6-mbcn-sm`

2) Specify the `sim_id` wanted in the `Snakefile`. 

3) Create your own `config/paths.yml` based on `paths-template.yml`. #TODO: make template

4) Modify `config/config.yml` to reflect method wanted (correct tasmin or dtr, reference dataset, time grouping)

5) If needed, personalize the `simple/config.v8+.yaml` for the right slurm parameters.

6) Run the workflow:

```bash
$ snakemake --profile simple
```

7) Run `inspection.ipynb` to make sure all looks good.

Snakemake should build a dag that looks like this for 1 simulation: ![Texte alternatif](dag.png)
