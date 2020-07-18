#!/usr/bin/sh

## this must be run from directory where run.py exists.
## --workdir is not used in this file.

#SBATCH --job-name=data_generation
#SBATCH --qos=csqos
##SBATCH --workdir=/scratch/akabir4/project_dir
#SBATCH --output=/scratch/akabir4/FDFAPBG/outputs/logs/data_generation_log-%N-%j.output
#SBATCH --error=/scratch/akabir4/FDFAPBG/outputs/logs/data_generation_log-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --partition=all-HiPri
## do not use --nodes if not MPI 
##SBATCH --nodes=5 
#SBATCH --mem=32000MB

## python full_run_recon_distmap_using_ae.py
##python ADMM/test_ADMM.py
python datasets/data_generator.py