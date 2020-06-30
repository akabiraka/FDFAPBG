## this must be run from directory where run.py exists.
## --workdir is not used in this file.

#!/usr/bin/sh

#SBATCH --job-name=recon_distmap_using_ae
#SBATCH --qos=csqos
##SBATCH --workdir=/scratch/akabir4/project_dir
#SBATCH --output=/scratch/akabir4/project_dir/outputs/logs/recon_distmap_using_ae_log_1-%N-%j.output
#SBATCH --error=/scratch/akabir4/project_dir/outputs/logs/recon_distmap_using_ae_log_1-%N-%j.error
#SBATCH --mail-user=<akabir4@gmu.edu>
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --gres=gpu:1
#SBATCH --partition=gpuq
#SBATCH --mem=32G

python full_run_recon_distmap_using_ae.py
