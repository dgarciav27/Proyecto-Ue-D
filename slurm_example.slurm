#!/bin/bash

#SBATCH --partition=Brain3090
#SBATCH --nodelist=sl-tp-br-524
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8

#SBATCH --job-name=ue_r_daniela

#SBATCH --output=/Brain/private/d25garci/runs/default_output/default_stdout.log
#SBATCH --error=/Brain/private/d25garci/runs/default_output/default_stderr.log
#SBATCH --time=01:00:00

# Set paths
home="/Brain/private/d25garci"
project_dir="$home"

# Generate the logs
## First set the date
date=$(date +'%Y_%m_%d_%H_%M_%S')

## Then create the log directory 
log_dir="$home/runs/test_setup/$date"
mkdir -p "$log_dir"

## Redirect output to your custom log files
exec > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"

# Activate the venv
source $home/.venv/bin/activate

# Test
srun nvidia-smi

srun python -c "print('Hello World!')"

srun python $project_dir/main.py