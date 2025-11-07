#!/bin/bash

#SBATCH --partition=Brain3090
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8

#SBATCH --job-name=ue_r_daniela

#SBATCH --output=/Brain/private/d25garci/default_stdout.log
#SBATCH --error=/Brain/private/d25garci/default_stderr.log
#SBATCH --time=10:00:00

# Set paths
home="/Brain/private/d25garci"

# Generate the logs
## First set the date
date=$(date +'%Y_%m_%d_%H_%M_%S')

## Then create the log directory 
log_dir="$home/runs/$date"
mkdir -p "$log_dir"

## Redirect output to your custom log files
exec > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"

# Activate the venv
source $home/.venv/bin/activate

# Test
srun nvidia-smi

srun python $home/train_yolo.py

# Procesar VALIDATION
#dataset_path="/Brain/public/datasets/BioDCASE_task2/biodcase_development_set/validation"
#echo "starting: $dataset_path"
#srun python $home/preprocess_data.py \
#   --dataset_path "$dataset_path" \
#   --train_mode

# Procesar TRAIN
#srun python $home/preprocess_data.py \
#  --dataset_path "/Brain/public/datasets/BioDCASE_task2/biodcase_development_set/train" \
#  --train_mode
