#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=f_parallel
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128

export PROJECT_ROOT=$SLURM_SUBMIT_DIR
export OPENAI_API_KEY="Your_api_key"

eval "$(conda shell.bash hook)"
conda activate cosmos-predict2

cd ${PROJECT_ROOT}
mkdir -p logs

srun python simulative_reasoning_planning_scripts/structured_simulation_planning/VLM-WM_reasoning_cosmos.py \
  --jobs_jsonl datasets/simulative_reasoning_planning/structured_simulation_planning/samples.jsonl \
  --best_of_n 3 \
  --fps 16 \
  --cosmos_version cosmos2 \

