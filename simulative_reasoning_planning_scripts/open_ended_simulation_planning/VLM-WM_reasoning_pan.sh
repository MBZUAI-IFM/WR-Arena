#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=open_ended_pan
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --qos=wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128

export PROJECT_ROOT=$SLURM_SUBMIT_DIR
export OPENAI_API_KEY="Your_api_key"

eval "$(conda shell.bash hook)"
conda activate video-api

export PYTHONPATH="thirdparty/pan:$PYTHONPATH"
srun python simulative_reasoning_planning_scripts/open_ended_simulation_planning/VLM-WM_reasoning_pan.py \
  --jobs_jsonl datasets/simulative_reasoning_planning/open_ended_simulation_planning/samples.jsonl \
  --api_endpoint "http://10.24.2.126:8000" \
  --best_of_n 3 \
  --fps 20
