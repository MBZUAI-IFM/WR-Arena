#!/bin/bash
#SBATCH --partition=main
#SBATCH --job-name=step_wise_pan
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --qos=wm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=116

eval "$(conda shell.bash hook)"
conda activate video-api

input_jsonl=datasets/simulative_reasoning_planning/step_wise_simulation_subset/samples_pan.jsonl

export PYTHONPATH="thirdparty/pan:$PYTHONPATH"
# use pan video generation script from open ended simulation planning
srun python simulative_reasoning_planning_scripts/open_ended_simulation_planning/variant_generation_pan.py \
    --demo_info $input_jsonl\
    --api_endpoint "http://10.24.0.14:8000" \
    --fps 20 \
    --best_of_n 1 \
    --guidance_scale 4.0 \
