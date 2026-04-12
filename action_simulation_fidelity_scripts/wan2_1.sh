#!/bin/bash
#SBATCH --job-name=asf_wan2_1
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=main
#SBATCH --qos=wm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --exclusive

export PROJECT_ROOT=$SLURM_SUBMIT_DIR
export DATA_PATH="${PROJECT_ROOT}/datasets/action_simulation_fidelity_subset"

MODEL_NAME="wan2_1"
DATASET_JSON="${PROJECT_ROOT}/datasets/action_simulation_fidelity_subset/samples_subset.json"

eval "$(conda shell.bash hook)"
conda activate wan2_1

cd ${PROJECT_ROOT}
mkdir -p logs

echo "Starting generation: $MODEL_NAME"

srun --label bash -c "
    torchrun \
        --standalone \
        --nnodes=1 \
        --nproc_per_node=$SLURM_GPUS_ON_NODE \
        world_generators/generate_videos.py \
        --model_name $MODEL_NAME \
        --prompt_set $DATASET_JSON \
        --gen_rank \$SLURM_NODEID \
        --gen_world_size $SLURM_JOB_NUM_NODES
"

echo "All nodes completed"