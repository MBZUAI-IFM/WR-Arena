#!/bin/bash
#SBATCH --job-name=step_wise_cosmos2
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
export SAMPLES_FILE="${PROJECT_ROOT}/datasets/simulative_reasoning_planning/step_wise_simulation_subset/samples.jsonl"
export OUTPUT_PATH="${PROJECT_ROOT}/outputs/simulative_reasoning_planning/step_wise_simulation"

export MODEL_NAME=${MODEL_NAME:-"cosmos2"}
export GPU_PER_NODE=${GPU_PER_NODE:-8}

eval "$(conda shell.bash hook)"
conda activate cosmos-predict2

cd ${PROJECT_ROOT}
mkdir -p logs
mkdir -p ${OUTPUT_PATH}

srun --label bash -c "
    python simulative_reasoning_planning_scripts/step_wise_simulation_scripts/generate_videos.py \
        --samples_file $SAMPLES_FILE \
        --output_path $OUTPUT_PATH \
        --model_name "$MODEL_NAME" \
        --node_id \$SLURM_PROCID \
        --total_nodes $SLURM_NTASKS \
        --gpu_per_node $GPU_PER_NODE
"