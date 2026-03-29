#!/bin/bash
#SBATCH --job-name=sm_eval
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

# Set the model whose generated videos you want to evaluate
MODEL_NAME="cosmos-predict1"

VIDEOS_DIR="${PROJECT_ROOT}/outputs/smoothness_eval/${MODEL_NAME}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/smoothness_eval/${MODEL_NAME}_scores"
RAFT_CKPT="${PROJECT_ROOT}/thirdparty/SEA-RAFT/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth"

eval "$(conda shell.bash hook)"
conda activate cosmos-predict1  # any env with torch, opencv, numpy

cd ${PROJECT_ROOT}
mkdir -p logs

echo "Starting smoothness evaluation: $MODEL_NAME"

srun --label bash -c "
    python smoothness_eval_scripts/compute_smoothness_scores.py \
        --videos_dir $VIDEOS_DIR \
        --output_dir $OUTPUT_DIR \
        --raft_ckpt $RAFT_CKPT \
        --num_workers $SLURM_GPUS_ON_NODE \
        --rank \$SLURM_PROCID \
        --world_size $SLURM_NTASKS
"

echo "Evaluation completed"
