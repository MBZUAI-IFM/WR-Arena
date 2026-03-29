#!/bin/bash
#SBATCH --job-name=gc_eval
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --partition=main
#SBATCH --qos=wm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128

export PROJECT_ROOT=$SLURM_SUBMIT_DIR

# Set the model whose generated videos you want to evaluate
MODEL_NAME="pan"

VIDEOS_ROOT="${PROJECT_ROOT}/outputs/generation_consistency_eval/${MODEL_NAME}"
PREPARED_ROOT="${PROJECT_ROOT}/outputs/generation_consistency_eval/${MODEL_NAME}_eval"
DATASET_JSON="${PROJECT_ROOT}/datasets/generation_consistency_eval/samples.json"

eval "$(conda shell.bash hook)"
conda activate worldscore   # env with worldscore pip-installed + thirdparty deps

cd ${PROJECT_ROOT}
mkdir -p logs

echo "Preparing WorldScore directory structure for $MODEL_NAME ..."
python generation_consistency_eval_scripts/prepare_worldscore_dirs.py \
    --videos_root  $VIDEOS_ROOT \
    --dataset_json $DATASET_JSON \
    --output_root  $PREPARED_ROOT

echo "Running evaluation ..."
python generation_consistency_eval_scripts/run_evaluate_multiround.py \
    --model_name      $MODEL_NAME \
    --visual_movement static \
    --runs_root       $PREPARED_ROOT \
    --num_jobs        24 \
    --use_slurm       True \
    --slurm_partition $SLURM_JOB_PARTITION \
    --slurm_qos       wm \
    --slurm_job_name  gc_eval_${MODEL_NAME}

echo "Evaluation completed. Results in ${PREPARED_ROOT}/worldscore_output/"
