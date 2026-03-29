#!/bin/bash
export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
export IMAGE_ROOT="/path/to/WorldScore-Dataset"  # set to your local image root
export END_POINT="pan_endpoint"
export OPENAI_KEY="Your_api_key"

MODEL_NAME="pan"
DATASET_JSON="${PROJECT_ROOT}/datasets/generation_consistency_eval/samples.json"
NUM_JOBS=4

cd ${PROJECT_ROOT}

timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${MODEL_NAME}_parallel_${timestamp}"

mkdir -p ${LOG_DIR}

eval "$(conda shell.bash hook)"
conda activate video-api

echo "Starting $NUM_JOBS parallel API jobs..."

for i in $(seq 0 $((NUM_JOBS-1))); do
    echo "Starting Job $i..."
    python world_generators/generate_videos.py \
        --model_name $MODEL_NAME \
        --prompt_set $DATASET_JSON \
        --image_root $IMAGE_ROOT \
        --output_root outputs/generation_consistency_eval/$MODEL_NAME \
        --no-slurm \
        --num-jobs $NUM_JOBS \
        --batch-index $i \
        > "${LOG_DIR}/${i}.out" 2>&1 &
done

echo "All $NUM_JOBS jobs started in parallel"
echo "Check progress: tail -f ${LOG_DIR}/*.out"
echo "Waiting for all jobs to complete..."

wait

echo "All API jobs completed!"
echo "Results saved"
