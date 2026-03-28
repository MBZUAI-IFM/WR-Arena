#!/bin/bash
export PROJECT_ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
export DATA_PATH="${PROJECT_ROOT}/datasets/action_simulation_fidelity_subset"
export MINIMAX_API_KEY="Your_api_key"
export MINIMAX_GROUP_ID="Your_group_id"

MODEL_NAME="minimax"
DATASET_JSON="${PROJECT_ROOT}/datasets/action_simulation_fidelity_subset/samples_subset.json"
NUM_JOBS=4

cd ${PROJECT_ROOT}

timestamp=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/${MODEL_NAME}_parallel_${timestamp}"

mkdir -p ${LOG_DIR}

eval "$(conda shell.bash hook)"
conda activate video-api

echo "📊 Starting $NUM_JOBS parallel API jobs..."

for i in $(seq 0 $((NUM_JOBS-1))); do
    echo "🔄 Starting Job $i..."
    python world_generators/generate_videos.py \
    --model_name $MODEL_NAME \
    --prompt_set $DATASET_JSON \
    --no-slurm \
    --num-jobs $NUM_JOBS \
    --batch-index $i \
    > "${LOG_DIR}/${i}.out" 2>&1 &
done

echo "📋 All $NUM_JOBS jobs started in parallel"
echo "📝 Check progress: tail -f ${LOG_DIR}/*.out"
echo "⏳ Waiting for all jobs to complete..."

# Wait for all background jobs to finish
wait

echo "✅ All API jobs completed!"
echo "📁 Results saved"