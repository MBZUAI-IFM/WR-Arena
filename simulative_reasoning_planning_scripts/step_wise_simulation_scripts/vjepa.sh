#!/bin/bash
#SBATCH --job-name=step_wise_vjepa
#SBATCH --partition=main       
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --qos=wm
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=8    
#SBATCH --gpus-per-task=1      
#SBATCH --cpus-per-task=14
set -eo pipefail

# --- Conda initialise & activate -------------------------------------------
set +u
eval "$(conda shell.bash hook)"
conda activate vjepa2
set -u

export OUTPUT_DIR=outputs/simulative_reasoning_planning/step_wise_simulation/vjepa2/subset.jsonl
export INPUT_FILE=datasets/simulative_reasoning_planning/step_wise_simulation_subset/samples.jsonl
export TOTAL_SPLITS=$SLURM_NTASKS

srun --ntasks=$TOTAL_SPLITS --ntasks-per-node=8 bash -c '
  set -eo pipefail
  RANK=$SLURM_PROCID
  echo "[Node $SLURM_NODEID] starting shard $RANK / '$TOTAL_SPLITS' on GPU $CUDA_VISIBLE_DEVICES …" >&2

  python simulative_reasoning_planning_scripts/step_wise_simulation_scripts/vjepa.py \
    --data_path "'$INPUT_FILE'" \
    --output_file "'$OUTPUT_DIR'" \
    --split_idx "$RANK" \
    --num_splits "'$TOTAL_SPLITS'"

  echo "[Node $SLURM_NODEID] shard $RANK finished." >&2
'
echo "Merging output shards…" >&2
cat $(dirname $OUTPUT_DIR)/subset_*.jsonl > $(dirname $OUTPUT_DIR)/subset.jsonl
echo "✅Done." >&2