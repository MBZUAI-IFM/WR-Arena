# WR-Arena
A diagnostic tool and a guideline for advancing next-generation world models capable of robust understanding, forecasting, and purposeful action.

## Table of Contents
- [Action Simulation Fidelity](#action-simulation-fidelity)
- [Simulative Reasoning & Planning](#simulative-reasoning--planning)
- [Smoothness Evaluation](#smoothness-evaluation)


## Action Simulation Fidelity

This section evaluates the ability of video generation models to simulate actions faithfully based on given prompts.

### Supported Models

We evaluate multiple state-of-the-art video generation models, categorized into local models and API-based models:

#### Local Models (require local setup and checkpoints):
1. **Cosmos-Predict1-14B-Video2World** - [GitHub](https://github.com/nvidia-cosmos/cosmos-predict1)
2. **Cosmos-Predict2-14B-Video2World** - [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2)
3. **WAN 2.1-I2V-14B** - [GitHub](https://github.com/Wan-Video/Wan2.1)
4. **WAN 2.2-I2V-A14B** - [GitHub](https://github.com/Wan-Video/Wan2.2)

#### API-based Models (require API access):
5. **Gen-3** 
6. **KLING**
7. **MiniMax-Hailuo**
8. **PAN** - Our proprietary model (requires custom endpoint, not yet publicly released)
9. **Sora 2** 
10. **Veo 3**

### Setup for Local Models

#### 1. Create Conda Environments

For each model, create a dedicated conda environment and follow the installation instructions from their respective repositories:

- **Cosmos-Predict1**: Follow setup instructions at https://github.com/nvidia-cosmos/cosmos-predict1 (environment name: `cosmos-predict1`)
- **Cosmos-Predict2**: Follow setup instructions at https://github.com/nvidia-cosmos/cosmos-predict2 (environment name: `cosmos-predict2`)
- **WAN 2.1**: Follow setup instructions at https://github.com/Wan-Video/Wan2.1 (environment name: `wan2_1`)
- **Wan 2.2**: Follow setup instructions at https://github.com/Wan-Video/Wan2.2 (environment name: `wan2_2`)

#### 2. Download Model Checkpoints

Download the corresponding checkpoints for each model and place them in the respective directories:

```
thirdparty/
├── cosmos-predict1/checkpoints/
├── cosmos-predict2/checkpoints/
├── wan2_1/checkpoints/
└── wan2_2/checkpoints/
```

#### 3. Run Video Generation

Execute the generation scripts using SLURM for local models:

```bash
# Local Models
sbatch action_simulation_fidelity_scripts/cosmos1.sh
sbatch action_simulation_fidelity_scripts/cosmos2.sh
sbatch action_simulation_fidelity_scripts/wan2_1.sh
sbatch action_simulation_fidelity_scripts/wan2_2.sh
```

### Setup for API-based Models

For models that use API calls:

#### 1. Create API Environment

```bash
conda create -n video-api python=3.10 -y
conda activate video-api
pip install -r requirements_api.txt
```

#### 2. Run API-based Generation

Execute the generation scripts for API-based models:

```bash
# API-based Models
bash action_simulation_fidelity_scripts/gen3.sh
bash action_simulation_fidelity_scripts/kling.sh
bash action_simulation_fidelity_scripts/minimax.sh
bash action_simulation_fidelity_scripts/pan.sh
bash action_simulation_fidelity_scripts/sora2.sh
bash action_simulation_fidelity_scripts/veo3.sh
```

### Evaluation

After generating videos for each model, evaluate their action simulation fidelity using GPT-4o:

```bash
python action_simulation_fidelity_scripts/action_simulation_fidelity_eval.py \
    --openai_api_key YOUR_OPENAI_API_KEY \
    --base_path outputs/action_simulation_fidelity/MODEL_NAME \
    --dataset_json datasets/action_simulation_fidelity_subset/samples_subset.json \
    --save_name MODEL_NAME
```

**Examples:**

```bash
# Evaluate Cosmos-Predict1
python action_simulation_fidelity_scripts/action_simulation_fidelity_eval.py \
    --openai_api_key YOUR_KEY \
    --base_path outputs/action_simulation_fidelity/cosmos1 \
    --dataset_json datasets/action_simulation_fidelity_subset/samples_subset.json \
    --save_name cosmos1

# Evaluate PAN
python action_simulation_fidelity_scripts/action_simulation_fidelity_eval.py \
    --openai_api_key YOUR_KEY \
    --base_path outputs/action_simulation_fidelity/pan \
    --dataset_json datasets/action_simulation_fidelity_subset/samples_subset.json \
    --save_name pan
```

Results will be saved in `outputs/action_simulation_fidelity/MODEL_NAME/MODEL_NAME_results.json`.

## Simulative Reasoning & Planning

This section evaluates video generation models on their ability to perform simulative reasoning and planning for robotic tasks.

### Fine-tuning Setup

Before evaluation, models need to be fine-tuned on specific datasets:

### Fine-tuning Setup

Both Cosmos-Predict1 and Cosmos-Predict2 models need to be fine-tuned on specific datasets for each evaluation task:

# Open-ended Simulation Planning

- **Dataset**: Agibot World Colosseo – “A large-scale manipulation platform for scalable and intelligent embodied systems” (Bu et al., 2025)  
- **Models to fine-tune**: Cosmos-Predict1, Cosmos-Predict2  
- **Purpose**: Enables open-ended reasoning about robotic manipulation tasks  

# Structured Simulation Planning

- **Dataset**: Language Table – “Interactive language: Talking to robots in real time” (Lynch et al., 2023)  
- **Models to fine-tune**: Cosmos-Predict1, Cosmos-Predict2  
- **Purpose**: Enables structured reasoning with specific action constraints

### Fine-tuning Process

1. **For each model (Cosmos-Predict1 and Cosmos-Predict2)**, fine-tune on **both datasets**:
   - Fine-tune on Agibot dataset for open-ended simulation planning
   - Fine-tune on Language Table dataset for structured simulation planning

2. **Fine-tuning methods**: Follow the fine-tuning instructions in the respective model repositories:
   - [Cosmos-Predict1 Fine-tuning](https://github.com/nvidia-cosmos/cosmos-predict1)
   - [Cosmos-Predict2 Fine-tuning](https://github.com/nvidia-cosmos/cosmos-predict2)

3. **Checkpoint replacement**: Replace the original checkpoints with your fine-tuned checkpoints in the `thirdparty/*/checkpoints/` directories.

### Evaluation Scripts

#### Open-ended Simulation Planning

Run the open-ended simulation planning evaluation:

```bash
# Cosmos-Predict1
sbatch simulative_reasoning_planning_scripts/open_ended_simulation_planning/VLM-WM_reasoning_cosmos1.sh

# Cosmos-Predict2
sbatch simulative_reasoning_planning_scripts/open_ended_simulation_planning/VLM-WM_reasoning_cosmos2.sh
```

**Evaluation**: After execution, check the generated results in `outputs/simulative_reasoning_planning/open_ended_simulation_planning/[task_name]/[model_name]/[task_name]_refined.json`. Evaluate whether the action list successfully achieves the specified goal to determine task completion success.

#### Structured Simulation Planning

For structured simulation planning, different tasks have different maximum action steps (5 or 10):

**Maximum 5 Actions:**
```bash
# Cosmos-Predict1
sbatch simulative_reasoning_planning_scripts/structured_simulation_planning/VLM-WM_reasoning_cosmos1_max_action_5.sh

# Cosmos-Predict2
sbatch simulative_reasoning_planning_scripts/structured_simulation_planning/VLM-WM_reasoning_cosmos2_max_action_5.sh
```

**Maximum 10 Actions:**
```bash
# Cosmos-Predict1
sbatch simulative_reasoning_planning_scripts/structured_simulation_planning/VLM-WM_reasoning_cosmos1_max_action_10.sh

# Cosmos-Predict2
sbatch simulative_reasoning_planning_scripts/structured_simulation_planning/VLM-WM_reasoning_cosmos2_max_action_10.sh
```

**Evaluation**: Similar to open-ended simulation planning, check the generated results in `outputs/simulative_reasoning_planning/structured_simulation_planning/[task_name]/[model_name]/[task_name]_refined.json`. Analyze the action sequence to determine whether the model successfully completed the structured task within the specified action limits.

## Smoothness Evaluation

This section evaluates the temporal smoothness of multi-round generated videos using optical flow. Consecutive frame pairs are processed with SEA-RAFT to compute velocity and acceleration magnitudes, which are combined into a smoothness score (`vmag × exp(−λ × amag)`).

### Dataset

`datasets/smoothness_eval/samples.json` contains 100 photorealistic outdoor scenes, each with a 10-round sequential prompt list. Reference images are not bundled — set `IMAGE_ROOT` in the generation scripts to point to your local copy of the WorldScore-Dataset.

### Setup: Download SEA-RAFT Checkpoint

```bash
wget https://huggingface.co/datasets/memcpy/SEA-RAFT/resolve/main/Tartan-C-T-TSKH-spring540x960-M.pth \
    -O thirdparty/SEA-RAFT/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth
```

### Step 1: Generate Videos

Scripts for all supported models are in `smoothness_eval_scripts/`. Example using PAN:

```bash
# Edit IMAGE_ROOT inside the script first, then:
bash smoothness_eval_scripts/pan.sh
```

Generated videos are saved under `outputs/smoothness_eval/pan/{instance_id}/rounds/`.

### Step 2: Compute Smoothness Scores

```bash
python smoothness_eval_scripts/compute_smoothness_scores.py \
    --videos_dir outputs/smoothness_eval/pan \
    --output_dir outputs/smoothness_eval/pan_scores \
    --raft_ckpt thirdparty/SEA-RAFT/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth \
    --num_workers 4
```

Per-instance results are written to `outputs/smoothness_eval/pan_scores/{instance_id}/smoothness.json`. An aggregate `summary.json` is written once all instances are scored. For multi-node SLURM evaluation, set `MODEL_NAME` in `smoothness_eval_scripts/eval.sh` and run `sbatch smoothness_eval_scripts/eval.sh`.