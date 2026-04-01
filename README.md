# WR-Arena
A diagnostic tool and a guideline for advancing next-generation world models capable of robust understanding, forecasting, and purposeful action.

<p align="center">
    <img src="assets/evaluation_benchmark.png" width="850"/>
</p>

## Table of Contents
- [Action Simulation Fidelity](#action-simulation-fidelity)
- [Smoothness Evaluation](#smoothness-evaluation)
- [Generation Consistency Evaluation](#generation-consistency-evaluation)
- [Simulative Reasoning & Planning](#simulative-reasoning--planning)

## 🔹 Action Simulation Fidelity

This section evaluates the ability of world models to simulate actions faithfully based on multi-round prompts.

### Supported Models

We evaluate multiple state-of-the-art world models, categorized into local models and API-based models:

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

### Setup for Local Models

#### 1. Create Conda Environments

For each model, create a dedicated conda environment and follow the installation instructions from their respective repositories:

- **Cosmos-Predict1**: Follow setup instructions at [GitHub](https://github.com/nvidia-cosmos/cosmos-predict1) (env name: `cosmos-predict1`)
- **Cosmos-Predict2**: Follow setup instructions at [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2) (env name: `cosmos-predict2`)
- **WAN 2.1**: Follow setup instructions at [GitHub](https://github.com/Wan-Video/Wan2.1) (env name: `wan2_1`)
- **WAN 2.2**: Follow setup instructions at [GitHub](https://github.com/Wan-Video/Wan2.2) (env name: `wan2_2`)

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
# Local Models. Example:
sbatch action_simulation_fidelity_scripts/cosmos1.sh
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
# API-based Models. Example:
bash action_simulation_fidelity_scripts/gen3.sh
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

Results will be saved in `outputs/action_simulation_fidelity/MODEL_NAME/MODEL_NAME_results.json`.

## 🔹 Smoothness Evaluation

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

## 🔹 Generation Consistency Evaluation

This section evaluates video generation models on 7 aspects of multi-round generation consistency using the [WorldScore](https://github.com/haoyi-duan/WorldScore) benchmark framework (MIT License).

| Aspect | Metric | Key dependency |
|---|---|---|
| `camera_control` | camera reprojection error | **DROID-SLAM** |
| `object_control` | object detection score | **GroundingDINO + SAM2** |
| `content_alignment` | CLIP score | CLIP |
| `3d_consistency` | reprojection error | **DROID-SLAM** |
| `photometric_consistency` | optical flow AEPE | SEA-RAFT |
| `style_consistency` | Gram matrix distance | VGG |
| `subjective_quality` | CLIP-IQA+, MUSIQ | QAlign, MUSIQ |

Each score is a **list** of per-round values rather than a single scalar. The bold aspects require heavy thirdparty dependencies (see WorldScore's own setup guide). The remaining four aspects (`content_alignment`, `photometric_consistency`, `style_consistency`, `subjective_quality`) can be run on any GPU without those dependencies.

### Setup

#### 1. Add WorldScore as a submodule and install it

```bash
git submodule update --init thirdparty/WorldScore
pip install -e thirdparty/WorldScore
```

Follow [WorldScore's setup instructions](https://github.com/haoyi-duan/WorldScore) for the thirdparty dependencies (DROID-SLAM, GroundingDINO, SAM2) if you need all 7 aspects.

#### 2. Install WR-Arena patches

```bash
bash generation_consistency_eval_scripts/install_patches.sh
```

This copies the modified evaluator into the WorldScore submodule.

### Step 1: Generate Videos

Edit `IMAGE_ROOT` in the script to point to your local WorldScore-Dataset, then run:

```bash
bash generation_consistency_eval_scripts/pan.sh
```

Generated videos are saved under `outputs/generation_consistency_eval/pan/`.

### Step 2: Prepare WorldScore Directory Structure

```bash
python generation_consistency_eval_scripts/prepare_worldscore_dirs.py \
    --videos_root  outputs/generation_consistency_eval/pan \
    --dataset_json datasets/generation_consistency_eval/samples.json \
    --output_root  outputs/generation_consistency_eval/pan_eval
```

### Step 3: Evaluate

```bash
python generation_consistency_eval_scripts/run_evaluate_multiround.py \
    --model_name      pan \
    --visual_movement static \
    --runs_root       outputs/generation_consistency_eval/pan_eval \
    --num_jobs        24 \
    --use_slurm       True \
    --slurm_partition main \
    --slurm_qos       wm
```

Results are written to `outputs/generation_consistency_eval/pan_eval/worldscore_output/worldscore_multiround.json`. For SLURM-based end-to-end runs, set `MODEL_NAME` in `generation_consistency_eval_scripts/eval.sh` and run `sbatch generation_consistency_eval_scripts/eval.sh`

## 🔹 Simulative Reasoning & Planning

This section evaluates video generation models on their ability to perform simulative reasoning and planning for robotic tasks. Part of the code has been uploaded and the rest is currently being prepared.

### Fine-tuning Setup

Both **Cosmos-Predict1** and **Cosmos-Predict2** models need to be fine-tuned on specific datasets for the evaluation tasks:

| Task Type | Dataset | Models to Fine-tune | Purpose |
|-----------|---------|-------------------|---------|
| Open-ended Simulation Planning | Agibot World Colosseo – “A large-scale manipulation platform for scalable and intelligent embodied systems” (Bu et al., 2025) | Cosmos-Predict1, Cosmos-Predict2 | Enables open-ended reasoning about robotic manipulation tasks |
| Structured Simulation Planning | Language Table – “Interactive language: Talking to robots in real time” (Lynch et al., 2023) | Cosmos-Predict1, Cosmos-Predict2 | Enables structured reasoning with specific action constraints |

**Fine-tuning process**:

1. Follow the respective model repository instructions for fine-tuning:
   - [Cosmos-Predict1 Fine-tuning](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/examples/post-training_diffusion_video2world.md)  
   - [Cosmos-Predict2 Fine-tuning](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/post-training_video2world.md)  

2. Replace the original checkpoints with your fine-tuned versions in the `thirdparty/*/checkpoints/` directories.

---

### Evaluation & Results

Run the evaluation scripts and check the generated results for both open-ended and structured simulation planning.

```bash
# Example: Cosmos-Predict1
sbatch simulative_reasoning_planning_scripts/open_ended_simulation_planning/VLM-WM_reasoning_cosmos1.sh
sbatch simulative_reasoning_planning_scripts/structured_simulation_planning/VLM-WM_reasoning_cosmos1.sh
```

**Result Checking**:

After execution, check the results in the following paths:

- **Open-ended**:  
  `outputs/simulative_reasoning_planning/open_ended_simulation_planning/[task_name]/[model_name]/[task_name]_refined.json`  

- **Structured**:  
  `outputs/simulative_reasoning_planning/structured_simulation_planning/[task_name]/[model_name]/[task_name]_refined.json`  

Analyze the action sequences to determine whether the models successfully completed the tasks.
