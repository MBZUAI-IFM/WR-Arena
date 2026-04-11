import os
import json
import sys

vjepa_project_root = 'thirdparty/vjepa2'
if vjepa_project_root not in sys.path:
    sys.path.insert(0, vjepa_project_root)

import torch
import numpy as np
from PIL import Image
import mediapy as media
import io
from tqdm import tqdm
import argparse
from torch.nn import functional as F
from transformers import UMT5EncoderModel, AutoTokenizer
from torch.amp import autocast

from app.vjepa_droid.transforms import make_transforms
from src.models.ac_predictor import vit_ac_predictor
from src.models.vision_transformer import vit_giant_xformers

# --- Model Initialization ---
def initialize_models():
    """Initializes and returns the V-JEPA encoder, predictor, text encoder, and tokenizer."""
    print("Initializing models...")
    
    # Text Encoder and Tokenizer
    text_encoder = UMT5EncoderModel.from_pretrained(
        "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="tokenizer")
    
    # V-JEPA Encoder
    encoder = vit_giant_xformers(
        img_size=256,
        patch_size=16,
        num_frames=512,
        tubelet_size=2,
        uniform_power=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        use_rope=True,
    ).eval().to("cuda")

    # V-JEPA Predictor
    predictor = vit_ac_predictor(
        img_size=256,
        patch_size=16,
        num_frames=512,
        tubelet_size=2,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=1024,
        depth=24,
        is_frame_causal=True,
        num_heads=16,
        uniform_power=True,
        use_rope=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
    ).eval().to("cuda")

    # Load state dicts
    # Note: Replace with the correct path to your checkpoint file
    state_dict_path = "thirdparty/vjepa2/checkpoints/e275.pt"
    state_dict = torch.load(state_dict_path)
    
    renamed_encoder_state_dict = {k.replace("module.", ""): v for k, v in state_dict["encoder"].items()}
    renamed_predictor_state_dict = {k.replace("module.", ""): v for k, v in state_dict["predictor"].items()}
    
    encoder.load_state_dict(renamed_encoder_state_dict)
    predictor.load_state_dict(renamed_predictor_state_dict)
    
    encoder = encoder.to(torch.bfloat16)
    predictor = predictor.to(torch.bfloat16)
    
    print("Models initialized successfully.")
    return encoder, predictor, text_encoder, tokenizer

# --- Data Loading ---
def load_evaluation_data(jsonl_file_path):
    """Loads tasks from a JSONL file."""
    tasks = []
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks

# --- Inference ---
def forward_target(c, encoder):
    """
    Computes the feature representation for a single input frame.
    The single frame is repeated to simulate a 2-frame sequence for the video model.
    """
    with torch.no_grad():
        # Input c shape: (B, T, C, H, W). Expect B=1, T=1.
        # Permute to (B, C, T, H, W) for the model.
        c = c.permute(0, 2, 1, 3, 4)
        # Repeat the frame dimension to create a 2-frame sequence.
        c = c.repeat(1, 1, 2, 1, 1)
        
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            # Encoder output h shape: (B, N_patches*T, D) -> (1, 512, D)
            h = encoder(c)
            
        # We only need the representation for the first frame (256 patches).
        h = h[:, :256, :]
        
        # The original notebook's view/flatten logic was the source of the shape errors.
        # The output from slicing is already in the correct (B, N, D) shape.
        h = F.layer_norm(h, (h.size(-1),))
        return h

def step_predictor(_z, _t, predictor):
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        _z = predictor(_z, _t)
    _z = F.layer_norm(_z, (_z.size(-1),))
    return _z

def run_vjepa_rollout(first_frame_rep, actions, text_encoder, tokenizer, predictor, n_steps):
    """
    Rolls out a single action plan and returns the final predicted frame representation.
    """
    plan_str = " ".join(actions)
    text_inputs = tokenizer(plan_str, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            encoded_text = text_encoder(**text_inputs).last_hidden_state

    z = first_frame_rep
    for n in range(n_steps):
        _z_nxt = step_predictor(z, encoded_text, predictor)[:, -256:]
        z = torch.cat([z, _z_nxt], dim=1)

    # Return the final predicted latent representation
    last_frame_rep = z.reshape(1, n_steps + 1, 256, -1)[:, -1, :, :]
    return last_frame_rep

# --- Main Pipeline ---
def main():
    parser = argparse.ArgumentParser(description="V-JEPA Single Plan Evaluation Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the evaluation results.")
    parser.add_argument("--rollout_steps", type=int, default=20, help="Number of steps to roll out each plan.")
    parser.add_argument("--split_idx", type=int, default=None, help="Index of the current processing split.")
    parser.add_argument("--num_splits", type=int, default=None, help="Total number of processing splits.")
    args = parser.parse_args()

    # Initialize
    encoder, predictor, text_encoder, tokenizer = initialize_models()
    transform = make_transforms(crop_size=256, motion_shift=False)

    # Load Data
    all_eval_tasks = load_evaluation_data(args.data_path)
    
    # Split tasks for parallel processing if specified
    if args.num_splits is not None and args.split_idx is not None:
        if len(all_eval_tasks) < args.num_splits:
            print(f"Warning: Number of tasks ({len(all_eval_tasks)}) is less than number of splits ({args.num_splits}). Some workers will have no tasks.")
        
        tasks_per_split = np.array_split(all_eval_tasks, args.num_splits)
        
        # Calculate the global starting index for this split
        start_index = sum(len(s) for s in tasks_per_split[:args.split_idx])
        
        eval_tasks = tasks_per_split[args.split_idx]
    else:
        eval_tasks = all_eval_tasks
        start_index = 0

    # Modify output file path for each split to avoid race conditions
    output_file_path = args.output_file
    if args.num_splits is not None and args.split_idx is not None:
        base, ext = os.path.splitext(output_file_path)
        output_file_path = f"{base}_part_{args.split_idx}{ext}"

    results = []
    all_scores = []

    for i, task in enumerate(tqdm(eval_tasks, desc="Evaluating Single Plans")):
        try:
            # Construct the full, absolute path to the images
            image_path_str = task["source"]

            # 1. Load, convert to RGB, and explicitly resize to 256x256 to match model's expected input size.
            initial_frame_raw = np.array(Image.open(image_path_str).convert("RGB").resize((256, 256)))
            initial_frame_tensor = transform(np.expand_dims(initial_frame_raw, axis=0)).unsqueeze(0).to("cuda").to(torch.bfloat16)

            # 2. If the transform pipeline converted the image to grayscale (1 channel), expand it back to 3.
            if initial_frame_tensor.shape[2] == 1:
                initial_frame_tensor = initial_frame_tensor.repeat(1, 1, 3, 1, 1)

            # Get feature representations for start and end frames
            first_frame_rep = forward_target(initial_frame_tensor, encoder)
            
            # Run the rollout for the given plan
            predicted_final_rep = run_vjepa_rollout(
                first_frame_rep, [task["prompt"]], text_encoder, tokenizer, predictor, args.rollout_steps
            )
            
            # Compare predicted final state with all choices
            pooled_predicted = predicted_final_rep.mean(dim=1)
            
            similarity_scores = []
            for choice_path in task["image_choices"]:
                goal_frame_raw = np.array(Image.open(choice_path).convert("RGB").resize((256, 256)))
                goal_frame_tensor = transform(np.expand_dims(goal_frame_raw, axis=0)).unsqueeze(0).to("cuda").to(torch.bfloat16)

                if goal_frame_tensor.shape[2] == 1:
                    goal_frame_tensor = goal_frame_tensor.repeat(1, 1, 3, 1, 1)
                
                goal_frame_rep = forward_target(goal_frame_tensor, encoder)
                pooled_goal = goal_frame_rep.mean(dim=1)
                
                score = F.cosine_similarity(pooled_predicted.to(torch.float32), pooled_goal.to(torch.float32)).item()
                similarity_scores.append(score)

            best_choice_score = max(similarity_scores)
            best_choice_index = similarity_scores.index(best_choice_score) + 1 # 1-indexed
            all_scores.append(best_choice_score)

            result = {
                "task_index": start_index + i,
                "source_path": task["source"],
                "image_choices": task["image_choices"],
                "prompt": task["prompt"],
                "ground_truth_answer": task["best"],
                "similarity_scores": similarity_scores,
                "best_choice_index": best_choice_index,
                "best_choice_score": best_choice_score,
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing task {start_index + i}: {e}")
            continue

    # Save results to output file
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(output_file_path, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')

    # Final Summary
    average_score = np.mean(all_scores) if all_scores else 0
    summary = f"\n--- Single Plan Evaluation Summary ---\n"
    summary += f"Total Tasks Evaluated: {len(results)}\n"
    summary += f"Average Similarity Score: {average_score:.4f}\n"
    summary += f"Results saved to: {output_file_path}\n"
    
    summary_path = output_file_path.replace(".jsonl", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
        
    print(summary)


if __name__ == "__main__":
    main() 