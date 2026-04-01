#!/usr/bin/env python3
import argparse
import html
import numpy as np
import tqdm
import mediapy as media
from PIL import Image
from typing import List, Optional
import json
import os
import tempfile
import subprocess
from pathlib import Path
import ftfy
import regex as re
import base64
import io
import openai
from inference_connector import WM_inference

SYSTEM_PROMPT2 = """You are an expert evaluator of visual reasoning capabilities in videos involving robotic. Your role is to examine sampled frames from a video and assess the performance based on two key aspects: object permanence and action following. The intended action for the video will be provided for reference.

Guidelines for Evaluation:

• Carefully analyze the given frames to assess whether objects persist logically across time, even when occluded or temporarily out of view (object permanence).
• Check whether the agent's behavior in the frames aligns with the provided intended action sequence, maintaining proper order and logical continuity (action following).
• Use the rubric provided below to assign a numeric score from 1 (poor) to 5 (excellent) for each aspect.
• Justify each score with a concise explanation based on visual evidence from the frames. Mention specific object behaviors or inconsistencies observed.
• If frames are ambiguous or insufficient, state the limitations and base your score on best-available evidence.
• Maintain objective, grounded language. Do not speculate beyond what is observable in the frames.

Rubric:
Object Permanence:
1 - Object disappears illogically; major continuity errors.
2 - Object inconsistently reappears or changes without explanation.
3 - Object is briefly occluded or shifted but mostly consistent.
4 - Object is consistently tracked with minor visual noise.
5 - Object is clearly and continuously represented across all frames.

Action Following:
1 - Agent behavior deviates entirely from the intended actions.
2 - One or more major steps are skipped, reversed, or incorrect.
3 - Rough adherence to the sequence with minor errors.
4 - Sequence is mostly correct with only subtle deviations.
5 - Precise match to the intended sequence in both order and execution.

Example:

Intended actions: ["Move the robot arm to the left of the red circle", "Move the yellow pentagon into the yellow heart"]
Sampled frames: [image blocks]
Evaluation:
- Object Permanence Score: 2
  Justification: The red circle is occluded in some frames, and another red circle pops up from nowhere in the video.
- Action Following Score: 5
  Justification: The yellow pentagon is pushed and moved continuously towards the yellow heart, and the motion clearly aligned to the intended plan.

Do not return code or refer to APIs. Output only the two scores (each out of 5) and brief explanations under clear headings. Do not include vague references or generalities. Base all judgments strictly on what is observable in the provided frames."""

function_schema = {
    "name": "evaluate_variant",
    "description": "Scores one video variant on object permanence and action following",
    "parameters": {
        "type": "object",
        "properties": {
            "object_permanence": {"type": "integer"},
            "action_following": {"type": "integer"},
            "final_check": {"type": "integer"},
            "total": {"type": "integer"},
            "rationale": {"type": "string", "description": "A brief justification for the scores."}
        },
        "required": ["object_permanence", "action_following", "final_check", "total", "rationale"]
    }
}

VIDEO_SIZE = (832, 480)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run WM_inference on a video with specified actions"
    )
    parser.add_argument("--demo_info", type=str, required=True,
                        help="Path to a JSONL file; each line must have keys video_path, output_path, actions")
    parser.add_argument("--fps", type=int, default=20,
                        help="FPS for input reading and output writing")
    parser.add_argument("--best_of_n", type=int, default=1,
                        help="How many variants to generate per job")
    parser.add_argument("--api_endpoint", type=str, default="http://10.24.1.42:8000",
                        help="API endpoint for WM_inference service")
    parser.add_argument("--segment_save_dir", type=str, default=None,
                        help="Directory to save intermediate video segments. Defaults to a temporary directory.")
    parser.add_argument("--guidance_scale", type=float, default=4.0,
                        help="Classifier-free guidance scale for generation.")
    return parser.parse_args()


def resize_image(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resizes and pads a PIL image to the target_size.
    """
    w0, h0 = img.size
    t_width, t_height = target_size
    scale = max(t_width / w0, t_height / h0) # Changed to max to ensure it fits, will pad later
    nw, nh = int(w0 * scale), int(h0 * scale)

    # Resize
    img_resized = img.resize((nw, nh), Image.LANCZOS)

    # Pad
    new_img = Image.new("RGB", target_size, (0,0,0)) # Pad with black
    pad_left = (t_width - nw) // 2
    pad_top = (t_height - nh) // 2
    new_img.paste(img_resized, (pad_left, pad_top))
    return new_img


def load_first_frame(video_path: str) -> Image.Image:
    """
    Read a video and return the first frame as a PIL RGB image.
    If the path is an image, load and return it directly.
    """
    # Check if the path is an image file by extension
    image_exts = (".png", ".jpg", ".jpeg", ".webp")
    if video_path.lower().endswith(image_exts):
        img = Image.open(video_path).convert("RGB")
        return img
    else:
        video = media.read_video(video_path)
        first = video[0]
        if first.dtype != np.uint8:
            first = (np.clip(first, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(first)


def concatenate_videos(segments: list[Path], out_path: Path):
    """Concatenate MP4s via ffmpeg CLI."""
    list_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    for seg in segments:
        list_file.write(f"file '{seg.resolve()}'\n") # Use absolute paths for ffmpeg
    list_file.close()
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', list_file.name, '-c', 'copy', str(out_path.resolve())
        ], check=True, capture_output=True) # capture_output for better error reporting
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg concatenation: {e}")
        print(f"ffmpeg stdout: {e.stdout.decode()}")
        print(f"ffmpeg stderr: {e.stderr.decode()}")
        raise
    finally:
        os.remove(list_file.name)


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))

def encode_frame_to_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def chat_with_retries(max_retries: int = 3, **create_kwargs):
    """
    Wrapper around openai.chat.completions.create that retries
    up to max_retries times if the model "apologizes" or refuses.
    """
    resp = ""
    for attempt in range(1, max_retries + 1):
        last_resp = openai.chat.completions.create(**create_kwargs)
        msg = last_resp.choices[0].message
        content = msg.content or ""
        low = content.lower()
        if "sorry" in low and ("cannot assist" in low or "can't assist" in low):
            if attempt < max_retries:
                print(f"⚠️  Model refusal detected, retrying ({attempt}/{max_retries})…")
                continue

        # no refusal detected (or maxed out retries) → return this response
        return last_resp
    # if we fall out of the loop, return the last response anyway
    return resp

def select_best_segment(
    action: str,
    variant_paths: list[Path],
    variant_numbers: list[int],
    episode: str,
    round: int,
    out_dir: Path,
    system_prompt: str = SYSTEM_PROMPT2
) -> int:
    # Prepare log and score container
    log_entry = {"action": action, "scores": [], "chosen_by_code": None}
    if len(variant_paths) == 1:
        return 0
    for i, (vp, variant_num) in enumerate(zip(variant_paths, variant_numbers)):
        # Read video and sample 6 frames evenly
        vid = media.read_video(str(vp))
        idxs = np.linspace(0, len(vid) - 1, num=6, dtype=int)
        images = []
        for j in idxs:
            b64 = encode_frame_to_b64(vid[j])
            images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"}
            })

        # Build messages for this single variant
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Intended action: \"{action}\""},
                {"type": "text", "text": f"Variant #{variant_num} frames:"},
                *images
            ]},
            {"role": "user", "content": (
                "Score this single variant and provide a brief justification. Respond via the function call "
                "`evaluate_variant` with properties object_permanence (1–5), "
                "action_following (1–5), final_check (0-1),total = sum of all, and rationale."
            )}
        ]

        # Call GPT with function-calling
        resp = chat_with_retries(
            max_retries=3,
            model="o3",
            messages=messages,
            #temperature=0,
            functions=[function_schema],
            function_call={"name": "evaluate_variant"}
        )

        # Extract the scores from the function_call
        msg = resp.choices[0].message
        args = json.loads(msg.function_call.arguments)
        obj_perm = args["object_permanence"]
        act_fol  = args["action_following"]
        final_check = args["final_check"]
        total    = args["total"]
        rationale = args.get("rationale", "No rationale provided.")

        # Record with actual variant number
        log_entry["scores"].append({
            "variant": variant_num,
            "object_permanence": obj_perm,
            "action_following": act_fol,
            "final_check": final_check,
            "total": total,
            "rationale": rationale
        })

    # Pick the best variant (highest total, tie → lowest variant number)
    best = min(
        log_entry["scores"],
        key=lambda x: (-x["total"], x["variant"])
    )["variant"]
    log_entry["chosen_by_code"] = best

    # Write out a JSONL log
    out_path = out_dir / f"gpt_segement_score_{episode}_{round}.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Return the index in variant_paths, not the variant number
    return variant_numbers.index(best)

def main():
    args = parse_args()

    # SLURM parallel slicing
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    with open(args.demo_info, 'r') as f:
        all_jobs = [json.loads(line) for line in f if line.strip()]
    jobs_for_this_rank = [job for idx, job in enumerate(all_jobs) if idx % world_size == rank]
    if not jobs_for_this_rank:
        print(f"[Rank {rank}/{world_size}] No jobs assigned, exiting.")
        return

    # Initialize WM_inference client
    print(f"[Rank {rank}] Initializing WM_inference with API endpoint: {args.api_endpoint}")
    wm_instance = WM_inference(api_endpoint=args.api_endpoint)

    # Explicitly cast potentially problematic float types from args
    python_fps = args.fps
    python_guidance_scale = float(args.guidance_scale)
    denoising_steps = 50
    for job_idx, job_data in enumerate(tqdm.tqdm(jobs_for_this_rank, desc=f"Rank {rank} Processing Jobs")):
        video_input_path = job_data["video_path"]
        round_num = job_data.get("round_num", 1)
        prev_state_id = job_data.get("prev_state_id")
        prev_video_id = job_data.get("prev_video_id")

        if job_data.get("output_path"):
            output_video_path_template = Path(job_data["output_path"])
        else:
            output_dir = Path.cwd() / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video_path_template = output_dir / f"output_{job_idx+1}.mp4"
        actions_sequence = job_data["actions"]

        # Load the frame needed for this generation round.
        # For round 1, it's the initial frame of the whole sequence.
        # For subsequent rounds, it's the last frame of the previous best segment.
        print(f"[Rank {rank}, Job {job_idx+1}] Loading frame from: {video_input_path}")
        try:
            initial_frame_pil = load_first_frame(video_input_path)
            resized_initial_frame_pil = resize_image(initial_frame_pil, VIDEO_SIZE)
            print(f"[Rank {rank}, Job {job_idx+1}] Frame loaded and resized.")
        except Exception as e:
            print(f"[Rank {rank}, Job {job_idx+1}] ERROR: Failed to load/resize frame for {video_input_path}: {e}")
            continue # Skip to next job

        if round_num > 1:
            print(f"[Rank {rank}, Job {job_idx+1}] Continuing generation from a previous state (Round {round_num}).")

        output_videos_dir = output_video_path_template.parent
        output_videos_dir.mkdir(parents=True, exist_ok=True)
        base_output_filename = output_video_path_template.stem

        # New: Per-action best segment selection
        best_segments: List[Path] = []
        best_segments_metadata = []  # To record metadata for each action
        current_prev_state_id: Optional[str] = prev_state_id
        current_prev_video_id: Optional[str] = prev_video_id
        current_frame_pil = resized_initial_frame_pil
        current_round_num = round_num
        for action_idx, action_instruction in enumerate(tqdm.tqdm(actions_sequence, desc=f"  Actions")):
            cleaned_action_instruction_base = prompt_clean(action_instruction)
            final_prompt_for_api = f"{cleaned_action_instruction_base}"
            variant_paths: List[Path] = []
            variant_numbers: List[int] = []  # Track actual variant numbers
            variant_state_ids: List[Optional[str]] = []
            variant_video_ids: List[Optional[str]] = []

            # All segments will be saved in the final output directory by default
            save_dir = Path(args.segment_save_dir) if args.segment_save_dir else output_videos_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Rank {rank}, Job {job_idx+1}, Action {action_idx+1}] Generating {args.best_of_n} variants in '{save_dir}'")
            
            # Preserve the state at the start of the action's variants
            action_start_state_id = current_prev_state_id
            action_start_video_id = current_prev_video_id
            
            # Get the unique ID passed from the calling script to ensure globally unique session IDs
            unique_id_base = job_data.get("unique_id", f"job{job_idx}_action{action_idx}")

            for variant_num in range(args.best_of_n):
                try:
                    if current_round_num == 1:
                        segment_file_path_str, next_video_id, next_state_id = wm_instance.inference_round(
                            current_round_num,
                            final_prompt_for_api,
                            image=current_frame_pil,
                            session_id=f"{unique_id_base}",
                            save_dir=str(save_dir),
                            fps=python_fps,
                            denoising_steps=denoising_steps,
                            guidance_scale=python_guidance_scale,
                            random_seed=variant_num,
                            prev_state_id=action_start_state_id, # Always use the state from the START of the action
                            prev_video_id=action_start_video_id, # Always use the video_id from the START of the action                            
                        )
                        print(f"round 1: segment_file_path_str:{segment_file_path_str}, next_video_id:{next_video_id}, next_state_id:{next_state_id}")
                    else:
                        print(f"round_num:{current_round_num}, prompt:{final_prompt_for_api}, session_id:{unique_id_base}, save_dir:{save_dir}, fps:{python_fps}, denoising_steps:{denoising_steps}, guidance_scale:{python_guidance_scale}, random_seed:{variant_num}, prev_state_id:{action_start_state_id}, prev_video_id:{action_start_video_id}")
                        segment_file_path_str, next_video_id, next_state_id = wm_instance.inference_round(
                            current_round_num,
                            final_prompt_for_api,
                            session_id=f"{unique_id_base}",
                            save_dir=str(save_dir),
                            fps=python_fps,
                            denoising_steps=denoising_steps,
                            guidance_scale=python_guidance_scale,
                            random_seed=variant_num,
                            prev_state_id=action_start_state_id,
                            prev_video_id=action_start_video_id,
                        )
                        print(f"round {current_round_num}: segment_file_path_str:{segment_file_path_str}, next_video_id:{next_video_id}, next_state_id:{next_state_id}")
                    
                    if not segment_file_path_str:
                        # build a fallback path if we got None or empty string
                        segment_file_path_str = str(save_dir / f"round-{current_round_num}_single.mp4")
                        print(f"[Warning] inference_round returned no path; using fallback: {segment_file_path_str}")

                    
                    # Rename the file to be unique for this variant AND action
                    p = Path(segment_file_path_str)
                    #new_path = p.with_name(f"{base_output_filename}_action{action_idx}_variant{variant_num+1}{p.suffix}")
                    new_path = p.with_name(f"{base_output_filename}_round{current_round_num}_variant{variant_num+1}{p.suffix}")
                    p.rename(new_path)
                    variant_paths.append(new_path)
                    variant_numbers.append(variant_num + 1)  # Track actual variant number (1-based)
                    variant_video_ids.append(next_video_id)
                    variant_state_ids.append(next_state_id)

                except Exception as e:
                    print(f"[Rank {rank}, Job {job_idx+1}, Action {action_idx+1}, Variant {variant_num+1}] ERROR: Exception during inference_round: {e}")
                    continue
            if not variant_paths:
                # If we failed to generate any variants for this action, the job is unrecoverable.
                raise RuntimeError(f"Failed to generate any variants for action {action_idx+1} ('{action_instruction}'). Aborting job.")

            # Select best segment
            best_idx = select_best_segment(action_instruction, variant_paths, variant_numbers, base_output_filename, action_idx, output_videos_dir)
            best_segment_path = variant_paths[best_idx]
            best_segments.append(best_segment_path)

            # Set the state for the NEXT action to be the state from the BEST variant of THIS action
            current_prev_state_id = variant_state_ids[best_idx]
            current_prev_video_id = variant_video_ids[best_idx]

            # Read the select_best_segment log for this action
            gpt_score_log_path = output_videos_dir / f"gpt_segement_score_{base_output_filename}_{action_idx}.jsonl"
            scores = None
            rationale = None
            if gpt_score_log_path.exists():
                try:
                    with open(gpt_score_log_path, 'r') as f:
                        # Only one log entry per action, so just read the last line
                        last_line = list(f)[-1]
                        log_entry = json.loads(last_line)
                        # Find the score entry for the best variant
                        best_variant_choice = log_entry.get('chosen_by_code')
                        if best_variant_choice is not None:
                            for score_data in log_entry.get('scores', []):
                                if score_data.get('variant') == best_variant_choice:
                                    rationale = score_data.get('rationale')
                        scores = log_entry.get('scores', None)
                except Exception as e:
                    print(f"[Warning] Could not read scores/rationale from {gpt_score_log_path}: {e}")
            best_segments_metadata.append({
                'action': action_instruction,
                'best_segment': str(best_segment_path),
                'all_variants': [str(p) for p in variant_paths],
                'best_idx': best_idx,
                'scores': scores,
                'rationale': rationale
            })
            # Update current_frame_pil for next action
            try:
                frames_data = media.read_video(str(best_segment_path))
                if isinstance(frames_data, (list, np.ndarray)) and len(frames_data) > 0:
                    last_frame_np = frames_data[-1]
                    current_frame_pil = Image.fromarray(last_frame_np)
                else:
                    print(f"[Warning] Could not extract last frame from {best_segment_path}; reusing previous frame.")
            except Exception as _e:
                print(f"[Warning] Error reading video '{best_segment_path}' to update initial frame: {_e}")
            current_round_num += 1
        # Concatenate best segments if all actions succeeded
        if best_segments:
            final_output_path = None
            if len(actions_sequence) > 1:
                final_output_path = output_videos_dir / f"{base_output_filename}_concat.mp4"
                print(f"[Rank {rank}, Job {job_idx+1}] Concatenating {len(best_segments)} best segments for: {final_output_path}")
                try:
                    concatenate_videos(best_segments, final_output_path)
                    print(f"[Rank {rank}, Job {job_idx+1}] Successfully saved concatenated video.")
                except Exception as e:
                    print(f"[Rank {rank}, Job {job_idx+1}] ERROR: Failed to concatenate videos: {e}")
                    continue
            else: # Single-action job: rename the best segment to the final output name
                final_output_path = output_videos_dir / f"{base_output_filename}_final.mp4"
                print(f"[Rank {rank}, Job {job_idx+1}] Renaming best segment to: {final_output_path}")
                try:
                    best_segments[0].rename(final_output_path)
                    print(f"[Rank {rank}, Job {job_idx+1}] Successfully saved final video.")
                except Exception as e:
                    print(f"[Rank {rank}, Job {job_idx+1}] ERROR: Failed to rename best segment: {e}")
                    continue

            # Write metadata.json for the final output
            metadata_path = output_videos_dir / f"{base_output_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'actions': best_segments_metadata,
                    'final_output': str(final_output_path),
                    'final_state_id': current_prev_state_id,
                    'final_video_id': current_prev_video_id
                }, f, indent=2)

if __name__ == "__main__":
    main()
