import argparse
import base64
import html
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List

import ftfy
import mediapy as media
import numpy as np
import openai
import regex as re
import tqdm
from PIL import Image

SYSTEM_PROMPT2 = """You are an expert evaluator of visual reasoning capabilities in videos involving robotic. Your role is to examine sampled frames from a video and assess the performance based on two key aspects: object permanence, action following, and final check. The intended action for the video will be provided for reference.

Guidelines for Evaluation:

• Carefully analyze the given frames to assess whether objects persist logically across time, even when occluded or temporarily out of view (object permanence).
• Check whether the agent's behavior in the frames aligns with the provided intended action sequence, maintaining proper order and logical continuity (action following).
• Use the rubric provided below to assign a numeric score from 1 (poor) to 5 (excellent) for each aspect.
• Justify each score with a concise explanation based on visual evidence from the frames. Mention specific object behaviors or inconsistencies observed.
• If frames are ambiguous or insufficient, state the limitations and base your score on best-available evidence.
• Maintain objective, grounded language. Do not speculate beyond what is observable in the frames.
• The agent should only do one action, interact with one object, and do not use both arms at the same time in a single segment. 

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

Final Check:
The agent should only do one action at a time. If the agent does multiple actions at once, it should be scored as 0. Otherwise, it should be scored as 1.

Example:

Intended actions: ["Pick up the red apple with the left arm", "Place the red apple held in the left arm on the blue plate"]
Sampled frames: [image blocks]
Evaluation:
- Object Permanence Score: 4
  Justification: The red apple is visible in all relevant frames except for a brief occlusion during placement.
- Action Following Score: 5
  Justification: The apple is successfully picked up, with each motion clearly aligned to the intended plan.
- Final Check Score: 1
  Justification: The agent only uses its right arm to interact with one object, which is picking up the apple. The whole process only contains one action, which is correct.

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WM_inference on a video with specified actions"
    )
    parser.add_argument("--demo_info", type=str, required=True,
                        help="Path to a JSONL file; each line must have keys video_path, output_path, actions")
    parser.add_argument("--fps", type=int, default=20,
                        help="FPS for input reading and output writing")
    parser.add_argument("--best_of_n", type=int, default=1,
                        help="How many variants to generate per job")
    parser.add_argument("--segment_save_dir", type=str, default=None,
                        help="Directory to save intermediate video segments. Defaults to a temporary directory.")
    parser.add_argument("--cosmos_version", type=str, choices=["cosmos1", "cosmos2"], default="cosmos1",
                        help="Which cosmos version to use: cosmos1 or cosmos2")
    return parser.parse_args()


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


def concatenate_videos(segments: list[Path], out_path: Path) -> None:
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

def chat_with_retries(max_retries: int = 3, **create_kwargs) -> openai.types.chat.ChatCompletion:
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
    base_filename: str,
    action_idx: int,
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
    out_path = out_dir / f"gpt_segement_score_{base_filename}_{action_idx}.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(log_entry) + "\n")

    # Return the index in variant_paths, not the variant number
    return variant_numbers.index(best)


def main() -> None:
    args = parse_args()

    # Process all jobs in the demo_info file (should be just one job per worker)  
    with open(args.demo_info, 'r') as f:
        all_jobs = [json.loads(line) for line in f if line.strip()]
    
    # Initialize generator based on cosmos_version
    if args.cosmos_version == "cosmos1":
        from cosmos1 import Cosmos1
        print(f"Initializing Cosmos1 generator...")
        generator = Cosmos1()
        save_video_func = lambda video, path, fps: media.write_video(str(path), video, fps=fps)
    elif args.cosmos_version == "cosmos2":
        from cosmos2 import Cosmos2
        from imaginaire.utils.io import save_image_or_video
        print(f"Initializing Cosmos2 generator...")
        generator = Cosmos2()
        save_video_func = lambda video, path, fps: save_image_or_video(video, str(path), fps=fps)
    else:
        raise ValueError(f"Unsupported cosmos_version: {args.cosmos_version}")
    
    
    python_fps = args.fps
    for job_idx, job_data in enumerate(tqdm.tqdm(all_jobs, desc="Processing Jobs")):
        episode_name = job_data.get("episode", f"job_{job_idx+1}")
        video_input_path = job_data["video_path"]

        if job_data.get("output_path"):
            output_video_path_template = Path(job_data["output_path"])
        else:
            output_dir = Path.cwd() / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video_path_template = output_dir / f"output_{episode_name}.mp4"
        actions_sequence = job_data["actions"]

        print(f"[{episode_name}] Processing video: {video_input_path}")


        output_videos_dir = output_video_path_template.parent
        output_videos_dir.mkdir(parents=True, exist_ok=True)
        base_output_filename = output_video_path_template.stem

        best_segments: List[Path] = []
        best_segments_metadata = []
        for action_idx, action_instruction in enumerate(tqdm.tqdm(actions_sequence, desc=f"  Actions")):
            cleaned_action_instruction_base = prompt_clean(action_instruction)
            final_prompt_for_api = f"{cleaned_action_instruction_base}"
            variant_paths: List[Path] = []
            variant_numbers: List[int] = []

            # All segments will be saved in the final output directory by default
            save_dir = Path(args.segment_save_dir) if args.segment_save_dir else output_videos_dir
            save_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{episode_name}, Action {action_idx+1}] Generating {args.best_of_n} variants in '{save_dir}'")
            
            for variant_num in range(args.best_of_n):
                try:
                    video = generator.generate_video(
                        prompt=final_prompt_for_api,
                        input_path=video_input_path
                    )
                    out_name = f"{base_output_filename}_variant{variant_num+1}.mp4"
                    out_path = Path(save_dir) / out_name
                    save_video_func(video, out_path, python_fps)
                    segment_file_path_str = str(out_path)
                    
                    if not segment_file_path_str:
                        segment_file_path_str = str(save_dir / "fallback_single.mp4")

                
                    # Rename the file to be unique for this variant AND action
                    p = Path(segment_file_path_str)
                    new_path = p.with_name(f"{base_output_filename}_variant{variant_num+1}{p.suffix}")
                    p.rename(new_path)
                    variant_paths.append(new_path)
                    variant_numbers.append(variant_num + 1)  # Track actual variant number (1-based)

                except Exception as e:
                    print(f"[{episode_name}, Action {action_idx+1}, Variant {variant_num+1}] ERROR: Exception during inference_round: {e}")
                    continue
            if not variant_paths:
                # If we failed to generate any variants for this action, the job is unrecoverable.
                raise RuntimeError(f"Failed to generate any variants for action {action_idx+1} ('{action_instruction}'). Aborting job.")

            # Select best segment
            best_idx = select_best_segment(action_instruction, variant_paths, variant_numbers, base_output_filename, action_idx, output_videos_dir)
            best_segment_path = variant_paths[best_idx]
            best_segments.append(best_segment_path)
            video_input_path = str(best_segment_path)


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
        # Concatenate best segments if all actions succeeded
        if best_segments:
            final_output_path = None
            if len(actions_sequence) > 1:
                final_output_path = output_videos_dir / f"{base_output_filename}_concat.mp4"
                print(f"[{episode_name}] Concatenating {len(best_segments)} best segments for: {final_output_path}")
                try:
                    concatenate_videos(best_segments, final_output_path)
                    print(f"[{episode_name}] Successfully saved concatenated video.")
                except Exception as e:
                    print(f"[{episode_name}] ERROR: Failed to concatenate videos: {e}")
                    continue
            else: # Single-action job: rename the best segment to the final output name
                final_output_path = output_videos_dir / f"{base_output_filename}_final.mp4"
                print(f"[{episode_name}] Renaming best segment to: {final_output_path}")
                try:
                    best_segments[0].rename(final_output_path)
                    print(f"[{episode_name}] Successfully saved final video.")
                except Exception as e:
                    print(f"[{episode_name}] ERROR: Failed to rename best segment: {e}")
                    continue

            # Write metadata.json for the final output
            metadata_path = output_videos_dir / f"{base_output_filename}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'actions': best_segments_metadata,
                    'final_output': str(final_output_path)
                }, f, indent=2)

if __name__ == "__main__":
    main()
