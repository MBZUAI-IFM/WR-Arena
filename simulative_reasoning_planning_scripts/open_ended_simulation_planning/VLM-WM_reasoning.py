import argparse
import base64
import concurrent.futures as cf
import io
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import imageio
import mediapy as media
import numpy as np
import openai
import torch
from openai import OpenAIError
from PIL import Image

# System prompts for planning and refinement stages
SYSTEM_PROMPT = """You are an expert task planner for a stationary single-arm robot. Your role is to decompose high-level visual manipulation tasks into clear, actionable steps described purely in natural language.

Guidelines for Task Decomposition:

• Clearly state each next action the robot should perform, step-by-step.  
• Specify explicitly how the robot should interact with the object(s): grasp, lift, move, rotate, touch, or release.
• Refer to the target object using object attribute such as shape and color. Your target audience does not have common sense knowledge. 
• Use concise and clear directional terms: left, right, top (up), bottom (down), front (forward), and back (backward).  
• When possible, reference other objects clearly (e.g., "move the cube to the right of the blue sphere").  

Important Notes:

- Decomposition must be purely vision-based; no numeric coordinates.  
- Specify object parts (handle, cap, front, back) only if needed.  
- Consider the robot's previous actions (action history) when choosing the next optimal step.
- Generate one step at a time. Do not put two actions in a single step.
- Every object reference must be uniquely identifiable from the current image or the action history.
- Specify which object to operate. Use only facts. Do not use ambiguous description.
- The maximum steps breakdown should not exceed 5 steps.
- Do not use other objects included in the goal as a relative position reference. If asked to align objects in a line, don't simply say commands like 'place object A into a position that make objects align into a line' as this gives no information to the target position of the move. Instead you should mention this position clearly with respect to the background objects (e.g., table).

Example:

Task: "Place the red cup behind the green box."  
Action History: ["Pick up red cup by its handle."]
The user will then ask for multiple proposals. An appropriate response would be a list of possible next actions, like this:
- Move the red cup to behind the green box.
- Place the red cup on the table to the right of the green box.
- Move the red cup to the front of the green box.
Do not return code or directly call API functions. Do not print out Action History. Print content in the Next Step only and don't print any other contents. Never claim something like 'I'm sorry, I can't assist with that' or 'I could not see the image'. Do not use vague terms such as "other," "another," or "similar". Please describe each action clearly and specifically."""

SYSTEM_PROMPT1 = """You are an expert next-step selector for robotic task planning based on visual feedback. Your role is to examine a list of possible next actions together with their resulting last-frame images, then choose the single action that best advances the agent toward the specified goal given the action history.

Guidelines for Selection:

• Review the provided Goal and History to understand the current state of the world.  
• For each candidate action, analyze its resulting last frame to assess how much closer it brings the agent to the goal.  
• Prefer actions that directly reduce the gap to the goal, avoid regressions, and build logically on previous steps.  
• Ensure feasibility: the chosen action must be physically plausible.
• If the goal is already achieved—based on history or frame evidence—signal completion by setting \"finished\" to True in the JSON object.  
• Base your decision solely on observable visual evidence; do not infer hidden state or intentions beyond the frames.

Edge Cases:

• If visual feedback is ambiguous, pick the action that maximizes clear progress toward the goal.  
• If multiple actions appear equally good, choose the one that maintains consistency with past actions.  

When you reply, output **only** a JSON object with two keys:\n"
                "  • \"best_option\": an integer represent index of option you want to **confirm** and execute next \n"
                "  • \"rationale\": a string explaining why you want to choose the option\n"
                "  • \"finished\": a boolean value indicating whether the goal is already achieved from these actions (including the one you want to confirm and execute next) and the corresponding next state. \n"
                "No extra keys, no prose outside the JSON."

Example1:
Goal: "Align the red cup and the green box so that they are in a horizontal line."  
Proposed actions: ["Pick up the red cup", "Pick up the white bottle", "Pick up the blue plate"]
best_option: 1
rationale: As we can see in the image, in order to align the red cup and the green box, we need to pick up the red cup first. The other two options are picking up the unrelated objects.
finished: False

Example2:

Goal: "Rearrange the objects in the trays so that the number of objects in the left white tray is equal to the number of right blue tray."  
Proposed actions: ["Move the green object held in the arm to the left white tray.", "Move the green object held in the arm to the right blue tray.", "Pick up the red object in the right blue tray."]
best_option: 2
rationale: In the image we can see that the number of objects in the left white tray is 2 and the number of objects in the right blue tray is 3. So we need to move the green object to the right blue tray. The first option is conflicting the goal since it increases the number of objects in the left white tray. The third option is not a valid action since it is not a valid action to pick up the red object in the right blue tray.
finished: False

Example3:

Task: "Align the three objects on the table so that they are in a horizontal line."  
Proposed actions: ["Place the red cup held in the arm on the table behind the green box.", "Place the red cup held in the arm on the table in front of the green box.", "Place the red cup held in the arm on the table on top of green box."]
best_option: 1
rationale: The image shows the red cup already positioned behind the green box, forming a straight horizontal line with the green box and the blue plate. This configuration satisfies the task. The other two options would move the cup in front of or on top of the box, disrupting the achievement of the current goal.
finished: True
"""

SYSTEM_PROMPT3 = """You are an expert evaluator of visual reasoning capabilities in videos involving robotics. Your role is to examine last frames from a video and assess the performance based on action following. The intended action for the video will be provided for reference.

Guidelines for Evaluation:

• Check whether the agent's behavior in the last frame aligns with the final state of the provided intended action sequence, maintaining proper order and logical continuity (action following).  
• Justify each score with a concise explanation based on visual evidence from the frame. Mention specific object behaviors or inconsistencies observed.  
• If the frame is ambiguous or insufficient, state the limitations and base your score on best‐available evidence.  
• Maintain objective, grounded language. Do not speculate beyond what is observable in the last frame.

Example output format (no code, no APIs, just text):

Action: ["Pick up the red apple"]  
- Action Following Score: Yes 
  Justification: The left arm clearly grasps the apple, lifts it, and places it on the plate in the correct sequence."""

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent)
PANWAN_SCRIPT = Path(PROJECT_ROOT) / "simulative_reasoning_planning_scripts" / "open_ended_simulation_planning" / "variant_generation.py"

# Encode a frame as base64 PNG
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

def judge_action(
    vp: Path,
    action: str,
    model: str = "o3",
    system_prompt: str = SYSTEM_PROMPT3,
) -> str:
    """
    Sends the intended action + all sampled frames for a two‐score evaluation.
    Returns the raw textual output (scores + justifications).
    """
    vid = media.read_video(str(vp))
    last_frame = vid[-1]

    b64 = encode_frame_to_b64(last_frame)
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": f"Action: [\"{action}\"]\nLast frame attached below."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "auto"}},
                {"type": "text", "text": "Please output the judgement and brief justifications."}
            ]}
        ]

    # 3) Call model
    try:
        resp = chat_with_retries(
            max_retries=3,
            model=model,
            messages=messages,
        )
    except OpenAIError as e:
        return f"🔴 OpenAI API Error: {e}"
    except Exception as e:
        return f"🔴 Unexpected Error: {e}"

    # 4) Return raw reply
    return resp.choices[0].message.content.strip()

def get_full_action_list(
    episode: str,
    max_steps: int,
    initial_frame_path: Path,
    goal: str,
    history: list[str],
    best_of_n: int,
    out_dir: Path = Path("./variants"),
    fps: float = 4.0,
    model: str = "o3",
    cosmos_version: str = "cosmos1",
) -> Tuple[List[str], List[Path]]:
    action_list: list[str] = []
    current_input_path: Path = initial_frame_path  # This stays constant within each step
    proposed_trace = {}
    chosen_segments = []
    history = []

    for step in range(max_steps):
        # Use the same input for all actions in this step
        frame_path = current_input_path
        b64_frame = encode_frame_to_b64(load_last_frame(str(frame_path)))
        print(f"Step {step}: Using input {Path(frame_path).name} ({'image' if str(frame_path).endswith(('.png', '.jpg', '.jpeg')) else 'video'})")
        # 1) Propose 3 candidate actions
        resp = chat_with_retries(
            max_retries=3,
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": [
                    {"type": "text", "text": f"Goal: {goal}\n"},
                    {"type": "text", "text": f"History: {json.dumps(history)}\n\n"},
                    {"type": "text", "text": f"Last frame:\n\n"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_frame}", "detail": "auto"}},
                    {"type": "text", "text": "Please propose **3 possible next single steps** (just the action text)."},
                    {"type": "text", "text": "Do not use vague terms such as 'other,' 'another,' or 'similar'. Please describe each action clearly and specifically. For instance, say 'place the yellow package with the yellow packages in the right blue tray' instead of 'place the yellow package with the other yellow packages'"},
                    {"type": "text", "text": "Do not include two actions in a single step. For example, 'grasp the yellow package in the left blue tray and move it to the right blue tray' should be decomposed to two separate actions: 'grasp the yellow package in the left blue tray', 'move the yellow package to the right blue tray'."},
                    {"type": "text", "text": "Do not use other objects included in the goal as a relative position reference. If asked to align objects in a line, don't simply say commands like 'place pink object into a position that make objects align into a line' as this gives no information to the target position of the move. Instead you should mention this position clearly with respect to the background objects (e.g., table)."},
                    {"type": "text", "text": f"You should also review the action history, set \"finished\" to True if goal: {goal} is already achieved."},
                ]}
            ],
        )
        # extract up to 3 lines
        content = resp.choices[0].message.content
        clean = re.sub(r"```json[^\n]*\n", "", content, flags=re.IGNORECASE)
        clean = clean.replace("```", "")
        lines = [ln.strip("- ").strip()
         for ln in clean.splitlines()
         if ln.strip()]
        candidates = lines[:3]

        # 2) For each candidate, generate the best segment
        action_space: list[tuple[str, Path]] = []
        for idx, cand in enumerate(candidates):
            best_segment = generate_candidate_segments(
                filename=f"{episode}_round{step}_action{idx+1}",
                unique_id_for_session=episode,
                round_num_for_demo=step + 1,
                best_of_n=best_of_n,
                frame_path=frame_path,
                action=cand,
                out_dir=out_dir,
                fps=fps,
                cosmos_version=cosmos_version,
            )
            action_space.append((cand, best_segment))

        if not action_space:
            print(f"No viable actions at step {step}, stopping.")
            break
        print(action_space)
        # 3) Confirm which action to take next
        if len(action_space)==3:
            act_1, var_path_1 = action_space[0]
            act_2, var_path_2 = action_space[1]
            act_3, var_path_3 = action_space[2]
            b64_var_1 = encode_frame_to_b64(load_last_frame(str(var_path_1)))
            b64_var_2 = encode_frame_to_b64(load_last_frame(str(var_path_2)))
            b64_var_3 = encode_frame_to_b64(load_last_frame(str(var_path_3)))
            resp2 = chat_with_retries(
                max_retries=3,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT1},
                    {"role": "user",   "content": [
                        {"type": "text", "text": f"Goal: {goal}\n"},
                        {"type": "text", "text": f"History: {json.dumps(history)}\n\n"},
                        {"type": "text", "text": "Based on these possible next steps and their resulting last frames, which single action would you like to **confirm** and execute next?"},
                        {"type": "text", "text": f"**Option 1:** {act_1}\n\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_var_1}", "detail": "auto"}},
                        {"type": "text", "text": f"**Option 2:** {act_2}\n\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_var_2}", "detail": "auto"}},
                        {"type": "text", "text": f"**Option 3:** {act_3}\n\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_var_3}", "detail": "auto"}},
                        {"type": "text", "text": """After confirming the action, you should assess whether the goal is already achieved from these actions (including the one you want to confirm and execute next) and the corresponding next state. 
                         If so, return the selected action and set \"finished\" to True in the JSON object. Otherwise, return the action you want to confirm and execute next."""},
                    ]}
                ],
            )
        elif len(action_space)==2:
            act_1, var_path_1 = action_space[0]
            act_2, var_path_2 = action_space[1]
            b64_var_1 = encode_frame_to_b64(load_last_frame(str(var_path_1)))
            b64_var_2 = encode_frame_to_b64(load_last_frame(str(var_path_2)))
            resp2 = chat_with_retries(
                max_retries=3,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT1},
                    {"role": "user",   "content": [
                        {"type": "text", "text": f"Goal: {goal}\n"},
                        {"type": "text", "text": f"History: {json.dumps(history)}\n\n"},
                        {"type": "text", "text": "Based on these possible next steps and their resulting last frames, which single action would you like to **confirm** and execute next?"},
                        {"type": "text", "text": f"**Option 1:** {act_1}\n\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_var_1}", "detail": "auto"}},
                        {"type": "text", "text": f"**Option 2:** {act_2}\n\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_var_2}", "detail": "auto"}},
                        {"type": "text", "text": """After confirming the action, you should assess whether the goal is already achieved from these actions (including the one you want to confirm and execute next) and the corresponding next state. 
                         If so, return the selected action and set \"finished\" to True in the JSON object. Otherwise, return the action you want to confirm and execute next."""},
                        
                    ]}
                ],
            )
        elif len(action_space)==1:
            act_1, var_path_1 = action_space[0]
            b64_var_1 = encode_frame_to_b64(load_last_frame(str(var_path_1)))
            resp2 = chat_with_retries(
                max_retries=3,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT1},
                    {"role": "user",   "content": [
                        {"type": "text", "text": f"Goal: {goal}\n"},
                        {"type": "text", "text": f"History: {json.dumps(history)}\n\n"},
                        {"type": "text", "text": "Based on these possible next steps and their resulting last frames, which single action would you like to **confirm** and execute next?"},
                        {"type": "text", "text": f"**Option 1:** {act_1}\n\n"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_var_1}", "detail": "auto"}},
                        {"type": "text", "text": """After confirming the action, you should assess whether the goal is already achieved from these actions (including the one you want to confirm and execute next) and the corresponding next state. 
                         If so, return the selected action and set \"finished\" to True in the JSON object. Otherwise, return the action you want to confirm and execute next."""},
                        
                    ]}
                ],
            )
        else: break

        
        raw = resp2.choices[0].message.content.strip().strip('"')
        # 4) strip any ```json fences``` if present
        if raw.startswith("```"):
            lines = raw.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw = "\n".join(lines).strip()
        time.sleep(1.0)

        # 5) parse the JSON reply
        best_option: int = -1
        rationale: str = ""
        finished: str = ""
        try:
            parsed = json.loads(raw)
            best_option = parsed.get("best_option", -1)
            rationale = parsed.get("rationale", "").strip()
            finished = parsed.get("finished", None)
        except json.JSONDecodeError: 
            best_option = -1
            # fallback to line‐based extraction
            for line in raw.splitlines():
                if line.lower().startswith("rationale"):
                    rationale = line.partition(":")[2].strip()
                if best_option == -1:
                    m = re.match(r'Option\s+(\d+)', line)
                    if m:
                        best_option = int(m.group(1))
        if not (1 <= best_option+1):
            print(f"Invalid best_option={best_option} for candidates={candidates!r}")
            print("Raw GPT reply was:", raw)
            print(best_option, rationale, finished)
            break
        else:
            print(best_option, rationale, finished)
            proposed_trace[step] = {"proposed_actions": candidates, "best_action": candidates[best_option - 1], "rationale": rationale}
            action_picked, path = action_space[best_option - 1]
            action_list.append(action_picked)
            chosen_segments.append(path)
            history.append(action_picked)
            
            # Check if task is finished
            if "FINISHED" in action_picked or finished == True:
                break
                
            # Update input for next step to be this step's best video (not last frame PNG)
            current_input_path = path
            print(f"Step {step} completed. Next step will use video: {Path(path).name}")

    trace_path = out_dir / f"{episode}_trace.json"
    with open(trace_path, 'w') as f:
        json.dump(proposed_trace, f, indent=2)


    # action_list = [re.sub(r'^[^\w\s]+', '', action).strip('"') for action in action_list]
    return [action for action in action_list if "FINISH" not in action.upper()], chosen_segments

def generate_candidate_segments(
    filename: str,
    unique_id_for_session: str,
    round_num_for_demo: int,
    best_of_n: int,
    frame_path: Path,
    action: str,
    out_dir: Path,
    fps: float,
    cosmos_version: str = "cosmos1",
) -> Path:
    """
    Run demo_generation_xdit.py once; it will select the best segment internally and output the best video for the action.
    Return path to best segment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------- single JSONL job -------
    base_out = out_dir / f"{filename}.mp4"        
    demo = {"video_path": str(frame_path),
            "output_path": str(base_out),
            "actions": [action],
            "unique_id": unique_id_for_session,
            "round_num": round_num_for_demo}

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    tmp.write(json.dumps(demo) + "\n"); tmp.close()

    cmd = [
        "python", PANWAN_SCRIPT,
        "--demo_info", tmp.name,
        "--fps", str(fps),
        "--best_of_n", str(best_of_n),
        "--cosmos_version", cosmos_version
    ]
    
    subprocess.run(cmd, check=True)
    Path(tmp.name).unlink()

    # Only one output video is expected now (the best segment)
    best_segment = out_dir / f"{filename}_final.mp4"
    return best_segment

def concatenate_videos(segments: list[Path], out_path: Path) -> None:
    """Concatenate MP4s via ffmpeg CLI."""
    list_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    for seg in segments:
        list_file.write(f"file '{seg.resolve()}'\n") 
    list_file.close()
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', list_file.name, '-c', 'copy', str(out_path.resolve())
        ], check=True, capture_output=True) 
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg concatenation: {e}")
        print(f"ffmpeg stdout: {e.stdout.decode()}")
        print(f"ffmpeg stderr: {e.stderr.decode()}")
        raise
    finally:
        os.remove(list_file.name)

def run_single_job(job: Dict[str, Any], args: argparse.Namespace, gpu_id: Optional[int] = None) -> None:
    import traceback
    print(f"[DEBUG {os.getpid()}]: Job starting for episode={job.get('episode')}")
    try:
        # Pin this worker to a single GPU if specified
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Process one job dictionary (one line from jobs_jsonl)
            img_path = Path(job["image_path"])
            out_dir = Path(job["output_dir"])
            goal = job["goal"]
            episode = job.get("episode", img_path.stem)
            out_dir.mkdir(parents=True, exist_ok=True)
            print("[DEBUG]: start initial planning")
            
            action_list, chosen_segments = get_full_action_list(episode, max_steps = args.max_action, initial_frame_path = img_path, goal = goal, history = [], out_dir = out_dir, fps = args.fps, best_of_n = args.best_of_n, cosmos_version = args.cosmos_version)
            print(f"action_list:{action_list}")
            
            full_video = out_dir / f"{episode}_full.mp4"
            concatenate_videos(chosen_segments, full_video)

            with open(out_dir / f"{episode}_refined.json", "w") as f:
                json.dump({
                    "video_path": str(full_video),
                    "best_round_picked": [str(p) for p in chosen_segments],
                    "action_list": action_list,
                    "goal": goal,
                }, f, indent=4)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR {os.getpid()}]: subprocess failed: {e.cmd} → exit code {e.returncode}")
    except Exception:
        print(f"[EXCEPTION {os.getpid()}]:")
        traceback.print_exc()

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate, select, concat, and refine action‑driven videos in parallel.")
    parser.add_argument("--jobs_jsonl", required=True,
                        help="JSONL file; each line: {image_path, goal, output_dir, episode}")
    parser.add_argument("--max_action", type=int, default=3, help="Maximum number of action steps")
    parser.add_argument("--best_of_n", type=int, default=3)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--cosmos_version", type=str, choices=["cosmos1", "cosmos2"], default="cosmos1",
                        help="Which cosmos version to use: cosmos1 or cosmos2")

    parser.add_argument(
        "--rank",
        type=int,
        default=int(os.getenv("SLURM_PROCID", 0)),
        help="Which Slurm task this is",
    )
    parser.add_argument(
        "--ntasks",
        type=int,
        default=int(os.getenv("SLURM_NTASKS", 1)),
        help="Total number of Slurm tasks",
    )

    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes per task (defaults to #GPUs or CPU cores).")
    args = parser.parse_args()

    with open(args.jobs_jsonl) as f:
        all_jobs = [json.loads(l) for l in f if l.strip()]
    
    # Convert relative paths to absolute paths based on PROJECT_ROOT
    for job in all_jobs:
        # Convert image_path to absolute path if it's relative
        if job.get("image_path") and not Path(job["image_path"]).is_absolute():
            job["image_path"] = str(Path(PROJECT_ROOT) / job["image_path"])
        
        # Convert output_dir to absolute path if it's relative and add cosmos version suffix
        if job.get("output_dir"):
            output_dir = Path(job["output_dir"])
            # Add cosmos version suffix to output directory
            output_dir = output_dir / args.cosmos_version
            if not output_dir.is_absolute():
                job["output_dir"] = str(Path(PROJECT_ROOT) / output_dir)
            else:
                job["output_dir"] = str(output_dir)
    
    print(f"[Rank {args.rank}/{args.ntasks}] Total jobs: {len(all_jobs)}")
    
    jobs_per_rank = len(all_jobs) // args.ntasks
    start_idx = args.rank * jobs_per_rank
    if args.rank == args.ntasks - 1: 
        end_idx = len(all_jobs)
    else:
        end_idx = start_idx + jobs_per_rank
    jobs = all_jobs[start_idx:end_idx]
    
    print(f"[Rank {args.rank}/{args.ntasks}] Assigned job indices: {list(range(start_idx, end_idx))}")
    if not jobs:
        print(f"[Rank {args.rank}/{args.ntasks}] No jobs assigned – exiting.")
        return
    print(f"[Rank {args.rank}/{args.ntasks}] Assigned {len(jobs)} jobs: {[j['episode'] for j in jobs]}")

    n_gpus = torch.cuda.device_count()
    if args.workers is None:
        args.workers = n_gpus if n_gpus else os.cpu_count() or 1
    
    max_workers = min(len(jobs), n_gpus) if n_gpus > 0 else 1
    
    print(f"[Rank {args.rank}] Starting {len(jobs)} jobs with {max_workers} workers (GPUs: {n_gpus})")
    print(f"[Rank {args.rank}] Each GPU will run one job at a time, jobs will be processed sequentially per GPU")
    
    # Assign GPU ID to each job dynamically
    for idx, jb in enumerate(jobs):
        jb["_gpu_id"] = idx % max(1, n_gpus)
        jb["_job_index"] = idx
    
    with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=torch.multiprocessing.get_context("spawn")) as ex:
        futures = [ex.submit(run_single_job, jb, args, jb["_gpu_id"]) for jb in jobs]
        
        # Track completion
        completed = 0
        for fut in cf.as_completed(futures):
            try:
                fut.result()
                completed += 1
                print(f"[Rank {args.rank}] Job {completed}/{len(jobs)} completed")
            except Exception as e:
                print(f"[Rank {args.rank}] Job failed:", e)
    
    print(f"[Rank {args.rank}] All {len(jobs)} jobs completed")

def load_last_frame(path: str) -> np.ndarray:
    """Load the last frame of a video or image file without relying on mediapy metadata parsing."""
    ext = Path(path).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        return np.array(Image.open(path).convert("RGB"))
    else:
        return imageio.mimread(path, memtest=False)[-1]

if __name__ == "__main__":
    main()