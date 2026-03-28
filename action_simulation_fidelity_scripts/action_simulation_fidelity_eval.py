import sys
import argparse
from pathlib import Path
import cv2
import base64
import io
from PIL import Image
import numpy as np
from openai import OpenAI
from mmengine import load, dump
import re

EVAL_PROMPT_TEMPLATE = (
    "You are given a sequence of frames sampled in chronological order from a video.\n"
    "Evaluate whether the sequence follows the instruction: \"{instruction}\".\n"
    "Use the following scoring criteria:\n"
    "- 0: The sequence does not follow the instruction at all.\n"
    "- 1: The sequence includes the correct object but performs the wrong action, or includes the correct action but the wrong object.\n"
    "- 2: The sequence follows the instruction and shows a tendency toward the intended goal.\n"
    "- 3: The sequence follows the instruction precisely and successfully achieves the goal.\n"
    "Return ONLY one integer: 0, 1, 2, or 3. Do not output any other text."
)


def extract_frames(video_path: str, num_frames: int = 8):
    """Extract frames from video as base64 PNGs"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames_b64 = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        b64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        frames_b64.append(b64_string)
    
    cap.release()
    return frames_b64

def evaluate_part(video_path: str, instruction: str, client: OpenAI, model: str):
    """Evaluate one video segment with GPT-4o"""
    frames = extract_frames(video_path)
    
    messages = [
        {"role": "system", "content": "You are a video evaluation assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": EVAL_PROMPT_TEMPLATE.format(instruction=instruction)}
        ] + [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame}"}}
            for frame in frames
        ]}
    ]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return ""

def compute_final_scores(scores: dict):
    """Compute final agent/env scores from predictions"""
    segment_scores = {i: [] for i in range(1, 7)}

    for path, score_list in scores.items():
        if not score_list:
            continue

        match = re.search(r"(\d+)_([1-6])", path)
        if not match:
            continue

        index = int(match.group(2))

        try:
            valid_scores = [int(s.strip()) for s in score_list if s.strip().isdigit()]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                segment_scores[index].append(avg_score)
        except (ValueError, TypeError):
            continue

    segment_averages = {
        segment: round(sum(scores) / len(scores), 2) 
        for segment, scores in segment_scores.items() if scores
    }

    agent_segments = [segment_averages[i] for i in [1, 2, 3] if i in segment_averages]
    env_segments = [segment_averages[i] for i in [4, 5, 6] if i in segment_averages]

    return {
        "segment_averages": segment_averages,
        "average": {
            "agent": round(sum(agent_segments) / len(agent_segments), 2) if agent_segments else 0.0,
            "env": round(sum(env_segments) / len(env_segments), 2) if env_segments else 0.0
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate action simulation fidelity using GPT-4o")
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path for the video files (e.g., outputs/action_simulation_fidelity/cosmos1)"
    )
    
    parser.add_argument(
        "--dataset_json",
        type=str,
        required=True,
        help="json path for the prompts (e.g., datasets/action_simulation_fidelity_subset/samples_subset.json)"
    )
    
    parser.add_argument(
        "--save_name",
        type=str,
        required=True,
        help="Name for saving results (e.g., cosmos1)"
    )
    
    args = parser.parse_args()

    # Setup OpenAI client
    api_key = args.openai_api_key
    if not api_key:
        print("OpenAI API key not provided.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    validation_set = load(args.dataset_json)
    scores = {}

    # === Step 1: GPT Evaluation ===
    print("🤖 Starting GPT evaluation...")
    
    for idx, entry in enumerate(validation_set):
        prompt_list = entry.get("prompt_list", [])
        if len(prompt_list) < 3:
            print(f"Warning: Not enough prompts for: {entry.get('image_path', entry.get('image'))}")
            continue

        image_key = entry["id"].split("_", 1)[-1]
        rounds_dir = Path(args.base_path) / image_key / "rounds"
        part_scores = []

        print(f"📹 Evaluating {image_key} ({idx+1}/{len(validation_set)})")

        for i, prompt in enumerate(prompt_list[-3:]):
            part_path = rounds_dir / f"round_{i:03d}.mp4"
            if not part_path.exists():
                print(f"Warning: Part video not found: {part_path}")
                continue
            
            instruction = prompt if prompt.endswith('.') else prompt + '.'
            print(f"  🎬 Evaluating round {i}: {instruction[:50]}...")
            score = evaluate_part(str(part_path), instruction, client, "gpt-4o")
            part_scores.append(score)
            print(f"  📊 Score: {score}")

        scores[image_key] = part_scores
        
        # Save intermediate results
        output_dir = Path("outputs/action_simulation_fidelity") / args.save_name
        output_dir.mkdir(parents=True, exist_ok=True)
        results_data = {"scores": scores}
        dump(results_data, output_dir / f"{args.save_name}_results.json", indent=4)
        print(f"💾 Saved intermediate results for {image_key}")

    # === Step 2: Compute Final Scores ===
    print("🧮 Computing final scores...")
    final_results = compute_final_scores(scores)
    
    # === Step 3: Save Final Results ===
    output_dir = Path("outputs/action_simulation_fidelity") / args.save_name
    output_dir.mkdir(parents=True, exist_ok=True)
    results_data = {
        "scores": scores,
        "average": final_results['average']
    }
    final_results_path = output_dir / f"{args.save_name}_results.json"
    dump(results_data, final_results_path, indent=4)
    print(f"💾 Detailed predictions with averages saved to: {final_results_path}")
    
    # === Final Report ===
    print("✅ Evaluation completed!")
    print(f"📊 Final Results:")
    print(f"   Agent Score: {final_results['average']['agent']}")
    print(f"   Env Score: {final_results['average']['env']}")
    
    return final_results

if __name__ == "__main__":
    main()
