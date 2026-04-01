import io
import json
import base64
import re
import openai
import numpy as np
import os
import argparse
from PIL import Image

SYSTEM_PROMPT = """You are an expert task planner for a stationary single-arm robot. Your role is to decompose high-level visual manipulation tasks into clear, actionable steps described purely in natural language.
──────────────── Inventory (silent) ────────────────
• First, scan the current frame and build an internal list of all visible blocks  
  using the template  
  “<color> <shape> at <landmark>”  
  where  
    ▸ <shape> ∈ (circle, square, triangle, cube, star, heart, hexagon, cylinder)  
    ▸ <landmark> ∈ (top‑left corner, top‑right corner, bottom‑left corner,
                    bottom‑right corner, top edge, bottom edge, left edge,
                    right edge, centre).  
• Do **not** output this list; it is for disambiguation only.
──────────────── Task Decomposition ────────────────
Guidelines for Task Decomposition:

• Clearly state each next action the robot should perform, step-by-step.  
• Specify explicitly how the robot should interact with the object(s): move.
• Reference every block by the *unique* <color> + <shape> from inventory and
  by clear relative landmarks (e.g. “to the left of the yellow hexagon” or
  “towards the bottom edge”).  Numeric coordinates and gripper jargon are
  forbidden.  
• Use concise and clear directional terms: left, right, top (up), bottom (down), front (forward), and back (backward).  
• When possible, reference other objects clearly (e.g., “move the blue cube to the right of the yellow hexagon”).  

Important Notes:

- Decomposition must be purely vision-based; no numeric coordinates.  
- Specify object parts (front, back) only if needed.  
- Consider the robot's previous actions (action history) when choosing the next optimal step.
- Generate one step at a time. Finish with \"FINISHED\"
- Every object reference must be uniquely identifiable from the current image or the action history.
- Specify which object to operate. Use only facts. Do not use ambiguous description.
- The maximum steps breakdown should not exceed {max_steps} steps.

Example:

Task: "Group all objects by color."
Action History: ["move red circle towards the red star", "move the blue square downwards to the top of blue triangle"]
Next Step: "move yellow hexagon to the right of the yellow heart"

Do not return code or directly call API functions. Do not print out Action History. Print content in the Next Step only and don't print any other contents. Never claim something like I could not assist or I could not see the image. Do not use vague terms such as "other," "another," or "similar". Please describe each action clearly and specifically."""

def encode_frame_to_b64(frame: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def get_full_action_list(max_steps: int, frame_b64: str, goal: str, history: list[str]) -> list[str]:
    action_list = []

    for _ in range(max_steps):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(max_steps=max_steps)},
            {"role": "user", "content": [
                {"type": "text", "text": f"Goal: {goal}"},
                {"type": "text", "text": f"History: {json.dumps(history)}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame_b64}", "detail": "auto"}},
                {"type": "text", "text": f"Please propose the next single step (just the action). You should also review the action history, return \"FINISHED\" if goal: {goal} is already achieved."}
            ]}
        ]
        resp = openai.chat.completions.create(model="o3", messages=messages)
        act = resp.choices[0].message.content.strip()
        action_list.append(act)
        history.append(act) 

        if "FINISHED" in act:
            break 

    action_list = [
    re.sub(r'^[^\w\s]+', '', action).strip('"') for action in action_list
]
    return [action for action in action_list if action != "FINISHED"]

def load_dataset(dataset_path):
    """Load the dataset from JSONL file"""
    cases = []
    with open(dataset_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases

def process_single_case(case, max_steps):
    """Process a single case and return the result"""
    # Load image
    image_path = case["image_path"]
    init_img = Image.open(image_path).convert("RGB").resize((256, 256))
    
    # Convert image to base64
    buf = io.BytesIO()
    init_img.save(buf, "PNG")
    frame_b64 = base64.b64encode(buf.getvalue()).decode()
    
    # Get goal
    goal = case["goal"]
    
    # Generate actions
    history = []
    actions = get_full_action_list(max_steps, frame_b64, goal, history)
    
    return {
        "goal": goal,
        "actions": actions,
        "episode": case["episode"],
        "image_path": case["image_path"]
    }

def save_result(result, output_path):
    """Save result to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process VLM-only planning tasks')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the dataset JSONL file')
    parser.add_argument('--openai_api_key', type=str, required=True,
                        help='OpenAI API key')
    
    args = parser.parse_args()
    
    # Set OpenAI API key
    openai.api_key = args.openai_api_key
    
    # Load dataset
    cases = load_dataset(args.dataset_path)
    
    print(f"Processing {len(cases)} cases...")
    
    # Process each case
    for i, case in enumerate(cases, 1):
        # Get max_action from case, default to 5 if not specified
        max_action = case.get("max_action", 5)
        print(f"Processing case {i}/{len(cases)}: {case['episode']} (max_steps={max_action})")
        
        try:
            # Process the case
            result = process_single_case(case, max_action)
            
            # Save result
            output_path = os.path.join(case["output_dir"], "VLM_only.json")
            save_result(result, output_path)
            
            print(f"Saved result to: {output_path}")
            print(f"Goal: {result['goal']}")
            print(f"Generated {len(result['actions'])} actions:")
            for j, action in enumerate(result['actions'], 1):
                print(f"  {j}. {action}")
            print("-" * 80)
            
        except Exception as e:
            print(f"Error processing case {case['episode']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Completed processing all {len(cases)} cases!")

if __name__ == "__main__":
    main()