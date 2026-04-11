import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "thirdparty", "vjepa2"))
import torch
torch.cuda.empty_cache()
import numpy as np
import re

import torch
from torch.nn import functional as F
import json
import mediapy as media
import io
import base64
from PIL import Image
import openai
import numpy as np
from pathlib import Path

from app.vjepa_droid.transforms import make_transforms
from transformers import UMT5EncoderModel, AutoTokenizer
import torch
from torch.nn import functional as F
from src.models.ac_predictor import vit_ac_predictor
from src.models.vision_transformer import vit_giant_xformers

parser = argparse.ArgumentParser(description='VJEPA2 Reasoning and Planning')
parser.add_argument('--openai_key', type=str, required=True, 
                    help='OpenAI API key')
parser.add_argument('--dataset_path', type=str, 
                    default='datasets/simulative_reasoning_planning/open_ended_simulation_planning/samples_vjepa2.jsonl',
                    help='Path to dataset JSONL file)')
args = parser.parse_args()

openai.api_key = args.openai_key
state_dict = torch.load("thirdparty/vjepa2/checkpoints/e275.pt")

text_encoder = UMT5EncoderModel.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="text_encoder"
).to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained("Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="tokenizer")


# Initialize VJEPA 2-AC model
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
        use_sdpa=False,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
    ).eval().to("cuda")



renamed_encoder_state_dict = {}
for k, v in state_dict["encoder"].items():
    if k.startswith("module."):
        renamed_encoder_state_dict[k[7:]] = v
    else:
        renamed_encoder_state_dict[k] = v

renamed_predictor_state_dict = {}
for k, v in state_dict["predictor"].items():
    if k.startswith("module."):
        renamed_predictor_state_dict[k[7:]] = v
    else:
        renamed_predictor_state_dict[k] = v

encoder.load_state_dict(renamed_encoder_state_dict)
predictor.load_state_dict(renamed_predictor_state_dict)

# Initialize transform
crop_size = 256
tokens_per_frame = int((crop_size // encoder.patch_size) ** 2)
transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1., 1.),
    random_resize_scale=(1., 1.),
    reprob=0.,
    auto_augment=False,
    motion_shift=False,
    crop_size=crop_size,
)

@torch.no_grad()
def step_predictor(_z, _t):
    _z = predictor(_z, _t)
    _z = F.layer_norm(_z, (_z.size(-1),))
    return _z

# get image rep by v-jepa encoder
@torch.no_grad()
def forward_target(c):
    batch_size = c.shape[0]
    c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    h = encoder(c)
    h = h.view(batch_size, 1, -1, h.size(-1)).flatten(1, 2)
    h = F.layer_norm(h, (h.size(-1),))
    return h

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
    if create_kwargs.get("model") in ("o3", "o3-mini", "o3-mini-high"):
        create_kwargs.pop("temperature", None)
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

from PIL import Image
import numpy as np
from tqdm import tqdm

SYSTEM_PROMPT="""
You are an expert task planner for a stationary single-arm robot. Your role is to decompose high-level visual manipulation tasks into clear, actionable steps described purely in natural language.

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
- The maximum steps breakdown should not exceed {max_steps} steps.
- Do not use other objects included in the goal as a relative position reference. If asked to align objects in a line, don't simply say commands like 'place object A into a position that make objects align into a line' as this gives no information to the target position of the move. Instead you should mention this position clearly with respect to the background objects (e.g., table).

Example:

Task: "Place the red cup behind the green box."  
Action History: ["Pick up red cup by its handle."]
The user will then ask for multiple proposals. An appropriate response would be a list of possible next actions, like this:
- Move the red cup to behind the green box.
- Place the red cup on the table to the right of the green box.
- Move the red cup to the front of the green box.
Do not return code or directly call API functions. Do not print out Action History. Print content in the Next Step only and don't print any other contents. Never claim something like 'I'm sorry, I can't assist with that' or 'I could not see the image'. Do not use vague terms such as "other," "another," or "similar". Please describe each action clearly and specifically."""



def load_goal_state_frame(path):
    path = path.strip()
    with open(path, 'rb') as f:
        image = Image.open(f)
        return np.array(image)
    
def infer_correct_action(first_frame_representation, candidate_actions, goal_frame_representation, N):
    """
    Selects the best plan from candidate_actions by rolling out N steps for each plan,
    and comparing the predicted final frame representation to the goal_frame_representation.

    Args:
        first_frame_representation (torch.Tensor): the initial frame feature, shape [1, 256, 1408]
        candidate_actions (list[str]): list of text instructions, each instruction is a one-step action
        goal_frame_representation (torch.Tensor): goal frame feature, shape [1, 256, 1408]
        N (int): number of rollout steps

    Returns:
        str: the plan with the highest similarity to the goal representation
    """

    best_score = -float('inf')
    best_plan = None
    best_idx = None
    for idx, plan in enumerate(candidate_actions):
        # Encode the text plan into a text embedding
        text_inputs = tokenizer(plan, return_tensors="pt").to("cuda")
        encoded_text = text_encoder(**text_inputs).last_hidden_state  # shape [1, L, dim]
        # Initialize the rollout with the first frame
        z = first_frame_representation

        # print(z.dtype)
        # print(encoded_text.dtype)
        assert z.dtype == encoded_text.dtype, f"Mismatch: z={z.dtype}, text={encoded_text.dtype}"


        # Predict forward for N steps
        for i in range(N):
            z_next = step_predictor(z, encoded_text)[:, -256:]  # predict next frame tokens
            z = torch.cat([z, z_next], dim=1)

        # Reshape the rollout to separate each frame
        total_T = 1 + N
        frame_by_frame = z.reshape(1, total_T, 256, -1)  # [1, total_T, 256, 1408]

        # Take the representation of the last predicted frame
        last_frame = frame_by_frame[:, -1, :, :]  # [1, 256, 1408]

        # Optionally pool over spatial tokens (mean pooling)
        pooled_last = last_frame.mean(dim=1)# [1, 1408]
        pooled_goal = goal_frame_representation.mean(dim=1)# [1, 1408]

        # Compute cosine similarity between last frame and goal frame
        score = torch.nn.functional.cosine_similarity(pooled_last, pooled_goal)

        # Keep track of the best scoring plan
        if score.item() > best_score:
            best_score = score.item()
            best_plan = plan
            best_idx = idx

    return (best_idx, best_plan, last_frame)

#  2. for each step, get candidate actions based on history action
def get_candidate_action(
        image_path: Path,
        goal: str,
        system_prompt: str,
        history: list[str],
        model: str = "o3",
        temperature: float = 0.0,
        ):
    initial_b64_frame=encode_frame_to_b64(media.read_video(str(image_path))[-1])

    # 1) Propose 3 candidate actions    
    resp = chat_with_retries(
        max_retries=3,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": [
                {"type": "text", "text": f"Goal: {goal}\n"},
                {"type": "text", "text": f"History: {json.dumps(history)}\n\n"},
                {"type": "text", "text": f"Initial frame:\n\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{initial_b64_frame}", "detail": "auto"}},
                    {"type": "text", "text": "Please propose **3 possible next single steps** (just the action text)."},
                    {"type": "text", "text": "Do not use vague terms such as 'other,' 'another,' or 'similar'. Please describe each action clearly and specifically. For instance, say 'place the yellow package with the yellow packages in the right blue tray' instead of 'place the yellow package with the other yellow packages'"},
                    {"type": "text", "text": "Do not include two actions in a single step. For example, 'grasp the yellow package in the left blue tray and move it to the right blue tray' should be decomposed to two separate actions: 'grasp the yellow package in the left blue tray', 'move the yellow package to the right blue tray'."},
                    {"type": "text", "text": "Do not use other objects included in the goal as a relative position reference. If asked to align objects in a line, don't simply say commands like 'place pink object into a position that make objects align into a line' as this gives no information to the target position of the move. Instead you should mention this position clearly with respect to the background objects (e.g., table)."},
                    {"type": "text", "text": f"You should also review the action history, set \"finished\" to True if goal: {goal} is already achieved."},
            ]}
        ],
        temperature=temperature
    )
    content = resp.choices[0].message.content
    clean = re.sub(r"```json[^\n]*\n", "", content, flags=re.IGNORECASE)
    clean = clean.replace("```", "")
    lines = [ln.strip("- ").strip()
        for ln in clean.splitlines()
        if ln.strip()]
    candidates = lines[:3]
    return candidates


data = []
with open(args.dataset_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():  
            data.append(json.loads(line))

processed_data = []
for item in data:
    goal = item['goal']
    image_path = item['input_state']
    goal_image_path = item["goal_state"]
    goal_image = load_goal_state_frame(goal_image_path)
    processed_data.append({
        "episode": item['episode'],
        "goal": goal,
        "image_path": image_path,
        "goal_image": goal_image,
        "output_dir": item['output_dir'],
        "max_action": item['max_action']
    })
    

system_prompt = SYSTEM_PROMPT
for sample in tqdm(processed_data, desc="Processing samples"):
    #  3. get first/goal frame representation
    max_steps = sample["max_action"]  
    text_instruction = sample["goal"]
    text_instruction = tokenizer(text_instruction, return_tensors="pt", padding="max_length", truncation=True, max_length=32)
    text_input_ids, mask = text_instruction.input_ids[0], text_instruction.attention_mask[0]
    text_input_ids = text_input_ids.to("cuda").unsqueeze(0)
    mask = mask.to("cuda").unsqueeze(0)
    with torch.no_grad():
        encoded_text = text_encoder(text_input_ids, attention_mask=mask).last_hidden_state
    image_path = sample["image_path"]
    image = Image.open(image_path).convert('RGB')  
    first_image = np.expand_dims(np.array(image), axis=0)
    first_image = transform(first_image)

    goal_image = sample["goal_image"]
    goal_image = np.expand_dims(goal_image, axis=0)
    goal_image = transform(goal_image)

    first_frame_rep = forward_target(first_image.unsqueeze(0).to("cuda"))
    goal_frame_rep = forward_target(goal_image.unsqueeze(0).to("cuda"))

    history = []
    intermidiate_frame = first_frame_rep
    for step in range(max_steps):
        canditate_actions = get_candidate_action(image_path = sample["image_path"], goal = sample["goal"], system_prompt = system_prompt.format(max_steps=max_steps), history=history)
        _, best_action, intermidiate_frame = infer_correct_action(intermidiate_frame, canditate_actions, goal_frame_rep, N=10)
        history.append(best_action)
    
    # Create output for this sample
    sample_result = {
        "episode": sample["episode"],
        "goal": sample["goal"],
        "image_path": sample["image_path"],
        "plan": history,
        "max_action": sample["max_action"]
    }
    
    # Create output directory structure with vjepa2 subfolder
    output_dir = os.path.join(sample["output_dir"], "vjepa2")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save result to file
    output_file = os.path.join(output_dir, f"{sample['episode']}_refined.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_result, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {sample['episode']}: {sample['image_path']}")
    print(f"Plan: {history}")
    print(f"Saved to: {output_file}")
    
print("All samples processed successfully!")
