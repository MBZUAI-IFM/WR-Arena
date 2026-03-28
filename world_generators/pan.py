from PIL import Image
from typing import Literal, Union
from pathlib import Path
import os
from types import SimpleNamespace
from thirdparty.pan.inference_connector import WM_inference
from thirdparty.pan.prompt_processor.upsampler import upsample_prompt
import mediapy as media
import uuid
import re

class PAN:

    def __init__(
        self, 
        generation_type: Literal["t2v", "i2v"], 
        model_params: dict, 
        inference: dict, 
        **kwargs
    ):
        """PAN model initialization - standard format"""
        self.model_params = SimpleNamespace(**model_params)
        self.inf = SimpleNamespace(**inference)
        self.generation_type = generation_type

        if generation_type != "i2v":
            raise ValueError("PAN currently supports i2v generation only.")

        # Extract parameters from model_params and inference
        self.endpoint = os.getenv("END_POINT", getattr(self.model_params, "endpoint", "http://10.24.3.85:8000"))
        
        self.seed = getattr(self.inf, 'seed', 42)
        self.num_gpus = getattr(self.inf, 'num_gpus', 1)
        self.guidance = getattr(self.inf, 'guidance', 4.0)
        self.num_steps = getattr(self.inf, 'num_steps', 50)
        self.height = getattr(self.inf, 'height', 480)
        self.width = getattr(self.inf, 'width', 832)
        self.fps = getattr(self.inf, 'fps', 20)
        self.num_video_frames = getattr(self.inf, 'frame_num', 41)
        self.num_input_frames = getattr(self.inf, 'num_input_frames', 1)
        
        self.instance = WM_inference(api_endpoint=self.endpoint)
        self.states = []
        self.i = 1
        self.video_id = None
        self.cache_root = Path("./cache")
        self.cache_root.mkdir(exist_ok=True)
        self.session_cache_dir = None  # will be created on first round


    def generate_video(self, prompt: Union[list, str], image_path: str):
        
        if isinstance(prompt, str):
            video_frames = self._generate_video_single_prompt(prompt, image_path)
        elif isinstance(prompt, list):
            video_frames = self._generate_video_multiround_prompt(prompt, image_path)
        else:
            raise ValueError("Prompt must be a string or a list of strings.")
        
        return video_frames
        
    def _generate_video_multiround_prompt(self, prompts: list[str], image_path: str):
        """
        Take a list of action-prompts and drive PAN over multiple rounds,
        concatenating the returned segments into one long list of PIL frames.

        Args
        ----
        prompts      : list of action-prompts (round-1, round-2, … order)
        image_path   : path to the starting key-frame (only sent for round-1)

        Returns
        -------
        List[PIL.Image] – all frames, with the first frame of every *later*
        round removed so there is no duplicate between segments.
        """
        
        if self.session_cache_dir is None:
            # first round for this PAN instance – create a unique sub-dir
            unique_id = uuid.uuid4().hex
            self.session_cache_dir = self.cache_root / unique_id
            self.session_cache_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path)
        image = self.resize_image(image, (self.width, self.height))
      
        # House-keeping handles for state-tracking across rounds
        state_id: str | None = None
        video_id: str | None = None
        round_idx = self.i
        assert round_idx == 1, "Round index should start at 1 for the first round."
        
        all_frames: list[Image.Image] = []
        
        # Iterate over every prompt in the list
        for prompt in prompts:
            
            print(f"prompt: {prompt}")
            print(f"image: {type(image)}")

            max_upsample_attempts = 3
            prompt_candidate = None
            success = False

            for attempt in range(max_upsample_attempts):
                prompt_candidate, cost_dict = upsample_prompt(prompt, image)
                if prompt_candidate is None:
                    continue

                words = set(re.findall(r"\b\w+\b", prompt_candidate.lower()))

                if "sorry" not in words:
                    prompt = prompt_candidate
                    success = True
                    break

            print(f"Upsampled prompt: {prompt}")
            print(f"Successfully upsampled: {success}")

            prompt = self.add_fps(prompt, self.fps)
        
            single_path, video_id, state_id = self.instance.inference_round(
                curr_round=round_idx,
                prompt_dict=prompt,
                image=image,            # ignored by backend when round>1
                save_dir=str(self.session_cache_dir),
                fps=self.fps,
                guidance_scale=self.guidance,
                denoising_steps=self.num_steps,
                prev_state_id=state_id,
                prev_video_id=video_id,
            )
        
            segment = media.read_video(single_path)
            segment_frames = [Image.fromarray(f) for f in segment]

            if round_idx == 1:
                all_frames.extend(segment_frames)
            else:
                all_frames.extend(segment_frames[1:])   # drop the overlap frame

            round_idx += 1

        # Clean up class-level trackers (fresh for next call)
        self.states = []
        self.i = 1
        self.video_id = None
        
        return all_frames

    def _generate_video_single_prompt(self, prompt: str, image_path: str):
        
        if self.session_cache_dir is None:
            # first round for this PAN instance – create a unique sub-dir
            unique_id = uuid.uuid4().hex
            self.session_cache_dir = self.cache_root / unique_id
            self.session_cache_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path)
        image = self.resize_image(image, (self.width, self.height))
        
        print(f"prompt: {prompt}")
        print(f"image: {type(image)}")

        max_upsample_attempts = 3
        prompt_candidate = None
        success = False

        for attempt in range(max_upsample_attempts):
            prompt_candidate, cost_dict = upsample_prompt(prompt, image)
            if prompt_candidate is None:
                continue

            words = set(re.findall(r"\b\w+\b", prompt_candidate.lower()))

            if "sorry" not in words:
                prompt = prompt_candidate
                success = True
                break

        print(f"Upsampled prompt: {prompt}")
        print(f"Successfully upsampled: {success}")
        prompt = self.add_fps(prompt, self.fps)
        
        if len(self.states) == 0:
            new_state_id = None
        else:
            new_state_id = self.states[-1]
            
        single_path, video_id, state_id = self.instance.inference_round(self.i,
                                                                    prompt_dict=prompt,
                                                                    image=image,
                                                                    save_dir=str(self.session_cache_dir),
                                                                    fps=self.fps,
                                                                    guidance_scale=4,
                                                                    denoising_steps=self.num_steps,
                                                                    prev_state_id=new_state_id,
                                                                    prev_video_id=self.video_id)
        self.states = [] #.append(state_id)
        self.i = 1 #+= 1
        self.video_id = None #video_id
        
        loaded_video = media.read_video(single_path)
        # Convert loaded_video (np.ndarray [T, H, W, C]) to List[Image.Image]
        video_frames = [Image.fromarray(frame) for frame in loaded_video]

        return video_frames

    def add_fps(self, in_str, fps: int):
        return f"FPS-{fps} " + in_str


    def resize_image(self, image, target_size):
        """Resize image to target size while maintaining aspect ratio.
        
        Args:
            image (PIL.Image): Input image
            target_size (tuple): Target size as (width, height)
            
        Returns:
            PIL.Image: Resized image
        """
        # Get current dimensions
        width, height = image.size
        
        # Calculate target aspect ratio
        target_width, target_height = target_size
        target_aspect = target_width / target_height
        
        # Calculate current aspect ratio
        aspect = width / height
        
        # Determine new dimensions preserving aspect ratio
        if aspect > target_aspect:
            # Image is wider than target
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Image is taller than target
            new_height = target_height
            new_width = int(target_height * aspect)
            
        # Resize the image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return resized_image
