from typing import Optional
from pathlib import Path
import os
import sys
import random
import contextlib

project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parents[1]))
cosmos1_path = project_root / "thirdparty" / "cosmos-predict1"

if str(cosmos1_path) not in sys.path:
    sys.path.insert(0, str(cosmos1_path))

from cosmos_predict1.diffusion.inference.world_generation_pipeline import DiffusionVideo2WorldGenerationPipeline
from cosmos_predict1.utils import misc

class Cosmos1:
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        diffusion_transformer_dir: Optional[str] = None,
        prompt_upsampler_dir: Optional[str] = None,
        guidance: float = 7.0,
        num_steps: int = 35,
        fps: int = 24,
        num_video_frames: int = 121,
        seed: Optional[int] = None,
        disable_prompt_upsampler: bool = True,
        disable_guardrail: bool = True,
        offload_prompt_upsampler: bool = True,
        offload_guardrail_models: bool = True,
        offload_diffusion_transformer: bool = False,
        offload_tokenizer: bool = False,
        offload_text_encoder_model: bool = False,
    ):
        """Initialize Cosmos1 video generator
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            diffusion_transformer_dir: Name of diffusion transformer directory
            prompt_upsampler_dir: Name of prompt upsampler directory
            guidance: Guidance scale for generation
            num_steps: Number of sampling steps
            fps: Video frames per second
            num_video_frames: Number of video frames to generate
            seed: Random seed (random if None)
            disable_prompt_upsampler: Disable prompt upsampler
            disable_guardrail: Disable safety guardrail
            offload_prompt_upsampler: Offload prompt upsampler
            offload_guardrail_models: Offload guardrail models
            offload_diffusion_transformer: Offload diffusion transformer
            offload_tokenizer: Offload tokenizer
            offload_text_encoder_model: Offload text encoder model
        """
        self.generation_type = "i2v"
        
        # Set paths
        self.checkpoint_dir = checkpoint_dir or str(cosmos1_path / "checkpoints")
        self.diffusion_transformer_dir = diffusion_transformer_dir or "Cosmos-Predict1-14B-Video2World"
        self.prompt_upsampler_dir = prompt_upsampler_dir or "Pixtral-12B"
        
        # Model configuration
        self.disable_prompt_upsampler = disable_prompt_upsampler
        self.offload_diffusion_transformer = offload_diffusion_transformer
        self.offload_tokenizer = offload_tokenizer
        self.offload_text_encoder_model = offload_text_encoder_model
        self.offload_prompt_upsampler = offload_prompt_upsampler
        self.offload_guardrail_models = offload_guardrail_models
        self.disable_guardrail = disable_guardrail
        
        # Generation parameters
        self.seed = seed if seed is not None else random.randint(0, 10000)
        self.guidance = guidance
        self.num_steps = num_steps
        self.fps = fps
        self.num_video_frames = num_video_frames
        self.negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

        print(f"Cosmos1 initialized with checkpoint_dir: {self.checkpoint_dir}, guidance: {self.guidance}, num_steps: {self.num_steps}")

    def _detect_input_type(self, input_path: str):

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext in image_extensions:
            return 'image'
        elif file_ext in video_extensions:
            return 'video'
        else:
            print(f"Warning: Unknown file extension: {file_ext}, assuming video")
            return 'video'

    def _get_inference_config(self, input_type: str):
        
        if input_type == 'image':
            return "video2world", 1
        else:
            return "video2world", 9

    def generate_video(self, prompt: str, input_path: str):
        misc.set_random_seed(self.seed)
        
        input_type = self._detect_input_type(input_path)
        print(f"Detected input type: {input_type} for file: {input_path}")
        
        inference_type, num_input_frames_to_use = self._get_inference_config(input_type)
        print(f"Using inference_type: {inference_type}, num_input_frames: {num_input_frames_to_use}")

        with cd(cosmos1_path):
            pipeline = DiffusionVideo2WorldGenerationPipeline(
                inference_type=inference_type,
                checkpoint_dir=self.checkpoint_dir,
                checkpoint_name=self.diffusion_transformer_dir,
                prompt_upsampler_dir=self.prompt_upsampler_dir,
                enable_prompt_upsampler=not self.disable_prompt_upsampler,
                offload_network=self.offload_diffusion_transformer,
                offload_tokenizer=self.offload_tokenizer,
                offload_text_encoder_model=self.offload_text_encoder_model,
                offload_prompt_upsampler=self.offload_prompt_upsampler,
                offload_guardrail_models=self.offload_guardrail_models,
                disable_guardrail=self.disable_guardrail,
                guidance=self.guidance,
                num_steps=self.num_steps,
                fps=self.fps,
                num_video_frames=self.num_video_frames,
                seed=self.seed,
                num_input_frames=num_input_frames_to_use,
            )

        generated_output = pipeline.generate(
            prompt=prompt,
            image_or_video_path=input_path,
            negative_prompt=self.negative_prompt,
        )
        
        if generated_output is not None:
            video, prompt = generated_output

        return video
    
@contextlib.contextmanager
def cd(new_path):
    saved_path = os.getcwd()
    try:
        os.chdir(os.path.expanduser(new_path))
        yield
    finally:
        os.chdir(saved_path)
    