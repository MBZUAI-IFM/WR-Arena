import os
import random
import sys
from pathlib import Path
from typing import Optional
import torch

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent)
_COSMOS_PREDICT2_ROOT = Path(PROJECT_ROOT) / "thirdparty" / "cosmos-predict2"
sys.path.insert(0, str(_COSMOS_PREDICT2_ROOT))

from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_14B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import misc

class Cosmos2:
    def __init__(
        self,
        tokenizer_vae_pth: Optional[str] = None,
        dit_path: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        seed: Optional[int] = None,
        guidance: float = 7.0,
        num_sampling_steps: int = 35,
        enable_guardrail: bool = False,
        enable_prompt_refiner: bool = False,
    ):
        """Initialize Cosmos2 video generator
        
        Args:
            tokenizer_vae_pth: Path to tokenizer VAE
            dit_path: Path to DiT model checkpoint
            text_encoder_path: Path to text encoder
            guidance: Guidance scale for generation
            num_sampling_steps: Number of sampling steps
            enable_guardrail: Enable safety guardrail
            enable_prompt_refiner: Enable prompt refiner
        """
        # Set paths 
        base_checkpoint_dir = Path(_COSMOS_PREDICT2_ROOT) / "checkpoints"
        self.tokenizer_vae_pth = tokenizer_vae_pth or str(base_checkpoint_dir / "nvidia/Cosmos-Predict2-14B-Video2World/tokenizer/tokenizer.pth")
        self.dit_path = dit_path or str(base_checkpoint_dir / "nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt")
        self.text_encoder_path = text_encoder_path or str(base_checkpoint_dir / "google-t5/t5-11b")
        
        # Generation parameters
        self.guidance = guidance
        self.num_sampling_steps = num_sampling_steps
        self.negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
        self.seed = seed if seed is not None else random.randint(0, 10000)
        
        # Configure pipeline 
        config = PREDICT2_VIDEO2WORLD_PIPELINE_14B
        config.tokenizer.vae_pth = self.tokenizer_vae_pth
        config.guardrail_config.enabled = enable_guardrail
        config.prompt_refiner_config.enabled = enable_prompt_refiner
        
        # Initialize pipeline
        self.pipeline = Video2WorldPipeline.from_config(
            config=config,
            dit_path=self.dit_path,
            text_encoder_path=self.text_encoder_path,
            device="cuda",
            torch_dtype=torch.bfloat16
        )
        
        print(f"Cosmos2 initialized with DiT: {Path(self.dit_path).name}")

    def _detect_input_type(self, input_path: str):
        """Detect whether input is an image or video based on file extension"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        file_ext = Path(input_path).suffix.lower()
        
        if file_ext in image_extensions:
            return 'image'
        elif file_ext in video_extensions:
            return 'video'
        else:
            print(f"Warning: Unknown file extension {file_ext}, assuming video")
            return 'video'

    def _get_conditional_frames(self, input_type: str):
        """Get number of conditional frames based on input type"""
        if input_type == 'image':
            return 1  
        else:
            return 5

    def generate_video(self, prompt: str, input_path: str):
        """Generate video from image or video input
        
        Args:
            prompt: Text prompt for generation
            input_path: Path to input image or video
            
        Returns:
            Generated video tensor
        """
        misc.set_random_seed(self.seed)
        input_type = self._detect_input_type(input_path)
        num_conditional_frames = self._get_conditional_frames(input_type)
        
        print(f"Generating video from {input_type} input: {Path(input_path).name}")
        print(f"Using {num_conditional_frames} conditional frames")
        
        try:
            video = self.pipeline(
                input_path=input_path,
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                num_conditional_frames=num_conditional_frames,
                guidance=self.guidance,
                seed=self.seed,
                num_sampling_step=self.num_sampling_steps
            )
            print("Video generation completed successfully")
            return video
            
        except Exception as e:
            print(f"Error during video generation: {e}")
            raise