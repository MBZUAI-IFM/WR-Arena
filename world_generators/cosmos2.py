from PIL import Image
from typing import Literal, Optional
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import torch
import random
import torch.distributed as dist

project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parents[1]))
cosmos2_path = project_root / "thirdparty" / "cosmos-predict2"

if str(cosmos2_path) not in sys.path:
    sys.path.insert(0, str(cosmos2_path))

from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_14B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from imaginaire.utils import distributed, misc
from megatron.core import parallel_state

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

def init_distributed_model(args, inf):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    print(f"{rank=}, {world_size=}, {device=}")
    
    # Initialize distributed environment for multi-GPU inference
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        
        # Check if distributed environment is already initialized
        if not parallel_state.is_initialized():
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=world_size)
            print(f"Context parallel group initialized with {world_size} GPUs")
        else:
            print("Distributed environment already initialized")
    
    return device, rank


def post_process_video(video):
    """Process video tensor to list of PIL Images"""
    video = video.squeeze(0)  
    video = (video.clamp(-1, 1) + 1) / 2 
    video = (video * 255).byte()  
    video = video.permute(1, 2, 3, 0).cpu().numpy()  
    frame_list = [Image.fromarray(frame) for frame in video]
    return frame_list


class Cosmos2:
    def __init__(
        self, 
        generation_type: Literal["t2v", "i2v"], 
        model_params: dict, 
        inference: dict, 
        **kwargs
    ):
        self.model_params = SimpleNamespace(**model_params)
        self.inf = SimpleNamespace(**inference)
        self.generation_type = generation_type

        if generation_type != "i2v":
            raise ValueError("The evaluation script is setup for i2v inference of Cosmos2 only.")

        device, rank = init_distributed_model(self.model_params, self.inf)
        
        self.rank = rank
        self.device = device
        self.seed = random.randint(0, 10000)
            
        config = PREDICT2_VIDEO2WORLD_PIPELINE_14B
        config.tokenizer.vae_pth = str(project_root / getattr(self.model_params, 'tokenizer_vae_pth'))
        config.prompt_refiner_config.checkpoint_dir = str(project_root / getattr(self.model_params, 'prompt_refiner_config_checkpoint_dir'))
     
        config.guardrail_config.enabled = getattr(self.inf, 'enable_guardrail', True)
        config.guardrail_config.offload_model_to_cpu = getattr(self.inf, 'offload_guardrail', False)
        config.prompt_refiner_config.enabled = getattr(self.inf, 'enable_prompt_refiner', True)
        config.prompt_refiner_config.offload_model_to_cpu = getattr(self.inf, 'offload_prompt_refiner', False)
        
        text_encoder_path = str(project_root / getattr(self.model_params, 'text_encoder_path', "checkpoints/google-t5/t5-11b"))
        
        dit_path = str(project_root / getattr(self.model_params, 'dit_path'))
        print(f"DiT path: {dit_path}")
        print(f"Text encoder path: {text_encoder_path}")
 
        self.pipeline = Video2WorldPipeline.from_config(
            config=config,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_prompt_refiner=config.prompt_refiner_config.enabled,
        )
        
        print("Cosmos2 Video2WorldPipeline loaded successfully")

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
            print(f"Unknown file extension: {file_ext}, assuming video")
            return 'video'

    def _get_conditional_frames(self, input_type: str):
        """Get number of conditional frames based on input type"""
        if input_type == 'image':
            return 1  
        else:
            return 5  

    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        if image_path is None:
            raise ValueError("image_path is required for i2v generation")
        
        if dist.is_initialized():
            img_path_obj = [image_path] if self.rank == 0 else [None]
            dist.broadcast_object_list(img_path_obj, src=0)
            processed_image_path = img_path_obj[0]
        else:
            processed_image_path = image_path

        misc.set_random_seed(self.seed, by_rank=True)
        print(f"Generating video with Cosmos2: {prompt}")
        
        # Detect input type and get appropriate parameters
        input_type = self._detect_input_type(processed_image_path)
        print(f"Detected input type: {input_type} for file: {processed_image_path}")
        
        try:
            # Set generation parameters
            negative_prompt = getattr(self.inf, 'negative_prompt', _DEFAULT_NEGATIVE_PROMPT)
            guidance = getattr(self.inf, 'guidance', 7.0)
            num_conditional_frames = self._get_conditional_frames(input_type)
            num_sampling_steps = getattr(self.inf, 'num_sampling_steps', 35)
            
            print(f"Using {num_conditional_frames} conditional frames for {input_type} input")
            
            video = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_path=processed_image_path,
                num_conditional_frames=num_conditional_frames,
                guidance=guidance,
                seed=self.seed,
                num_sampling_step=num_sampling_steps,
            )
            
            # Synchronize all ranks after generation
            if dist.is_initialized():
                dist.barrier()
                
            print("Cosmos2 video generation completed")
            
        except Exception as e:
            print(f"Cosmos2 generation failed: {e}")
            video = None

        # Process video output to frames
        if self.rank == 0 and video is not None:
            frames = post_process_video(video)
            print(f"Processed {len(frames)} frames from Cosmos2 generation")
        else:
            frames = None

        # Broadcast frames to all ranks
        if dist.is_initialized():
            obj = [frames] if self.rank == 0 else [None]
            dist.broadcast_object_list(obj, src=0)
            frames = obj[0]

        return frames if frames else []