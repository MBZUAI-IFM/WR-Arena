from PIL import Image
from typing import Literal, Optional
from pathlib import Path
from types import SimpleNamespace
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
        generation_type: Literal["t2v", "i2v"], 
        model_params: dict, 
        inference: dict, 
        **kwargs
    ):
        self.model_params = SimpleNamespace(**model_params)
        self.inf = SimpleNamespace(**inference)
        self.generation_type = generation_type

        if generation_type != "i2v":
            raise ValueError("The evaluation script is setup for i2v inference of Cosmos1 only.")
        
        self.checkpoint_dir = str(project_root / getattr(self.model_params, 'checkpoint_dir', 'checkpoints/cosmos_predict1_14b_video2world'))
        self.diffusion_transformer_dir = getattr(self.model_params, 'diffusion_transformer_dir', 'Cosmos-Predict1-14B-Video2World')
        self.prompt_upsampler_dir = getattr(self.model_params, 'prompt_upsampler_dir', 'Pixtral-12B')
        self.disable_prompt_upsampler = getattr(self.model_params, 'disable_prompt_upsampler', False)
        self.offload_diffusion_transformer = getattr(self.model_params, 'offload_diffusion_transformer', False)
        self.offload_tokenizer = getattr(self.model_params, 'offload_tokenizer', False)
        self.offload_text_encoder_model = getattr(self.model_params, 'offload_text_encoder_model', False)
        self.offload_prompt_upsampler = getattr(self.model_params, 'offload_prompt_upsampler', True)
        self.offload_guardrail_models = getattr(self.model_params, 'offload_guardrail_models', True)
        self.disable_guardrail = getattr(self.model_params, 'disable_guardrail', True)
        
        self.seed = random.randint(0, 10000)
        self.num_gpus = getattr(self.inf, 'num_gpus', 8)
        self.guidance = getattr(self.inf, 'guidance', 7)
        self.num_steps = getattr(self.inf, 'num_steps', 35)
        self.height = getattr(self.inf, 'height', 704)
        self.width = getattr(self.inf, 'width', 1280)
        self.fps = getattr(self.inf, 'fps', 24)
        self.num_video_frames = getattr(self.inf, 'frame_num', 121)
        self.negative_prompt = getattr(self.inf, 'negative_prompt', 
            "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.")

    def _detect_input_type(self, input_path: str):

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

    def _get_inference_config(self, input_type: str):
        
        if input_type == 'image':
            return "video2world", 1
        else:
            return "video2world", 9

    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        if image_path is None:
            raise ValueError("image_path is required for i2v generation")
        
        input_type = self._detect_input_type(image_path)
        print(f"Detected input type: {input_type} for file: {image_path}")
        
        inference_type, num_input_frames_to_use = self._get_inference_config(input_type)
        print(f"Using inference_type: {inference_type}, num_input_frames: {num_input_frames_to_use}")
         
        from megatron.core import parallel_state
        from cosmos_predict1.utils import distributed
        
        process_group = None
        if not parallel_state.is_initialized():
            distributed.init()
            parallel_state.initialize_model_parallel(context_parallel_size=self.num_gpus)
            
        if parallel_state.is_initialized():
            process_group = parallel_state.get_context_parallel_group()
        
        misc.set_random_seed(self.seed, by_rank=True)

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
                height=self.height,
                width=self.width,
                fps=self.fps,
                num_video_frames=self.num_video_frames,
                seed=self.seed,
                num_input_frames=num_input_frames_to_use,
            )
            
        if process_group is not None:
            pipeline.model.net.enable_context_parallel(process_group)

        generated_output = pipeline.generate(
            prompt=prompt,
            image_or_video_path=image_path,
            negative_prompt=self.negative_prompt,
        )
        
        if generated_output is not None:
            video, prompt = generated_output
            frame_list = []
            for frame in video:
                frame_list.append(Image.fromarray(frame))
            video = frame_list
        else:
            video = None

        return video if video else []
    
@contextlib.contextmanager
def cd(new_path):
    saved_path = os.getcwd()
    try:
        os.chdir(os.path.expanduser(new_path))
        yield
    finally:
        os.chdir(saved_path)
    