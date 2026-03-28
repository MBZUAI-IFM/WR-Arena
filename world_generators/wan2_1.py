import os
import sys
import time
import torch
import torch.distributed as dist
import random
from pathlib import Path
from PIL import Image
from typing import Optional, Literal
from types import SimpleNamespace
import torchvision
from torchvision import transforms
import thirdparty.wan2_1 as wan
from thirdparty.wan2_1.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from thirdparty.wan2_1.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parents[1]))

def init_distributed_model(model_params, cfg, inf):
    """Distributed model initialization"""
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    print(f"{rank=}, {world_size=}, {device=}")
    
    # Set offload_model default value
    if not hasattr(inf, 'offload_model') or inf.offload_model is None:
        inf.offload_model = False if world_size > 1 else True
        print(f"offload_model is not specified, set to {inf.offload_model}.")
    
    # Distributed initialization
    already_init = dist.is_available() and dist.is_initialized()
    if world_size > 1:
        if not already_init:
            print(f"-----------see if initialized")
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
    else:
        # Single GPU environment check
        assert not (
            model_params.t5_fsdp or model_params.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            model_params.ulysses_size > 1 or model_params.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    # xfuser model parallel initialization 
    if model_params.ulysses_size > 1 or model_params.ring_size > 1:
        assert model_params.ulysses_size * model_params.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=model_params.ring_size,
            ulysses_degree=model_params.ulysses_size,
        )
        print(f"xfuser model parallel initialization: ulysses={model_params.ulysses_size}, ring={model_params.ring_size}, 8GPU collaborative generation")
        
    if model_params.ulysses_size > 1:
        assert cfg.num_heads % model_params.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{model_params.ulysses_size=}`."
    
    # Set random seed (get from inference)
    if not hasattr(inf, 'base_seed') or inf.base_seed is None:
        inf.base_seed = random.randint(0, sys.maxsize)
        
    if dist.is_initialized():
        base_seed = [inf.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        inf.base_seed = base_seed[0]
        
    return device, rank

def dynamic_resize(img, video_size=(832, 480)):
    """Dynamically resize image"""
    width, height = img.size
    t_width, t_height = video_size
    k = max(t_width / width, t_height / height)
    new_width, new_height = int(width * k), int(height * k)
    trans = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.CenterCrop(video_size[::-1]),
    ])
    return trans(img)

def post_process_video(tensor):
    """Post-process video tensor"""
    tensor = tensor.clamp(-1, 1)
    tensor = torch.stack([
        torchvision.utils.make_grid(
            u, nrow=8, normalize=True, value_range=(-1, 1))
        for u in tensor.unbind(2)
    ], dim=1).permute(1, 2, 3, 0)
    tensor = (tensor * 255).type(torch.uint8).cpu()
    
    frames = [
        Image.fromarray(frame.numpy(), mode="RGB")
        for frame in tensor
    ]
    return frames


class WAN:
    def __init__(self, 
                 generation_type: Literal["t2v", "i2v"],
                 model_params: dict,
                 inference: dict,
                 **kwargs):
        """WAN model initialization - standard format"""
        self.model_params = SimpleNamespace(**model_params)
        self.inf = SimpleNamespace(**inference)
        self.generation_type = generation_type
        
        if generation_type == "i2v":
            cfg_name = "i2v-14B"
            cfg = WAN_CONFIGS[cfg_name]
        else:
            raise ValueError("Only i2v generation is supported")
        
        # Distributed initialization
        device, rank = init_distributed_model(self.model_params, cfg, self.inf)
        self.rank = rank
        self.world = int(os.getenv("WORLD_SIZE", 1))
        
        # Initialize WAN model
        self.model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=str(project_root / getattr(self.model_params, 'checkpoint_dir', 'checkpoints/wan2_1')),
            device_id=device,
            rank=rank,
            t5_fsdp=self.model_params.t5_fsdp,
            dit_fsdp=self.model_params.dit_fsdp,
            use_usp=(
                self.model_params.ulysses_size > 1 
                or self.model_params.ring_size > 1
            ),
            t5_cpu=self.model_params.t5_cpu,
        )
        
        print(f"WAN model loading completed - Rank {rank}")
        
        # Prompt expander initialization
        self.use_prompt_extend = getattr(self.inf, "use_prompt_extend", False)
        self.prompt_expander = None
        if self.use_prompt_extend:
            method = getattr(self.inf, "prompt_extend_method", "local_qwen")
            mname = getattr(self.inf, "prompt_extend_model", None)
            is_vl = generation_type == "i2v"
            if method == "dashscope":
                self.prompt_expander = DashScopePromptExpander(
                    model_name=mname, is_vl=is_vl
                )
            elif method == "local_qwen":
                self.prompt_expander = QwenPromptExpander(
                    model_name=mname, is_vl=is_vl, device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                raise ValueError(f"Unknown prompt_extend_method: {method}. Supported methods: 'dashscope', 'local_qwen'")
    
    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        
        """Generate video - core method"""  
        
        if image_path is None:
            raise ValueError("i2v generation requires image_path")
        
        # Load and broadcast image
        if self.rank == 0:
            image = Image.open(image_path).convert("RGB")
            image = dynamic_resize(image)
            print(f"Image loading completed: {image_path}")
            img_obj = [image]
        else:
            img_obj = [None]

        if dist.is_initialized():
            dist.broadcast_object_list(img_obj, src=0)
        image = img_obj[0]
        
        # Prompt extension
        final_prompt = prompt
        if self.use_prompt_extend:
            if self.rank == 0:
                retries = getattr(self.inf, "prompt_extend_retries", 3)
                for attempt in range(retries):
                    try:
                        out = self.prompt_expander(
                            prompt,
                            tar_lang="en",
                            image=image,
                            seed=self.inf.base_seed,
                        )
                        if out.status and out.prompt:
                            final_prompt = out.prompt
                            print(f"✅ Prompt extension successful: {final_prompt}")
                            break
                        print(f"⚠️ Prompt extension failed, retrying {attempt + 1}")
                    except Exception as e:
                        print(f"❌ Prompt extension error: {e}")
                    time.sleep(2)

            # Broadcast extended prompt
            if dist.is_initialized():
                obj = [final_prompt] if self.rank == 0 else [None]
                dist.broadcast_object_list(obj, src=0)
                final_prompt = obj[0]

        print(f"🎬 Starting video generation: {final_prompt}")
        
        # Generate video
        video = self.model.generate(
            input_prompt=final_prompt,
            img=image,
            max_area=MAX_AREA_CONFIGS[self.inf.size],
            frame_num=self.inf.frame_num,
            shift=self.inf.sample_shift,
            sample_solver=self.inf.sample_solver,
            sampling_steps=self.inf.sample_steps,
            guide_scale=self.inf.sample_guide_scale,
            seed=self.inf.base_seed,
            offload_model=self.inf.offload_model,
        )
        
        # Synchronize wait
        if dist.is_initialized():
            dist.barrier()
        print("✅ Video generation completed, starting post-processing...")
        
        # Post-process video
        if self.rank == 0 and video is not None:
            frames = post_process_video(video[None])
        else:
            frames = None
            
        # Broadcast frames
        if dist.is_initialized():
            obj = [frames] if self.rank == 0 else [None]
            dist.broadcast_object_list(obj, src=0)
            frames = obj[0]

        return frames