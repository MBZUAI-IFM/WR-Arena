import os
import sys
from pathlib import Path
from typing import Literal, Optional
from types import SimpleNamespace
import time
import torch
import random
from PIL import Image

import torch.distributed as dist
import torchvision
from torchvision import transforms

import thirdparty.wan2_2 as wan
from thirdparty.wan2_2.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
from thirdparty.wan2_2.utils.utils import save_video, str2bool
from thirdparty.wan2_2.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parents[1]))

def init_distributed_model(args, cfg, inf):
    rank        = int(os.getenv("RANK", 0))
    world_size  = int(os.getenv("WORLD_SIZE", 1))
    local_rank  = int(os.getenv("LOCAL_RANK", 0))
    device      = local_rank # if rank != -1 else rank
    
    print(f"{rank=}, {world_size=}, {device=}")
    
    if inf.offload_model is None:
        inf.offload_model = False if world_size > 1 else True
        print(
            f"offload_model is not specified, set to {inf.offload_model}."
        )
        
    already_init = dist.is_available() and dist.is_initialized()
    if world_size > 1:
        if not already_init:
            print("-----------see if initialized")
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
            )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1
        ), f"sequence parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, f"The number of ulysses_size should be equal to the world size."
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        
    if not hasattr(args, 'base_seed'):
        args.base_seed = random.randint(0, sys.maxsize)        
        
    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]
        
    return device, rank

def dynamic_resize(img, video_size=(832, 480)):
    width, height = img.size
    t_width, t_height = video_size
    k = max(t_width / width, t_height / height)
    new_width, new_height = int(width * k), int(height * k)
    trans = transforms.Compose(
        [
            transforms.Resize((new_height, new_width)),
            transforms.CenterCrop(video_size[::-1]),
        ]
    )
    return trans(img)


def post_process_video(
        tensor,
        nrow=8,
        normalize=True,
        value_range=(-1, 1),
    ):
    tensor = tensor.clamp(min(value_range), max(value_range))
    tensor = torch.stack([
        torchvision.utils.make_grid(
                u, nrow=nrow, normalize=normalize, value_range=value_range)
            for u in tensor.unbind(2)
        ],
        dim=1).permute(1, 2, 3, 0)
    tensor = (tensor * 255).type(torch.uint8).cpu()
    
    frames: list[Image.Image] = [
        Image.fromarray(frame.numpy(), mode="RGB")  
        for frame in tensor
    ]
    
    return frames
    
class WAN:
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

        if generation_type == "i2v":
            cfg_name                    = "i2v-A14B"                     
            cfg                         = WAN_CONFIGS[cfg_name]
        else:
            raise ValueError(
                "The evaluation script is setup for i2v inference of WAN only."
            )

        device, rank = init_distributed_model(self.model_params, cfg, self.inf)
        
        self.rank  = rank                
        self.world = int(os.getenv("WORLD_SIZE", 1))
        
        self.model = wan.WanI2V(
            config              = cfg,
            checkpoint_dir      = str(project_root / getattr(self.model_params, 'checkpoint_dir', 'checkpoints/wan2_2')),
            device_id           = device,
            rank                = rank,
            t5_fsdp             = self.model_params.t5_fsdp,
            dit_fsdp            = self.model_params.dit_fsdp,
            use_sp              = (self.model_params.ulysses_size > 1),
            t5_cpu              = self.model_params.t5_cpu,
            convert_model_dtype = False,
        )

        self.use_prompt_extend = getattr(self.inf, "use_prompt_extend", False)
        self.prompt_expander   = None
        if self.use_prompt_extend:
            method = getattr(self.inf, "prompt_extend_method", "local_qwen")
            mname  = getattr(self.inf, "prompt_extend_model", None)
            is_vl  = generation_type == "i2v"
            if method == "dashscope":
                self.prompt_expander = DashScopePromptExpander(
                    model_name=mname, is_vl=is_vl
                )
            elif method == "local_qwen":
                self.prompt_expander = QwenPromptExpander(
                    model_name=mname, task=cfg_name, is_vl=is_vl, device="cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                raise ValueError(f"Unknown prompt_extend_method: {method}")

    def generate_video(self, prompt: str, image_path: Optional[str] = None):
        
        if image_path is None:
            raise ValueError("image_path is required for i2v generation")
        
        if self.rank == 0:
            image = Image.open(image_path).convert("RGB")
            image = dynamic_resize(image)
            img_obj = [image]                       
        else:
            img_obj = [None]

        if dist.is_initialized():
            dist.broadcast_object_list(img_obj, src=0)
        image = img_obj[0]      

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
                        print(f"Prompt expander Output: {out}")
                        if out.status and out.prompt:
                            final_prompt = out.prompt
                            print("Prompt-extender succeeded.",
                                        original=prompt, extended=final_prompt)
                            break
                        print(
                            "Prompt-extender returned empty / refused, retrying …")
                    except Exception as e:
                        print("Prompt-extender error – retrying")
                    time.sleep(5)

            if dist.is_initialized():
                obj = [final_prompt] if self.rank == 0 else [None]
                dist.broadcast_object_list(obj, src=0)
                final_prompt = obj[0]
        
        print(f"Generating video ... using {final_prompt}")
            
        video = self.model.generate(
            input_prompt     = final_prompt,
            img              = image,
            max_area         = MAX_AREA_CONFIGS[self.inf.size],
            frame_num        = self.inf.frame_num,
            shift            = self.inf.sample_shift,
            sample_solver    = self.inf.sample_solver,
            sampling_steps   = self.inf.sample_steps,
            guide_scale      = self.inf.sample_guide_scale,
            seed             = self.inf.base_seed,
            offload_model    = self.inf.offload_model,
        )
        
        # make sure everyone is done before leaving the function
        if dist.is_initialized():
            dist.barrier()
        print("Post-processing WAN generated video.")
        
        
        # --- all ranks reach this together because WanI2V already calls barrier
        if self.rank == 0 and video is not None:
            frames = post_process_video(video[None])      # list[PIL] on rank-0
        else:
            frames = None                                 # placeholder
            
        # ---------- broadcast the frames list --------------------------------
        if dist.is_initialized():
            obj = [frames] if self.rank == 0 else [None]
            dist.broadcast_object_list(obj, src=0)
            frames = obj[0]      

        return frames