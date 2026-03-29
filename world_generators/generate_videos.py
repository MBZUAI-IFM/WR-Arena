import json
import os
import sys
import argparse
from pathlib import Path
import torch
import torch.distributed as dist
from tqdm import tqdm
import hydra
import omegaconf
import cv2
import numpy as np
import shutil
from typing import List
from PIL import Image

project_root = Path(__file__).resolve().parents[1]
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def setup_distributed(use_slurm=True):
    """Initialize distributed environment"""
    if not use_slurm:
        return 0, 1, 0
        
    if not dist.is_available():
        return 0, 1, 0
    
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    rank = dist.get_rank()
    world_size = dist.get_world_size() 
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def load_config(model_name: str):
    """Load model configuration"""
    config_path = Path(__file__).resolve().parent / "configs" / f"{model_name}.yaml"
    config = omegaconf.OmegaConf.load(config_path)
    return config

def create_video_from_frames(frames: List[Image.Image], output_path: Path, fps: int = 16):
    """Convert frames to MP4 video"""
    if not frames:
        print("No frames available to convert to video")
        return None
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        width, height = frames[0].size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        for frame in frames:
            frame_array = np.array(frame)
            if len(frame_array.shape) == 3 and frame_array.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_array
            out.write(frame_bgr)
        
        out.release()
        return output_path
    
    except Exception as e:
        print(f"Video creation failed: {e}")
        return None

def create_round_videos(frames: List[Image.Image], output_dir: Path, frames_per_round: int, fps: int, overlap: int = 1):
    """Split frames into multiple overlapping video segments"""
    if not frames:
        return []
    
    output_dir.mkdir(parents=True, exist_ok=True)
    created_videos = []
    
    step = frames_per_round - overlap
    round_num = 0
    
    for start_idx in range(0, len(frames) - overlap, step):
        end_idx = min(start_idx + frames_per_round, len(frames))
        
        if end_idx <= start_idx:
            break
        
        round_frames = frames[start_idx:end_idx]
        video_path = output_dir / f"round_{round_num:03d}.mp4"
        
        if created_video := create_video_from_frames(round_frames, video_path, fps):
            created_videos.append(created_video)
        
        round_num += 1
        
        if end_idx >= len(frames):
            break
    
    return created_videos

def process_instance(generator, instance, output_root, rank, config, gen_rank=None, image_root=None):
    """Process single instance"""
    raw_image_path = instance["image_path"]
    if image_root:
        image_path = Path(image_root) / raw_image_path
    else:
        image_path = Path(os.getenv("DATA_PATH", ".")) / raw_image_path
    instance_id = instance["id"].split("_", 1)[-1]
    output_dir = Path(output_root) / instance_id
    complete_video_path = output_dir / "video" / "complete_video.mp4"
    
    skip_processing = False
    if rank == 0:
        if complete_video_path.exists():
            skip_processing = True
            if gen_rank == 0 or gen_rank is None:
                print(f"Skipping {instance_id}: already exists")
        else:
            if gen_rank == 0 or gen_rank is None:
                print(f"Processing: {instance_id}")
    
    # Sync skip decision across all ranks to prevent partial processing  
    if dist.is_initialized():
        skip_obj = [skip_processing] if rank == 0 else [None]
        dist.broadcast_object_list(skip_obj, src=0)
        skip_processing = skip_obj[0]
    
    if skip_processing:
        return 0

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["frames", "video", "rounds"]:
            (output_dir / subdir).mkdir(exist_ok=True)
        input_image_copy = output_dir / "input_image.png"
        shutil.copy2(image_path, input_image_copy)
        print(f"Input image backup: {input_image_copy}")

    prompt_list = instance.get("prompt_list", [])
    current_image = str(image_path)
    temp_files = []
    
    if isinstance(prompt_list, list) and generator.__class__.__name__ == "PAN":
        print("starting multi-prompt PAN generate_video")
        all_frames = generator.generate_video(prompt_list, image_path)
        print("ending multi-prompt PAN generate_video")
    else:
        all_frames = []
        for i, prompt in enumerate(prompt_list):
            if rank == 0 and (gen_rank == 0 or gen_rank is None):
                print(f"  Step {i+1}/{len(prompt_list)}: {prompt[:50]}...")
            frames = generator.generate_video(prompt=prompt, image_path=current_image)
            
            if frames:
                if i == 0:
                    all_frames.extend(frames)
                else:
                    all_frames.extend(frames[1:])  # Skip first frame to avoid duplication between rounds
                if rank == 0:
                    first_frame = frames[0]
                    input_path = output_dir / f"input_image_{i+1}.png"
                    first_frame.save(input_path)
                    if i < len(prompt_list) - 1:
                        model_name = config.get('model_name', '')

                        if model_name == "cosmos-predict1" or model_name == "cosmos-predict2":
                            temp_video_path = output_dir / f"temp_input_round_{i+1}.mp4"
                            fps = config.get('inference', {}).get('fps', config.get('helper_config', {}).get('fps', 16))
                            print(f"🎬 Creating temporary video for next round: {temp_video_path.name}")
                            create_video_from_frames(frames, temp_video_path, fps)
                            temp_files.append(str(temp_video_path))
                            next_image = str(temp_video_path)
                            print(f"Next round will use video input: {len(frames)} frames at {fps}fps")
                        else:
                            last_frame = frames[-1]
                            temp_path = output_dir / "temp_input.png"
                            last_frame.save(temp_path)
                            temp_files.append(str(temp_path))
                            next_image = str(temp_path)
                            
                    else:
                        next_image = current_image
                else:
                    next_image = None
                if i < len(prompt_list) - 1:
                    # Sync next_image path across all ranks for consistent chaining
                    if dist.is_initialized():
                        image_obj = [next_image] if rank == 0 else [None]
                        dist.broadcast_object_list(image_obj, src=0)
                        current_image = image_obj[0]

    if rank == 0 and all_frames:
        print(f"Saving {len(all_frames)} frames...")
        frames_dir = output_dir / "frames"
        for i, frame in enumerate(all_frames):
            frame.save(frames_dir / f"frame_{i:03d}.png")
            
        fps = config.get('inference', {}).get('fps', config.get('helper_config', {}).get('fps', 16))
        frame_num = config.get('inference', {}).get('frame_num', 9)
        complete_video = create_video_from_frames(all_frames, output_dir / "video" / "complete_video.mp4", fps)
        round_videos = create_round_videos(all_frames, output_dir / "rounds", frame_num, fps)
        
        for temp_file in temp_files:
            try:
                Path(temp_file).unlink(missing_ok=True)
            except:
                pass
        if gen_rank == 0 or gen_rank is None:
            print(f"{instance_id}: {len(all_frames)} frames, {len(round_videos)} segments")
    
    return len(all_frames) if all_frames else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--prompt_set", required=True)
    parser.add_argument("--gen_rank", type=int, default=None)
    parser.add_argument("--gen_world_size", type=int, default=None)
    slurm_group = parser.add_mutually_exclusive_group()
    slurm_group.add_argument("--use-slurm", action="store_true", default=True, help="Use SLURM distributed processing (default)")
    slurm_group.add_argument("--no-slurm", action="store_true", help="Use API-only processing, no distributed")
    parser.add_argument("--num-jobs", type=int, default=1, help="Number of parallel jobs for batch processing")
    parser.add_argument("--batch-index", type=int, default=0, help="Current batch index (0-based)")
    parser.add_argument("--output_root", default=None,
                        help="Override the output root directory from the model config.")
    parser.add_argument("--image_root", default=None,
                        help="Root directory prepended to relative image_path values in the dataset. "
                             "Falls back to the DATA_PATH environment variable if not set.")

    args = parser.parse_args()
    config = load_config(args.model_name)
    use_slurm = not args.no_slurm
    rank, world_size, local_rank = setup_distributed(use_slurm)
    
    if use_slurm:
        startup_condition = rank == 0 and (args.gen_rank == 0 if args.gen_rank is not None else True)
    else:
        startup_condition = True
        
    if startup_condition:
        mode_str = "distributed generation" if use_slurm else "API-based generation"
        print(f"Starting {mode_str}")
        print(f"Model: {args.model_name}")
        print(f"Data: {args.prompt_set}")
    
    with open(args.prompt_set, 'r') as f:
        all_instances = json.load(f)
    
    incomplete_instances = []
    completed_count = 0
    if rank == 0 or not use_slurm:
        output_root = Path(args.output_root) if args.output_root else \
            project_root / config.get('helper_config', {}).get('output_root', 'outputs')
        for instance in all_instances:
            instance_id = instance["id"].split("_", 1)[-1]  # Must match process_instance parsing
            complete_video_path = Path(output_root) / instance_id / "video" / "complete_video.mp4"
            
            if complete_video_path.exists():
                completed_count += 1
            else:
                incomplete_instances.append(instance)
        should_report = (args.gen_rank == 0 if args.gen_rank is not None else True) or not use_slurm
        
        if should_report:
            print(f"Status: {completed_count} completed, {len(incomplete_instances)} pending")
            
    if use_slurm and dist.is_initialized():
        if rank == 0:
            obj = [incomplete_instances, completed_count]
        else:
            obj = [None, None]
        dist.broadcast_object_list(obj, src=0)
        incomplete_instances, completed_count = obj
        
    if use_slurm and args.gen_rank is not None and args.gen_world_size is not None:
        instances = incomplete_instances[args.gen_rank::args.gen_world_size] if incomplete_instances else []
        
        if rank == 0 and args.gen_rank == 0 and incomplete_instances:
            print(f"Distributing {len(incomplete_instances)} pending instances across {args.gen_world_size} nodes")
            for node_id in range(args.gen_world_size):
                node_instances = incomplete_instances[node_id::args.gen_world_size]
                print(f"   Node{node_id}: {len(node_instances)} instances")
                   
    elif not use_slurm and args.num_jobs > 1:
        instances = incomplete_instances[args.batch_index::args.num_jobs] if incomplete_instances else []
        if args.batch_index == 0:
            print(f"Distributing {len(incomplete_instances)} pending instances across {args.num_jobs} parallel jobs")
            for job_id in range(args.num_jobs):
                job_instances = incomplete_instances[job_id::args.num_jobs]
                print(f"   Job{job_id}: {len(job_instances)} instances")
        print(f"Job{args.batch_index}: {len(instances)} instances")  
        
    else:
        instances = incomplete_instances
        if (rank == 0 and use_slurm) or not use_slurm:
            mode_str = "API processing" if not use_slurm else "Single node"
            print(f"{mode_str}: {len(instances)} instances")
            
    all_completed = len(incomplete_instances) == 0
    
    if all_completed and rank == 0 and args.gen_rank == 0:
        print(f"All {len(all_instances)} instances already completed!")
        
    if all_completed:
        should_print = ((rank == 0 and args.gen_rank == 0) if use_slurm else True)
        if should_print:
            print(f"All {len(all_instances)} instances already completed!")
        if use_slurm and dist.is_initialized():
            dist.destroy_process_group()
        return
    
    should_print_loading = ((rank == 0 and args.gen_rank == 0) if use_slurm else True)
    
    if should_print_loading:
        print("Loading model...")
    model_config = {k: v for k, v in config.items() if k != 'helper_config'}
    generator = hydra.utils.instantiate(model_config)
    
    if should_print_loading:
        print("Model loaded")
    output_root = args.output_root or config.get('helper_config', {}).get('output_root', 'outputs')

    total_frames = 0
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    show_progress = (rank == 0 and (args.gen_rank == 0 if args.gen_rank is not None else True)) or not use_slurm
    count_results = (rank == 0) or not use_slurm
    
    for instance in tqdm(instances, desc="🎬 Processing", unit="video", disable=not show_progress):
        try:
            frames = process_instance(generator, instance, output_root, rank, config, args.gen_rank, args.image_root)
            if count_results:
                total_frames += frames
                if frames > 0:
                    processed_count += 1
                else:
                    skipped_count += 1
        except Exception as e:
            if count_results:
                failed_count += 1
                print(f"Processing failed: {str(e)}")
                
    if use_slurm:
        if rank == 0 and (args.gen_rank == 0 if args.gen_rank is not None else True):
            print(f"\n Generation completed!")
            print(f"Total: {len(all_instances)} instances ({completed_count} pre-completed, {processed_count} newly processed)")
            if args.gen_rank is not None:
                print(f"Node{args.gen_rank}: {processed_count}P/{skipped_count}S/{failed_count}F, {total_frames} frames")
            else:
                print(f"Single node: {processed_count}P/{skipped_count}S/{failed_count}F, {total_frames} frames")
        elif rank == 0 and args.gen_rank is not None:
            print(f"   ✅ Node{args.gen_rank}: {processed_count}P/{skipped_count}S/{failed_count}F, {total_frames} frames")
    else:
        if args.num_jobs > 1:
            print(f"\n API Batch {args.batch_index} completed!")
            print(f"Job{args.batch_index}: {processed_count}P/{skipped_count}S/{failed_count}F, {total_frames} frames")
        else:
            print(f"\n API Generation completed!")
            print(f"Total: {len(all_instances)} instances ({completed_count} pre-completed, {processed_count} newly processed)")
            print(f"API processing: {processed_count}P/{skipped_count}S/{failed_count}F, {total_frames} frames")
    
    if use_slurm and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
