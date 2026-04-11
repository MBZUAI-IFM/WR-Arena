import argparse
import json
import os
import sys
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import tempfile
import mediapy as media
from PIL import Image

# Add project root to path
project_root = Path(os.getenv('PROJECT_ROOT', Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(project_root))

def load_samples(file_path: str):
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def preprocess_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                print(f"Converting {img.mode} to RGB")
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    rgb_img = img.convert('RGB')
                    rgb_img.save(tmp.name)
                    return tmp.name
            return image_path
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return image_path

def save_video(video_data, output_path: str, sample_id: str, model_name: str, fps: int = 24):
    safe_id = sample_id.replace('/', '_').replace('\\', '_')
    output_file = Path(output_path) / f"{safe_id}.mp4"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if model_name == "cosmos2":
        from imaginaire.utils.io import save_image_or_video
        save_image_or_video(video_data, str(output_file), fps=fps)
    else:
        media.write_video(str(output_file), video_data, fps=fps)
    
    return str(output_file)

def process_sample(sample, model, model_name, output_dir):
    try:
        sample_id = sample['id']
        prompt = sample['prompt']
        source_image = sample['source']
        
        print(f"Processing: {sample_id}")
        
        if not os.path.exists(source_image):
            print(f"Source not found: {source_image}")
            return False
        
        # Preprocess image
        processed_image = preprocess_image(source_image)
        temp_created = (processed_image != source_image)
        
        try:
            # Generate video
            video_data = model.generate_video(prompt=prompt, input_path=processed_image)
            
            if video_data is None:
                print(f"Generation failed: {sample_id}")
                return False
            
            # Save video
            fps = getattr(model, 'fps', 24)
            output_file = save_video(video_data, str(output_dir), sample_id, model_name, fps=fps)
            print(f"Saved: {output_file}")
            return True
            
        finally:
            # Clean up temp file
            if temp_created:
                try:
                    Path(processed_image).unlink(missing_ok=True)
                except:
                    pass
                    
    except Exception as e:
        print(f"Error processing {sample.get('id', 'unknown')}: {e}")
        return False

def worker_process(device_id, task_queue, result_queue, model_name, output_dir):
    try:
        # Set GPU
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            print(f"Worker {device_id}: Using GPU {device_id}")
        
        # Initialize model - dynamic import
        if model_name == "cosmos1":
            from simulative_reasoning_planning_scripts.step_wise_simulation_scripts.cosmos1 import Cosmos1
            model = Cosmos1()
        elif model_name == "cosmos2":
            from simulative_reasoning_planning_scripts.step_wise_simulation_scripts.cosmos2 import Cosmos2
            model = Cosmos2()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        print(f"Worker {device_id}: Model ready")
        
        # Process samples
        while True:
            try:
                sample = task_queue.get(timeout=5.0)
                if sample is None:
                    break
                
                success = process_sample(sample, model, model_name, output_dir)
                result_queue.put({'worker_id': device_id, 'success': success, 'sample_id': sample.get('id')})
                
            except Exception as e:
                if "Empty" not in str(e):
                    print(f"Worker {device_id} error: {e}")
                break
                
    except Exception as e:
        print(f"Worker {device_id} init error: {e}")

def main():
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_file", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="cosmos1")
    parser.add_argument("--node_id", type=int, default=0)
    parser.add_argument("--total_nodes", type=int, default=1)
    parser.add_argument("--gpu_per_node", type=int, default=8)
    
    args = parser.parse_args()
    
    print(f"Model: {args.model_name}")
    print(f"Node: {args.node_id + 1}/{args.total_nodes}")
    print(f"GPUs: {args.gpu_per_node}")
    
    # Load samples
    all_samples = load_samples(args.samples_file)
    print(f"Total samples: {len(all_samples)}")
    
    # Split data across nodes
    chunk_size = (len(all_samples) + args.total_nodes - 1) // args.total_nodes
    start_idx = args.node_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(all_samples))
    samples = all_samples[start_idx:end_idx]
    
    if not samples:
        print("No samples for this node")
        return
    
    print(f"Processing {len(samples)} samples ({start_idx}-{end_idx-1})")
    
    # Setup output directory with model name
    model_output_path = os.path.join(args.output_path, args.model_name)
    output_dir = Path(model_output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Multi-GPU processing
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Add samples to queue
    for sample in samples:
        task_queue.put(sample)
    
    # Add stop signals
    for _ in range(args.gpu_per_node):
        task_queue.put(None)
    
    # Start workers
    processes = []
    for device_id in range(args.gpu_per_node):
        p = ctx.Process(target=worker_process, args=(device_id, task_queue, result_queue, args.model_name, str(output_dir)))
        p.start()
        processes.append(p)
    
    # Monitor progress
    successful = 0
    failed = 0
    
    with tqdm(total=len(samples), desc="Generating videos") as pbar:
        while successful + failed < len(samples):
            try:
                result = result_queue.get(timeout=10.0)
                if result.get('success', False):
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
            except:
                continue
    
    # Wait for workers
    for p in processes:
        p.join()
    
    print(f"Completed: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()