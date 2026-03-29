"""
prepare_worldscore_dirs.py

Reorganises the flat video output from generate_videos.py into the
hierarchical directory structure expected by the WorldScore evaluator, and
writes the per-instance metadata files (image_data.json, camera_data.json)
that the evaluator reads.

Input layout (produced by generate_videos.py):
    <videos_root>/<instance_id>/
        frames/
            frame_000.png, frame_001.png, ...
        input_image.png

Output layout (expected by WorldScore MultiRoundEvaluator):
    <output_root>/worldscore_output/static/<visual_style>/<scene_type>/<category>/<ws_instance_id>/
        frames/
            frame_000.png, ...
        input_image.png
        image_data.json
        camera_data.json   (stub — real camera tracking requires DROID-SLAM)

Usage:
    python generation_consistency_eval_scripts/prepare_worldscore_dirs.py \\
        --videos_root outputs/generation_consistency_eval/pan \\
        --dataset_json datasets/generation_consistency_eval/samples.json \\
        --output_root outputs/generation_consistency_eval/pan_eval
"""

import argparse
import json
import shutil
from pathlib import Path


def compute_anchor_frame_idx(total_frames: int, num_scenes: int) -> list:
    """
    Compute per-round boundary frame indices.

    generate_videos.py stitches rounds by skipping the first frame of each
    subsequent round, so the frame layout is:
        round 0:  frames_per_round frames
        round i:  (frames_per_round - 1) new frames

    total_frames = frames_per_round + (num_scenes - 1) * (frames_per_round - 1)
                 = 1 + num_scenes * (frames_per_round - 1)

    => frames_per_round = (total_frames - 1) // num_scenes + 1
    => anchor_frame_idx[i] = i * (frames_per_round - 1)  for i in 0..num_scenes
    """
    step = (total_frames - 1) // num_scenes
    return [i * step for i in range(num_scenes + 1)]


def prepare_instance(instance: dict, videos_root: Path, output_root: Path) -> bool:
    """
    Prepare one instance directory.  Returns True on success.
    """
    gc_id = instance["id"]                              # e.g. "gc_011"
    ws_instance_id = Path(instance["image_path"]).stem  # e.g. "001_1"
    visual_style  = instance["visual_style"]
    scene_type    = instance["scene_type"]
    category      = instance["category"]

    src_dir = videos_root / gc_id
    if not src_dir.exists():
        print(f"  [skip] {gc_id}: source directory not found")
        return False

    frames_src = src_dir / "frames"
    input_image_src = src_dir / "input_image.png"

    if not frames_src.exists() or not input_image_src.exists():
        print(f"  [skip] {gc_id}: frames/ or input_image.png missing")
        return False

    # Count actual generated frames
    frame_files = sorted(
        p for p in frames_src.iterdir()
        if p.suffix in (".png", ".jpg")
    )
    total_frames = len(frame_files)
    if total_frames == 0:
        print(f"  [skip] {gc_id}: no frames found")
        return False

    prompt_list  = instance["prompt_list"]
    num_scenes   = len(prompt_list)
    content_list = instance.get("content_list", [])
    camera_path  = instance.get("camera_path", [])

    anchor_frame_idx = compute_anchor_frame_idx(total_frames, num_scenes)

    # Create target directory
    tgt_dir = (
        output_root / "worldscore_output" / "static"
        / visual_style / scene_type / category / ws_instance_id
    )
    if tgt_dir.exists():
        print(f"  [skip] {ws_instance_id}: target already exists")
        return False

    tgt_dir.mkdir(parents=True, exist_ok=True)

    # Copy frames
    frames_tgt = tgt_dir / "frames"
    shutil.copytree(str(frames_src), str(frames_tgt))

    # Copy input image
    shutil.copy2(str(input_image_src), str(tgt_dir / "input_image.png"))

    # Write image_data.json
    image_data = {
        "total_frames":       total_frames,
        "num_scenes":         num_scenes,
        "anchor_frame_idx":   anchor_frame_idx,
        "prompt_list":        prompt_list,
        "content_list":       content_list,
        "camera_path":        camera_path,
    }
    with open(tgt_dir / "image_data.json", "w", encoding="utf-8") as f:
        json.dump(image_data, f, indent=4)

    # Write stub camera_data.json
    # Real camera poses require DROID-SLAM.  Without them, camera_control and
    # 3d_consistency aspects will fail gracefully with a fallback score.
    camera_data = {"cameras_interp": [], "scale": 1.0}
    with open(tgt_dir / "camera_data.json", "w", encoding="utf-8") as f:
        json.dump(camera_data, f, indent=4)

    print(f"  {gc_id} → {ws_instance_id}: {total_frames} frames, {num_scenes} rounds")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Reorganise flat video output into WorldScore directory structure."
    )
    parser.add_argument("--videos_root", required=True,
                        help="Root of flat video output from generate_videos.py")
    parser.add_argument("--dataset_json", required=True,
                        help="Path to datasets/generation_consistency_eval/samples.json")
    parser.add_argument("--output_root", required=True,
                        help="Destination root for WorldScore directory structure")
    args = parser.parse_args()

    videos_root = Path(args.videos_root)
    output_root = Path(args.output_root)
    dataset_json = Path(args.dataset_json)

    with open(dataset_json, encoding="utf-8") as f:
        instances = json.load(f)

    print(f"Preparing {len(instances)} instances...")
    print(f"  source : {videos_root}")
    print(f"  dest   : {output_root}")

    ok = skipped = 0
    for instance in instances:
        if prepare_instance(instance, videos_root, output_root):
            ok += 1
        else:
            skipped += 1

    print(f"\nDone: {ok} prepared, {skipped} skipped.")


if __name__ == "__main__":
    main()
