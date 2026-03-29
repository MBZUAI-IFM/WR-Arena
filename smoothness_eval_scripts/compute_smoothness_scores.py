"""
Smoothness evaluation for multi-round generated videos.

Computes optical flow (SEA-RAFT) between consecutive frames of each round video,
derives per-frame velocity and acceleration magnitudes, and combines them into a
single smoothness score per video. Supports SLURM multi-node + local multi-worker
parallelism, matching the distributed pattern used in the rest of WR-Arena.

Usage:
    python smoothness_eval_scripts/compute_smoothness_scores.py \
        --videos_dir outputs/smoothness_eval/cosmos1 \
        --output_dir outputs/smoothness_eval/cosmos1_scores \
        --raft_ckpt thirdparty/SEA-RAFT/checkpoints/Tartan-C-T-TSKH-spring540x960-M.pth
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: inject thirdparty/SEA-RAFT so its core/ package is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_RAFT_ROOT = _PROJECT_ROOT / "thirdparty" / "SEA-RAFT"
_DEFAULT_RAFT_CFG = _DEFAULT_RAFT_ROOT / "config" / "eval" / "spring-M.json"


def _add_raft_to_path(raft_root: Path) -> None:
    s = str(raft_root)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# SEA-RAFT wrapper (standalone — no worldscore dependency)
# ---------------------------------------------------------------------------

class RaftFlowEstimator:
    """Thin wrapper around SEA-RAFT that returns optical flow as a numpy array."""

    def __init__(self, raft_root: Path, ckpt_path: Path, device: torch.device):
        _add_raft_to_path(raft_root)
        from core.raft import RAFT
        from core.parser import parse_args
        from core.utils.utils import load_ckpt

        cfg_path = raft_root / "config" / "eval" / "spring-M.json"
        raw = argparse.Namespace(cfg=str(cfg_path), path=str(ckpt_path))
        self._args = parse_args(raw)
        self._device = device

        model = RAFT(self._args)
        load_ckpt(model, self._args.path)
        model.to(device).eval()
        self._model = model

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 (H,W,3) → 1×3×H×W float32 RGB tensor on device."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).float().permute(2, 0, 1)[None].to(self._device)

    @torch.no_grad()
    def compute_flow(self, bgr1: np.ndarray, bgr2: np.ndarray) -> np.ndarray:
        """Return optical flow (H, W, 2) float32 from bgr1 → bgr2."""
        t1 = self._to_tensor(bgr1)
        t2 = self._to_tensor(bgr2)
        scale = self._args.scale  # typically -1 → ×0.5 downsample

        def _resize(t, s):
            return F.interpolate(
                t, scale_factor=2 ** s, mode="bilinear", align_corners=False
            )

        t1_up = _resize(t1, scale)
        t2_up = _resize(t2, scale)

        with torch.amp.autocast(device_type="cuda", enabled=self._device.type == "cuda"):
            out = self._model(t1_up, t2_up, iters=self._args.iters, test_mode=True)
        flow = out["flow"][-1]

        # Rescale flow back to original resolution
        flow_down = F.interpolate(
            flow, scale_factor=0.5 ** scale, mode="bilinear", align_corners=False
        ) * (0.5 ** scale)

        return flow_down.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FrameFlowData:
    vmag: np.ndarray  # velocity magnitude (H, W) float32
    amag: np.ndarray  # acceleration magnitude (H, W) float32 — zeros for first frame


@dataclass
class VideoSmoothnessResult:
    video_path: str
    fps: float
    frames: List[FrameFlowData]
    score: float          # overall smoothness score for this video
    vmag_median: float    # median velocity magnitude across all frames
    amag_median: float    # median acceleration magnitude across all frames


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _percentile_clip(arr: np.ndarray, lo: int = 1, hi: int = 99) -> Tuple[float, float]:
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def _smoothness_score(vmag_median: float, amag_median: float, lam: float) -> float:
    """exp_product scoring: v * exp(-λ * a). Higher = smoother motion."""
    return vmag_median * math.exp(-lam * amag_median)


def compute_video_smoothness(
    video_path: Path,
    raft: RaftFlowEstimator,
    lam: float = 1.0,
    fps_override: Optional[float] = None,
) -> Optional[VideoSmoothnessResult]:
    """Compute smoothness score for a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 16.0
    frames_bgr: List[np.ndarray] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames_bgr.append(frame)
    cap.release()

    if len(frames_bgr) < 2:
        return None

    # Compute per-frame flow, velocity, and acceleration
    prev_vmag: Optional[np.ndarray] = None
    flow_frames: List[FrameFlowData] = []

    for i, (f1, f2) in enumerate(zip(frames_bgr[:-1], frames_bgr[1:])):
        flow = raft.compute_flow(f1, f2)
        vmag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)  # (H, W)

        if prev_vmag is None:
            amag = np.zeros_like(vmag)
        else:
            diff = vmag - prev_vmag
            amag = np.abs(diff)

        flow_frames.append(FrameFlowData(vmag=vmag, amag=amag))
        prev_vmag = vmag

    all_vmags = np.concatenate([f.vmag.ravel() for f in flow_frames])
    all_amags = np.concatenate([f.amag.ravel() for f in flow_frames])

    vmag_median = float(np.median(all_vmags))
    amag_median = float(np.median(all_amags))
    score = _smoothness_score(vmag_median, amag_median, lam)

    return VideoSmoothnessResult(
        video_path=str(video_path),
        fps=fps,
        frames=flow_frames,
        score=score,
        vmag_median=vmag_median,
        amag_median=amag_median,
    )


# ---------------------------------------------------------------------------
# Batch processing worker
# ---------------------------------------------------------------------------

def process_instance_dir(
    instance_dir: Path,
    output_dir: Path,
    raft: RaftFlowEstimator,
    lam: float,
) -> Optional[Dict]:
    """Score all round_*.mp4 files in an instance directory."""
    round_videos = sorted(instance_dir.glob("rounds/round_*.mp4"))
    if not round_videos:
        return None

    instance_id = instance_dir.name
    per_round = []

    for i, vid_path in enumerate(round_videos):
        result = compute_video_smoothness(vid_path, raft, lam=lam)
        if result is None:
            continue
        per_round.append({
            "round": i,
            "video": vid_path.name,
            "score": round(result.score, 6),
            "vmag_median": round(result.vmag_median, 6),
            "amag_median": round(result.amag_median, 6),
        })

    if not per_round:
        return None

    mean_score = float(np.mean([r["score"] for r in per_round]))
    record = {
        "instance_id": instance_id,
        "rounds": per_round,
        "mean_score": round(mean_score, 6),
        "timestamp": datetime.utcnow().isoformat(),
    }

    out_path = output_dir / instance_id / "smoothness.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return record


def process_batch(
    instance_dirs: List[Path],
    output_dir: Path,
    raft_root: Path,
    ckpt_path: Path,
    device_str: str,
    lam: float,
    worker_id: int,
) -> List[Dict]:
    """Worker function: load RAFT once, score all assigned instance dirs."""
    device = torch.device(device_str)
    raft = RaftFlowEstimator(raft_root, ckpt_path, device)

    results = []
    for inst_dir in tqdm(instance_dirs, desc=f"Worker {worker_id}", unit="instance"):
        try:
            rec = process_instance_dir(inst_dir, output_dir, raft, lam)
            if rec:
                results.append(rec)
        except Exception as e:
            print(f"  [worker {worker_id}] failed {inst_dir.name}: {e}")

    return results


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def write_summary(output_dir: Path) -> None:
    """Collect all per-instance smoothness.json files and write summary.json."""
    records = []
    for p in sorted(output_dir.rglob("smoothness.json")):
        with open(p) as f:
            records.append(json.load(f))

    if not records:
        return

    all_scores = [r["mean_score"] for r in records]
    summary = {
        "num_instances": len(records),
        "mean_smoothness": round(float(np.mean(all_scores)), 6),
        "std_smoothness": round(float(np.std(all_scores)), 6),
        "min_smoothness": round(float(np.min(all_scores)), 6),
        "max_smoothness": round(float(np.max(all_scores)), 6),
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary ({len(records)} instances):")
    print(f"  mean score : {summary['mean_smoothness']:.4f}")
    print(f"  std        : {summary['std_smoothness']:.4f}")
    print(f"  range      : [{summary['min_smoothness']:.4f}, {summary['max_smoothness']:.4f}]")
    print(f"  saved to   : {output_dir / 'summary.json'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute optical-flow smoothness scores for generated round videos."
    )
    parser.add_argument(
        "--videos_dir", required=True,
        help="Root directory of generated videos, e.g. outputs/smoothness_eval/cosmos1. "
             "Each subdirectory is treated as one instance and must contain rounds/round_*.mp4."
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to write per-instance smoothness.json and summary.json."
    )
    parser.add_argument(
        "--raft_root", default=str(_DEFAULT_RAFT_ROOT),
        help="Path to SEA-RAFT source root (default: thirdparty/SEA-RAFT)."
    )
    parser.add_argument(
        "--raft_ckpt", required=True,
        help="Path to SEA-RAFT .pth checkpoint file."
    )
    parser.add_argument(
        "--lam", type=float, default=1.0,
        help="Lambda for exp_product scoring: score = vmag * exp(-lam * amag). Default: 1.0."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of local parallel worker processes (one GPU each). Default: 4."
    )
    parser.add_argument(
        "--rank", type=int,
        default=int(os.environ.get("SLURM_PROCID", 0)),
        help="Node rank for multi-node SLURM jobs (default: SLURM_PROCID or 0)."
    )
    parser.add_argument(
        "--world_size", type=int,
        default=int(os.environ.get("SLURM_NTASKS", 1)),
        help="Total number of SLURM nodes (default: SLURM_NTASKS or 1)."
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    raft_root = Path(args.raft_root)
    ckpt_path = Path(args.raft_ckpt)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not videos_dir.is_dir():
        raise SystemExit(f"--videos_dir does not exist: {videos_dir}")
    if not ckpt_path.is_file():
        raise SystemExit(f"--raft_ckpt not found: {ckpt_path}\nSee thirdparty/SEA-RAFT/checkpoints/README.md")

    # Collect instance dirs that still need scoring
    all_instance_dirs = sorted(
        d for d in videos_dir.iterdir()
        if d.is_dir() and any(d.glob("rounds/round_*.mp4"))
    )
    pending = [
        d for d in all_instance_dirs
        if not (output_dir / d.name / "smoothness.json").exists()
    ]

    print(f"Found {len(all_instance_dirs)} instances, {len(pending)} pending scoring")

    # Distribute across SLURM nodes
    node_instances = pending[args.rank :: args.world_size]
    print(f"Node {args.rank}/{args.world_size}: {len(node_instances)} instances")

    if not node_instances:
        if args.rank == 0:
            write_summary(output_dir)
        return

    # Distribute node's work across local GPU workers
    num_gpus = torch.cuda.device_count()
    effective_workers = min(args.num_workers, len(node_instances), max(num_gpus, 1))
    chunks = [node_instances[i::effective_workers] for i in range(effective_workers)]

    print(f"Using {effective_workers} worker(s) across {num_gpus} GPU(s)")

    if effective_workers == 1:
        # Single-worker: run in-process (easier debugging)
        device_str = "cuda:0" if num_gpus > 0 else "cpu"
        results = process_batch(chunks[0], output_dir, raft_root, ckpt_path, device_str, args.lam, 0)
    else:
        mp.set_start_method("spawn", force=True)
        worker_args = [
            (chunks[i], output_dir, raft_root, ckpt_path,
             f"cuda:{i % num_gpus}" if num_gpus > 0 else "cpu",
             args.lam, i)
            for i in range(effective_workers)
        ]
        with mp.Pool(effective_workers) as pool:
            all_results = pool.starmap(process_batch, worker_args)
        results = [r for batch in all_results for r in batch]

    print(f"Scored {len(results)} instances on this node")

    # Only rank 0 writes the aggregate summary
    if args.rank == 0:
        write_summary(output_dir)


if __name__ == "__main__":
    main()
