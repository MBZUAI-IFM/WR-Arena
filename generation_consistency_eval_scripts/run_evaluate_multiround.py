# Adapted from WorldScore (https://github.com/haoyi-duan/WorldScore)
# Original work: Copyright (c) 2025 Haoyi Duan, MIT License
# Modifications:
#   - Replaced fire CLI with argparse (consistent with WR-Arena conventions)
#   - Added --runs_root / --model_configs_dir CLI overrides
#   - Removed worldscore model-registry dependency (check_model / get_model2type)
#   - Removed multiround_smoothness (VFIMamba) custom aspect
#   - worldscore_list["custom"] left empty; final score uses only static aspects

import argparse
import json
import os
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from worldscore.common.utils import print_banner

# ──────────────────────────────────────────────────────────────────────────────
# Aspect catalogue (static only; dynamic not evaluated here)
# ──────────────────────────────────────────────────────────────────────────────
worldscore_list = {
    "static": [
        "camera_control",
        "object_control",
        "content_alignment",
        "3d_consistency",
        "photometric_consistency",
        "style_consistency",
        "subjective_quality",
    ],
    "dynamic": [
        "motion_accuracy",
        "motion_magnitude",
        "motion_smoothness",
    ],
    "custom": [],   # VFIMamba-based multiround_smoothness removed
}


# ──────────────────────────────────────────────────────────────────────────────
# Score aggregation helpers  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
def _fill_empty_aspects(scores, fill_value=0.0):
    try:
        rounds = next(len(v) for v in scores.values() if v)
    except StopIteration:
        return scores
    for aspect, value in scores.items():
        if not value:
            scores[aspect] = [fill_value] * rounds
    return scores


def _mean_same_shape(values):
    arr = np.asarray(values)
    mean = arr.mean(axis=0)
    return float(mean) if np.isscalar(mean) else mean.tolist()


def _round_two(x, percentage: bool = False):
    if percentage:
        x = np.asarray(x) * 100.0
    if np.isscalar(x):
        return round(float(x), 2)
    return [round(float(v), 2) for v in x]


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _collapse_last_dim(arr: np.ndarray) -> np.ndarray:
    return arr if arr.ndim < 3 else arr.mean(axis=-1)


def calculate_mean_scores(metrics_results, visual_movement, scores, calculate_raw_score=False):
    aspect_list = worldscore_list[visual_movement] + worldscore_list["custom"]

    for aspect, aspect_metrics in metrics_results.items():
        if aspect not in aspect_list or not aspect_metrics:
            continue

        per_metric_means = []
        for metric_name, metric_scores in aspect_metrics.items():
            if not metric_scores:
                continue
            key = "score" if calculate_raw_score else "score_normalized"
            arr = _as_array([m[key] for m in metric_scores])
            arr = _collapse_last_dim(arr)
            mean_per_round = arr.mean(axis=0)
            per_metric_means.append(mean_per_round)

        if not per_metric_means:
            continue

        aspect_mean = np.stack(per_metric_means, axis=0).mean(axis=0)
        if calculate_raw_score:
            scores[aspect] = per_metric_means
        else:
            scores[aspect] = _round_two(aspect_mean, percentage=True)

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Build config dict without depending on WorldScore's base_config / env vars
# ──────────────────────────────────────────────────────────────────────────────
def _load_model_config(model_name: str, model_configs_dir: Path) -> dict:
    """Load model-specific YAML (frames, generate_type).  Returns {} on miss."""
    cfg_path = model_configs_dir / f"{model_name}.yaml"
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    return {}


def build_config(args) -> dict:
    model_cfg = _load_model_config(
        args.model_name, Path(args.model_configs_dir)
    )
    return {
        "model":           args.model_name,
        "runs_root":       args.runs_root,
        "output_dir":      "worldscore_output",
        "visual_movement": args.visual_movement,
        "dataset_root":    "",        # not needed for static eval
        "benchmark_root":  "",
        "focal_length":    500,
        "frames":          model_cfg.get("frames", 9),
        "generate_type":   model_cfg.get("generate_type", "i2v"),
        "regenerate":      False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate scores across all evaluated instances
# ──────────────────────────────────────────────────────────────────────────────
def calculate_worldscore(config: dict, visual_movement_list: list,
                         calculate_raw_score: bool = False):
    model_name = config["model"]

    scores = {
        aspect: []
        for movement in worldscore_list
        for aspect in worldscore_list[movement]
    }

    for visual_movement in visual_movement_list:
        root_path = Path(config["runs_root"]) / config["output_dir"] / visual_movement
        metrics_results = defaultdict(lambda: defaultdict(list))

        if visual_movement == "static":
            for visual_style_dir in sorted(root_path.iterdir()):
                if not visual_style_dir.is_dir():
                    continue
                for scene_type_dir in sorted(visual_style_dir.iterdir()):
                    if not scene_type_dir.is_dir():
                        continue
                    for category_dir in sorted(scene_type_dir.iterdir()):
                        if not category_dir.is_dir():
                            continue
                        for instance_dir in sorted(category_dir.iterdir()):
                            if not instance_dir.is_dir():
                                continue
                            result_file = instance_dir / "evaluation_multiround.json"
                            if result_file.exists():
                                try:
                                    with open(result_file, encoding="utf-8") as f:
                                        inst = json.load(f)
                                    for aspect, aspect_scores in inst.items():
                                        for metric_name, metric_score in aspect_scores.items():
                                            if metric_score:
                                                metrics_results[aspect][metric_name].append(
                                                    metric_score
                                                )
                                except Exception:
                                    pass

        calculate_mean_scores(metrics_results, visual_movement, scores, calculate_raw_score)

    scores = _fill_empty_aspects(scores, fill_value=0.0)

    out_dir = Path(config["runs_root"]) / config["output_dir"]

    if not calculate_raw_score:
        print(f"Scores: {scores}")
        metrics_static = worldscore_list["static"]
        worldscore_static = _round_two(_mean_same_shape([scores[m] for m in metrics_static]))

        print_banner("RESULT")
        print(model_name)
        print(f"WorldScore-Static: {worldscore_static}")

        scores["WorldScore-Static"] = worldscore_static
        out_path = out_dir / "worldscore_multiround.json"
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"Results saved to {out_path}")
    else:
        print_banner("RESULT")
        print(model_name)
        print(f"Raw Scores: {scores}")
        out_path = out_dir / "worldscore_multiround_raw.json"
        with open(out_path, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"Results saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Per-instance evaluation
# ──────────────────────────────────────────────────────────────────────────────
def run_evaluation(args: Namespace, config: dict, num_jobs: int, use_slurm: bool,
                   only_calculate_mean: bool, delete_calculated_results: bool,
                   **slurm_parameters) -> None:
    from worldscore.benchmark.helpers.evaluator_per_round import MultiRoundEvaluator as Evaluator

    config["visual_movement"] = args.visual_movement

    evaluator = Evaluator(config)
    evaluator.evaluate(
        num_jobs=num_jobs,
        use_slurm=use_slurm,
        only_calculate_mean=only_calculate_mean,
        delete_calculated_results=delete_calculated_results,
        **slurm_parameters,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Default model_configs_dir relative to this script's location
    _script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Evaluate generation consistency on WorldScore per-round metrics."
    )
    parser.add_argument("--model_name", required=True,
                        help="Model identifier (e.g. pan, cosmos1)")
    parser.add_argument("--visual_movement", default="static",
                        help="static or dynamic (default: static)")
    parser.add_argument("--runs_root", required=True,
                        help="Root directory of prepared WorldScore outputs "
                             "(output of prepare_worldscore_dirs.py)")
    parser.add_argument("--model_configs_dir",
                        default=str(_script_dir / "worldscore_patches" / "model_configs"),
                        help="Directory containing per-model YAML configs")
    parser.add_argument("--num_jobs", type=int, default=24,
                        help="Number of parallel evaluation jobs (SLURM array size)")
    parser.add_argument("--use_slurm", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Submit evaluation jobs via SLURM (True/False)")
    parser.add_argument("--only_calculate_mean", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Skip evaluation, only aggregate existing results")
    parser.add_argument("--delete_calculated_results", type=lambda x: x.lower() == "true",
                        default=False,
                        help="Delete and re-evaluate already-scored instances")
    # SLURM passthrough parameters
    parser.add_argument("--slurm_partition", default=None)
    parser.add_argument("--slurm_job_name", default=None)
    parser.add_argument("--slurm_qos", default=None)
    args = parser.parse_args()

    config = build_config(args)

    slurm_parameters = {}
    if args.slurm_partition:
        slurm_parameters["slurm_partition"] = args.slurm_partition
    if args.slurm_job_name:
        slurm_parameters["slurm_job_name"] = args.slurm_job_name
    if args.slurm_qos:
        slurm_parameters["slurm_qos"] = args.slurm_qos

    visual_movement_list = [args.visual_movement]

    for visual_movement in visual_movement_list:
        ns = Namespace(model_name=args.model_name, visual_movement=visual_movement)

        print_banner("EVALUATION")
        run_evaluation(
            ns, config,
            num_jobs=args.num_jobs,
            use_slurm=args.use_slurm,
            only_calculate_mean=args.only_calculate_mean,
            delete_calculated_results=args.delete_calculated_results,
            **slurm_parameters,
        )

    calculate_worldscore(config, visual_movement_list)


if __name__ == "__main__":
    main()
