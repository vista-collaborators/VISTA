import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vista.config import parse_configured_args
from vista.tasks import class_names_for_task, load_task_samples, normalize_task
from vista.trainer import evaluate_checkpoint
from vista.utils import (
    parse_bool_arg,
    parse_int_list,
    parse_str_list,
    torch_load,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved VISTA downstream checkpoint."
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--table", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--predictions_filename", default="predictions.csv")

    parser.add_argument("--task", default="")
    parser.add_argument("--image_col", default="")
    parser.add_argument("--label_col", default="")
    parser.add_argument("--class_names", default="")
    parser.add_argument("--biomarker", default="")
    parser.add_argument("--filter_col", default="")
    parser.add_argument("--filter_value", default="")

    parser.add_argument("--image_root", default="")
    parser.add_argument("--image_ext", default="")
    parser.add_argument("--slice_indices", default="")
    parser.add_argument("--slice_paths_col", default="")
    parser.add_argument("--input_size", type=int, default=0)
    parser.add_argument("--allow_missing_images", type=parse_bool_arg, default=False)

    parser.add_argument("--dinov3_model", default="")
    parser.add_argument("--dinov3_weights", default="")
    parser.add_argument("--dinov3_feature_mode", default="")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--use_data_parallel", type=parse_bool_arg, default=False)
    parser.add_argument("--cpu", action="store_true")
    return parser


def saved_config_from_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch_load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("checkpoint must contain a config dict saved by run_finetuning.py")
    return dict(checkpoint["config"])


def default_output_dir_for_checkpoint(checkpoint_path: str, timestamp: str) -> str:
    checkpoint = Path(checkpoint_path)
    if checkpoint.parent.name == "models":
        experiment_dir = checkpoint.parent.parent
    else:
        experiment_dir = checkpoint.parent
    return os.path.join(str(experiment_dir), "reports", f"manual_test_{timestamp}")


def main() -> None:
    args = parse_configured_args(build_parser)
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    if not args.table:
        raise ValueError("--table is required")

    saved_config = saved_config_from_checkpoint(args.checkpoint)
    task = normalize_task(args.task or saved_config.get("task", ""))
    class_names = class_names_for_task(
        task,
        parse_str_list(args.class_names, saved_config.get("class_names", []))
        if args.class_names
        else saved_config.get("class_names", []),
    )
    image_col = args.image_col or saved_config.get("image_col", "OCT_pathway")
    label_col = args.label_col or saved_config.get("label_col", "")
    biomarker = args.biomarker or saved_config.get("biomarker", "")
    filter_col = args.filter_col if args.filter_col else saved_config.get("filter_col", "")
    filter_value = args.filter_value if args.filter_value else saved_config.get("filter_value", "")

    samples = load_task_samples(
        args.table,
        task=task,
        image_col=image_col,
        label_col=label_col,
        class_names=class_names,
        biomarker=biomarker,
        filter_col=filter_col,
        filter_value=filter_value,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or default_output_dir_for_checkpoint(args.checkpoint, timestamp)
    overrides: Dict[str, Any] = {
        "task": task,
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_col": image_col,
        "label_col": label_col,
        "biomarker": biomarker,
        "filter_col": filter_col,
        "filter_value": filter_value,
        "image_root": args.image_root if args.image_root else saved_config.get("image_root", ""),
        "image_ext": args.image_ext if args.image_ext else saved_config.get("image_ext", ".jpg"),
        "slice_indices": parse_int_list(
            args.slice_indices,
            saved_config.get("slice_indices", [5, 7, 9, 11, 13]),
        ),
        "slice_paths_col": args.slice_paths_col if args.slice_paths_col else saved_config.get("slice_paths_col", ""),
        "input_size": int(args.input_size or saved_config.get("input_size", 224)),
        "allow_missing_images": bool(args.allow_missing_images),
        "dinov3_model": args.dinov3_model if args.dinov3_model else saved_config.get("dinov3_model", "dinov3_vitl16"),
        "dinov3_weights": args.dinov3_weights if args.dinov3_weights else saved_config.get("dinov3_weights", ""),
        "dinov3_feature_mode": args.dinov3_feature_mode if args.dinov3_feature_mode else saved_config.get("dinov3_feature_mode", "avg_patch"),
        "batch_size": int(args.batch_size or saved_config.get("batch_size", 16)),
        "num_workers": int(args.num_workers if args.num_workers >= 0 else saved_config.get("num_workers", 4)),
        "use_data_parallel": bool(args.use_data_parallel),
        "cpu": bool(args.cpu),
        "aug_strength": "none",
    }
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        samples=samples,
        config_overrides=overrides,
        output_dir=output_dir,
        predictions_filename=args.predictions_filename,
    )
    summary = {
        "checkpoint": args.checkpoint,
        "table": args.table,
        "output_dir": output_dir,
        "n_samples": len(samples),
        "metrics": metrics,
    }
    write_json(os.path.join(output_dir, "test_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
