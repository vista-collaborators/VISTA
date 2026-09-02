#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oph_dinov3_tasks.config import parse_configured_args
from oph_dinov3_tasks.tasks import class_names_for_task, load_task_samples, normalize_task
from oph_dinov3_tasks.trainer import DownstreamTrainer
from oph_dinov3_tasks.utils import (
    ensure_dir,
    parse_bool_arg,
    parse_float_list,
    parse_int_list,
    parse_str_list,
    sanitize_name,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune continued DINOv3 weights on OPH downstream tasks."
    )
    parser.add_argument("--task", default="injection", choices=["injection", "diagnosis", "biomarker"])
    parser.add_argument("--train_table", default="")
    parser.add_argument("--val_table", default="")
    parser.add_argument("--test_table", default="")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--run_name", default="")

    parser.add_argument("--image_col", default="OCT_pathway")
    parser.add_argument("--label_col", default="")
    parser.add_argument("--class_names", default="")
    parser.add_argument("--biomarker", default="")
    parser.add_argument("--filter_col", default="")
    parser.add_argument("--filter_value", default="")

    parser.add_argument("--image_root", default="")
    parser.add_argument("--image_ext", default=".jpg")
    parser.add_argument("--slice_indices", default="5,7,9,11,13")
    parser.add_argument("--slice_paths_col", default="")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--allow_missing_images", type=parse_bool_arg, default=False)

    parser.add_argument("--adaptation", default="finetune", choices=["lp", "finetune"])
    parser.add_argument("--dinov3_hub_repo", default="")
    parser.add_argument("--dinov3_hub_model", default="dinov3_vitl16")
    parser.add_argument("--dinov3_weights", default="")
    parser.add_argument("--dinov3_feature_mode", default="avg_patch", choices=["avg_patch", "cls"])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug_strength", default="default", choices=["none", "default", "strong"])
    parser.add_argument("--use_data_parallel", type=parse_bool_arg, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validate tables/config and stop before building DINOv3.")

    parser.add_argument("--fusion_layers", type=int, default=2)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden_dims", default="512,256")
    parser.add_argument("--head_dropouts", default="0.5,0.3")
    parser.add_argument("--selection_metric", default="macro_f1")
    return parser


def label_counts(samples: List[Dict[str, Any]], num_classes: int) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    for row in samples:
        counts[int(row["__label_id"])] += 1
    return counts


def make_experiment_dir(args: argparse.Namespace, task: str, biomarker: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.run_name.strip() or f"exp_{timestamp}"
    parts = [args.output_root, task]
    if biomarker:
        parts.append(sanitize_name(biomarker))
    parts.extend([args.adaptation, exp_name])
    return ensure_dir(os.path.join(*parts))


def main() -> None:
    args = parse_configured_args(build_parser)
    task = normalize_task(args.task)
    class_names = class_names_for_task(
        task,
        parse_str_list(args.class_names, []) if args.class_names else None,
    )
    label_col = args.label_col
    if task == "injection" and not label_col:
        label_col = "anti-VEGF injection"
    elif task == "diagnosis" and not label_col:
        label_col = "Diagnosis"

    missing_tables = [name for name in ("train_table", "val_table", "test_table") if not getattr(args, name)]
    if missing_tables:
        raise ValueError(f"missing required table arguments: {missing_tables}")
    if task == "biomarker" and not args.biomarker:
        raise ValueError("--biomarker is required for biomarker task")

    train_samples = load_task_samples(
        args.train_table,
        task=task,
        image_col=args.image_col,
        label_col=label_col,
        class_names=class_names,
        biomarker=args.biomarker,
        filter_col=args.filter_col,
        filter_value=args.filter_value,
    )
    val_samples = load_task_samples(
        args.val_table,
        task=task,
        image_col=args.image_col,
        label_col=label_col,
        class_names=class_names,
        biomarker=args.biomarker,
        filter_col=args.filter_col,
        filter_value=args.filter_value,
    )
    test_samples = load_task_samples(
        args.test_table,
        task=task,
        image_col=args.image_col,
        label_col=label_col,
        class_names=class_names,
        biomarker=args.biomarker,
        filter_col=args.filter_col,
        filter_value=args.filter_value,
    )

    slice_indices = parse_int_list(args.slice_indices, [5, 7, 9, 11, 13])
    config: Dict[str, Any] = {
        "task": task,
        "train_table": args.train_table,
        "val_table": args.val_table,
        "test_table": args.test_table,
        "output_root": args.output_root,
        "experiment_dir": make_experiment_dir(args, task, args.biomarker if task == "biomarker" else ""),
        "image_col": args.image_col,
        "label_col": label_col,
        "class_names": class_names,
        "num_classes": len(class_names),
        "biomarker": args.biomarker if task == "biomarker" else "",
        "filter_col": args.filter_col,
        "filter_value": args.filter_value,
        "image_root": args.image_root,
        "image_ext": args.image_ext,
        "slice_indices": slice_indices,
        "slice_paths_col": args.slice_paths_col,
        "input_size": int(args.input_size),
        "allow_missing_images": bool(args.allow_missing_images),
        "adaptation": args.adaptation,
        "dinov3_hub_repo": args.dinov3_hub_repo,
        "dinov3_hub_model": args.dinov3_hub_model,
        "dinov3_weights": args.dinov3_weights,
        "dinov3_feature_mode": args.dinov3_feature_mode,
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "aug_strength": args.aug_strength,
        "use_data_parallel": bool(args.use_data_parallel),
        "cpu": bool(args.cpu),
        "dry_run": bool(args.dry_run),
        "fusion_layers": int(args.fusion_layers),
        "fusion_dropout": float(args.fusion_dropout),
        "head_hidden_dims": parse_int_list(args.head_hidden_dims, [512, 256]),
        "head_dropouts": parse_float_list(args.head_dropouts, [0.5, 0.3]),
        "selection_metric": args.selection_metric,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "train_class_counts": label_counts(train_samples, len(class_names)),
        "val_class_counts": label_counts(val_samples, len(class_names)),
        "test_class_counts": label_counts(test_samples, len(class_names)),
    }

    write_json(os.path.join(config["experiment_dir"], "run_config.json"), config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        print("Dry run finished: tables and config are valid; model construction was skipped.")
        return

    trainer = DownstreamTrainer(config)
    summary = trainer.train(train_samples, val_samples, test_samples)
    write_json(os.path.join(config["experiment_dir"], "run_summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
