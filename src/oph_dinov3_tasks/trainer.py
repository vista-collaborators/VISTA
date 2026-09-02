import csv
import gc
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import MultiSliceOCTDataset
from .metrics import compute_classification_metrics
from .model import build_model_from_config
from .utils import (
    amp_autocast,
    compact_metrics,
    count_trainable_parameters,
    make_grad_scaler,
    sanitize_name,
    seed_everything,
    seed_worker,
    torch_load,
    write_json,
    write_rows_csv,
)


def _metric_for_selection(metrics: Dict[str, Any], metric_name: str) -> float:
    value = float(metrics.get(metric_name, float("nan")))
    return value if math.isfinite(value) else float("-inf")


class DownstreamTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)
        seed_everything(int(self.config.get("seed", 42)))
        self.device = torch.device(
            "cpu"
            if bool(self.config.get("cpu", False))
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.use_amp = self.device.type == "cuda"
        self.scaler = make_grad_scaler(self.use_amp)
        self.experiment_dir = str(self.config["experiment_dir"])
        self.dirs = {
            name: os.path.join(self.experiment_dir, name)
            for name in ("models", "logs", "reports")
        }
        for path in self.dirs.values():
            Path(path).mkdir(parents=True, exist_ok=True)

        model = build_model_from_config(self.config).to(self.device)
        if (
            bool(self.config.get("use_data_parallel", True))
            and self.device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model: nn.Module = nn.DataParallel(model)
        else:
            self.model = model

        params = count_trainable_parameters(self._unwrap_model())
        print(f"Device: {self.device}")
        print(
            "Parameters: "
            f"{params['trainable_params_m']:.2f}M trainable / "
            f"{params['total_params_m']:.2f}M total"
        )

    def _unwrap_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _build_loader(self, samples: List[Dict[str, Any]], is_train: bool) -> DataLoader:
        dataset = MultiSliceOCTDataset(
            samples=samples,
            image_col=str(self.config["image_col"]),
            slice_indices=self.config["slice_indices"],
            image_root=str(self.config.get("image_root", "")),
            image_ext=str(self.config.get("image_ext", ".jpg")),
            image_size=int(self.config.get("input_size", 224)),
            is_train=is_train,
            aug_strength=str(self.config.get("aug_strength", "default" if is_train else "none")),
            slice_paths_col=str(self.config.get("slice_paths_col", "")),
            allow_missing_images=bool(self.config.get("allow_missing_images", False)),
        )
        generator = torch.Generator()
        generator.manual_seed(int(self.config.get("seed", 42)) + (0 if is_train else 1))
        return DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 16)),
            shuffle=is_train,
            num_workers=int(self.config.get("num_workers", 4)),
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
            worker_init_fn=seed_worker if int(self.config.get("num_workers", 4)) > 0 else None,
            generator=generator,
        )

    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> Dict[str, Any]:
        self.model.eval()
        labels_all: List[int] = []
        preds_all: List[int] = []
        probs_all: List[List[float]] = []
        logits_all: List[List[float]] = []
        sample_indices: List[int] = []
        row_indices: List[int] = []
        losses: List[float] = []

        with torch.no_grad():
            for batch in loader:
                pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                with amp_autocast(enabled=self.use_amp):
                    logits = self.model(pixel_values)
                    loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                losses.append(float(loss.item()))
                labels_all.extend(labels.detach().cpu().tolist())
                preds_all.extend(preds.detach().cpu().tolist())
                probs_all.extend(probs.detach().cpu().tolist())
                logits_all.extend(logits.detach().cpu().tolist())
                sample_indices.extend([int(x) for x in batch["sample_index"].detach().cpu().tolist()])
                row_indices.extend([int(x) for x in batch["row_index"].detach().cpu().tolist()])

        metrics = compute_classification_metrics(
            labels_all,
            preds_all,
            probs_all,
            self.config["class_names"],
        )
        metrics.update(
            {
                "loss": float(sum(losses) / max(1, len(losses))),
                "labels": labels_all,
                "predictions": preds_all,
                "probs": probs_all,
                "logits": logits_all,
                "sample_indices": sample_indices,
                "row_indices": row_indices,
            }
        )
        return metrics

    def _train_one_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> Dict[str, Any]:
        self.model.train()
        labels_all: List[int] = []
        preds_all: List[int] = []
        probs_all: List[List[float]] = []
        losses: List[float] = []
        accumulation_steps = max(1, int(self.config.get("accumulation_steps", 1)))
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            with amp_autocast(enabled=self.use_amp):
                logits = self.model(pixel_values)
                loss = criterion(logits, labels) / accumulation_steps
            probs = torch.softmax(logits.detach(), dim=1)
            preds = torch.argmax(probs, dim=1)
            self.scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.item() * accumulation_steps))
            labels_all.extend(labels.detach().cpu().tolist())
            preds_all.extend(preds.detach().cpu().tolist())
            probs_all.extend(probs.detach().cpu().tolist())

        if len(loader) > 0 and len(loader) % accumulation_steps != 0:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics = compute_classification_metrics(
            labels_all,
            preds_all,
            probs_all,
            self.config["class_names"],
        )
        metrics["loss"] = float(sum(losses) / max(1, len(losses)))
        return metrics

    def _save_history(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        path = os.path.join(self.dirs["logs"], "training_history.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def _save_checkpoint(self, epoch: int, best_score: float) -> None:
        path = os.path.join(self.dirs["models"], "best_model.pth")
        tmp_path = f"{path}.tmp"
        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in self._unwrap_model().state_dict().items()
                },
                "epoch": int(epoch),
                "best_score": float(best_score),
                "selection_metric": self.config.get("selection_metric", "macro_f1"),
                "config": self.config,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch_load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self._unwrap_model().load_state_dict(state_dict, strict=True)

    def save_metrics(self, metrics: Dict[str, Any], filename: str) -> None:
        write_json(os.path.join(self.dirs["reports"], filename), compact_metrics(metrics))

    def save_predictions(
        self,
        samples: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        filename: str,
    ) -> None:
        rows: List[Dict[str, Any]] = []
        order = sorted(
            range(len(metrics["sample_indices"])),
            key=lambda i: int(metrics["sample_indices"][i]),
        )
        class_names = list(self.config["class_names"])
        for i in order:
            sample = dict(samples[int(metrics["sample_indices"][i])])
            for key in ("__row_index", "__label_id", "__label_name"):
                sample.pop(key, None)
            true_id = int(metrics["labels"][i])
            pred_id = int(metrics["predictions"][i])
            prob_vec = metrics["probs"][i]
            logit_vec = metrics["logits"][i] if "logits" in metrics else []
            sample["true_label_id"] = true_id
            sample["true_label_name"] = class_names[true_id]
            sample["pred_label_id"] = pred_id
            sample["pred_label_name"] = class_names[pred_id]
            for class_id, class_name in enumerate(class_names):
                suffix = sanitize_name(class_name).lower()
                sample[f"prob_{suffix}"] = float(prob_vec[class_id])
                if logit_vec:
                    sample[f"logit_{suffix}"] = float(logit_vec[class_id])
            sample["correct"] = int(true_id == pred_id)
            rows.append(sample)
        write_rows_csv(os.path.join(self.dirs["reports"], filename), rows)

    def train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        train_loader = self._build_loader(train_samples, is_train=True)
        val_loader = self._build_loader(val_samples, is_train=False)
        test_loader = self._build_loader(test_samples, is_train=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config.get("lr", 3e-6)),
            weight_decay=float(self.config.get("weight_decay", 1e-4)),
        )
        epochs = int(self.config.get("epochs", 50))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=max(float(self.config.get("lr", 3e-6)) * 0.1, 1e-8),
        )
        selection_metric = str(self.config.get("selection_metric", "macro_f1"))
        best_score = float("-inf")
        best_epoch = -1
        patience_counter = 0
        patience = int(self.config.get("patience", 15))
        history: List[Dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(train_loader, criterion, optimizer)
            val_metrics = self._evaluate_loader(val_loader, criterion)
            scheduler.step()

            score = _metric_for_selection(val_metrics, selection_metric)
            row = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "train_acc": train_metrics.get("accuracy"),
                "train_macro_f1": train_metrics.get("macro_f1"),
                "val_loss": val_metrics.get("loss"),
                "val_acc": val_metrics.get("accuracy"),
                "val_macro_f1": val_metrics.get("macro_f1"),
                f"val_{selection_metric}": val_metrics.get(selection_metric),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            if "auc" in train_metrics:
                row["train_auc"] = train_metrics["auc"]
                row["val_auc"] = val_metrics.get("auc")
            history.append(row)
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={float(row['train_loss']):.4f} "
                f"train_acc={float(row['train_acc']):.4f} | "
                f"val_loss={float(row['val_loss']):.4f} "
                f"val_acc={float(row['val_acc']):.4f} "
                f"val_macro_f1={float(row['val_macro_f1']):.4f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience_counter = 0
                self._save_checkpoint(epoch, best_score)
                self.save_metrics(val_metrics, "best_val_metrics.json")
                print(f"  -> saved best model, {selection_metric}={best_score:.6f}")
            else:
                patience_counter += 1
                print(f"  -> no improvement, patience={patience_counter}/{patience}")
            if patience_counter >= patience:
                print("  -> early stopping")
                break

        self._save_history(history)
        best_model_path = os.path.join(self.dirs["models"], "best_model.pth")
        if os.path.isfile(best_model_path):
            self.load_checkpoint(best_model_path)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        final_val = self._evaluate_loader(val_loader, criterion)
        final_test = self._evaluate_loader(test_loader, criterion)
        self.save_metrics(final_val, "val_metrics_final.json")
        self.save_metrics(final_test, "test_metrics_final.json")
        self.save_predictions(val_samples, final_val, "val_predictions.csv")
        self.save_predictions(test_samples, final_test, "test_predictions.csv")

        summary = {
            "status": "success",
            "task": self.config.get("task"),
            "biomarker": self.config.get("biomarker", ""),
            "experiment_dir": self.experiment_dir,
            "best_epoch": int(best_epoch),
            "selection_metric": selection_metric,
            "best_val_score": float(best_score),
            "val_accuracy_final": float(final_val["accuracy"]),
            "val_macro_f1_final": float(final_val["macro_f1"]),
            "test_accuracy_final": float(final_test["accuracy"]),
            "test_macro_f1_final": float(final_test["macro_f1"]),
        }
        if "auc" in final_test:
            summary["val_auc_final"] = float(final_val["auc"])
            summary["test_auc_final"] = float(final_test["auc"])
        return summary


def evaluate_checkpoint(
    checkpoint_path: str,
    samples: List[Dict[str, Any]],
    config_overrides: Dict[str, Any],
    output_dir: str,
    predictions_filename: str = "predictions.csv",
) -> Dict[str, Any]:
    checkpoint = torch_load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("checkpoint must contain a config dict saved by run_finetuning.py")
    config = dict(checkpoint["config"])
    config.update({k: v for k, v in config_overrides.items() if v not in ("", None)})
    config["experiment_dir"] = output_dir
    config["epochs"] = 1
    trainer = DownstreamTrainer(config)
    trainer.load_checkpoint(checkpoint_path)
    loader = trainer._build_loader(samples, is_train=False)
    metrics = trainer._evaluate_loader(loader, nn.CrossEntropyLoss())
    trainer.save_metrics(metrics, "metrics.json")
    trainer.save_predictions(samples, metrics, predictions_filename)
    return compact_metrics(metrics)
