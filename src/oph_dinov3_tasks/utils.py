import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch


TRUE_SET = {"true", "1", "yes", "y", "t"}
FALSE_SET = {"false", "0", "no", "n", "f"}


def parse_bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError("empty boolean value")
    text = str(value).strip().lower()
    if text in TRUE_SET:
        return True
    if text in FALSE_SET:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def parse_bool_arg(value: Any) -> bool:
    return parse_bool_value(value)


def parse_int_list(raw: Any, default: Sequence[int]) -> List[int]:
    if raw is None or not str(raw).strip():
        return [int(x) for x in default]
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_float_list(raw: Any, default: Sequence[float]) -> List[float]:
    if raw is None or not str(raw).strip():
        return [float(x) for x in default]
    return [float(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_str_list(raw: Any, default: Sequence[str]) -> List[str]:
    if raw is None or not str(raw).strip():
        return [str(x) for x in default]
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def sanitize_name(value: str) -> str:
    text = str(value).strip().replace("/", "_")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "class"


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def torch_load(path: str, map_location: Any = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def amp_autocast(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def extract_state_dict(checkpoint: Any) -> Any:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state_dict", "model", "state_dict", "teacher", "student"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def strip_state_prefix(key: str) -> str:
    out = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "backbone.", "teacher.", "student."):
            if out.startswith(prefix):
                out = out[len(prefix):]
                changed = True
    return out


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            pass
    return str(value)


def write_rows_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    hidden = {"labels", "predictions", "probs", "logits", "sample_indices", "row_indices"}
    return {key: value for key, value in metrics.items() if key not in hidden}


def count_trainable_parameters(module: torch.nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total_params_m": float(total / 1e6),
        "trainable_params_m": float(trainable / 1e6),
    }


def first_existing(paths: Iterable[str]) -> str:
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return ""
