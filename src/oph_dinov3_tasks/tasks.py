from typing import Any, Dict, List, Optional, Sequence

from .data import load_table
from .utils import parse_bool_value, sanitize_name


TASK_INJECTION = "injection"
TASK_DIAGNOSIS = "diagnosis"
TASK_BIOMARKER = "biomarker"

DEFAULT_INJECTION_CLASS_NAMES = ["no_injection", "need_injection"]
DEFAULT_DIAGNOSIS_CLASS_NAMES = ["nAMD", "DME", "RVO"]
DEFAULT_BIOMARKERS = ["IRF", "SRF", "SHRM", "DRT", "DRT/ME", "HRF", "NV", "PED"]


def normalize_task(task: str) -> str:
    value = str(task).strip().lower()
    if value in {"injection", "need_injection", "anti_vegf", "anti-vegf"}:
        return TASK_INJECTION
    if value in {"diagnosis", "diag", "disease", "disease_diagnosis"}:
        return TASK_DIAGNOSIS
    if value in {"biomarker", "marker"}:
        return TASK_BIOMARKER
    raise ValueError(f"unsupported task: {task!r}")


def normalize_biomarker_name(value: str) -> str:
    text = str(value).strip().upper().replace(" ", "")
    if text in {"DRT_ME", "DRT/ME", "ME", "DRT"}:
        return "DRT/ME"
    return text


def parse_injection_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text.endswith(".0"):
        text = text[:-2]
    if text == "2":
        return 0
    return int(parse_bool_value(text))


def build_label_to_id(class_names: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, name in enumerate(class_names):
        mapping[str(name).strip().lower()] = int(idx)
        mapping[sanitize_name(str(name)).lower()] = int(idx)
    if "AMD" in class_names:
        mapping.setdefault("namd", int(class_names.index("AMD")))
        mapping.setdefault("n-amd", int(class_names.index("AMD")))
    if "nAMD" in class_names:
        mapping.setdefault("amd", int(class_names.index("nAMD")))
        mapping.setdefault("n-amd", int(class_names.index("nAMD")))
    return mapping


def parse_diagnosis_label(value: Any, class_names: Sequence[str]) -> int:
    key = str(value).strip().lower()
    key = key.replace(" ", "")
    label_to_id = build_label_to_id(class_names)
    if key in label_to_id:
        return int(label_to_id[key])
    raise ValueError(f"unknown diagnosis label {value!r}; expected {list(class_names)}")


def resolve_biomarker_column(row: Dict[str, Any], biomarker: str) -> Optional[str]:
    wanted = normalize_biomarker_name(biomarker)
    candidates = [biomarker, sanitize_name(biomarker), wanted, sanitize_name(wanted)]
    lowered = {str(key).strip().lower(): key for key in row.keys()}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lowered:
            return str(lowered[key])
    if wanted == "DRT/ME":
        for candidate in ("DRT", "ME", "DRT_ME"):
            key = candidate.lower()
            if key in lowered:
                return str(lowered[key])
    return None


def load_task_samples(
    table_path: str,
    task: str,
    image_col: str,
    label_col: str = "",
    class_names: Optional[Sequence[str]] = None,
    biomarker: str = "",
    filter_col: str = "",
    filter_value: str = "",
) -> List[Dict[str, Any]]:
    task = normalize_task(task)
    rows = load_table(table_path)
    if not rows:
        raise ValueError(f"no rows loaded from {table_path}")

    samples: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if image_col not in row or not str(row.get(image_col, "")).strip():
            continue
        if filter_col:
            if filter_col not in row:
                raise ValueError(f"filter column {filter_col!r} missing in {table_path}")
            if str(filter_value).strip():
                try:
                    keep = parse_injection_label(row.get(filter_col)) == parse_injection_label(filter_value)
                except ValueError:
                    keep = str(row.get(filter_col, "")).strip().lower() == str(filter_value).strip().lower()
            else:
                keep = parse_injection_label(row.get(filter_col)) == 1
            if not keep:
                continue

        try:
            if task == TASK_INJECTION:
                if not label_col:
                    raise ValueError("label_col is required for injection task")
                label_id = parse_injection_label(row.get(label_col))
                label_name = DEFAULT_INJECTION_CLASS_NAMES[label_id]
            elif task == TASK_DIAGNOSIS:
                if not label_col:
                    raise ValueError("label_col is required for diagnosis task")
                names = list(class_names or DEFAULT_DIAGNOSIS_CLASS_NAMES)
                label_id = parse_diagnosis_label(row.get(label_col), names)
                label_name = names[label_id]
            else:
                if not biomarker:
                    raise ValueError("biomarker is required for biomarker task")
                column = resolve_biomarker_column(row, biomarker)
                if column is None:
                    raise ValueError(f"biomarker column for {biomarker!r} not found")
                label_id = int(parse_injection_label(row.get(column)))
                label_name = "present" if label_id == 1 else "absent"
        except Exception as exc:
            raise ValueError(f"{table_path} row {row_index + 2}: {exc}") from exc

        sample = dict(row)
        sample["__row_index"] = int(row_index)
        sample["__label_id"] = int(label_id)
        sample["__label_name"] = label_name
        samples.append(sample)

    if not samples:
        raise ValueError(f"no valid samples loaded from {table_path}")
    return samples


def class_names_for_task(
    task: str,
    class_names: Optional[Sequence[str]] = None,
) -> List[str]:
    task = normalize_task(task)
    if task == TASK_DIAGNOSIS:
        return list(class_names or DEFAULT_DIAGNOSIS_CLASS_NAMES)
    if task == TASK_BIOMARKER:
        return ["absent", "present"]
    return list(DEFAULT_INJECTION_CLASS_NAMES)
