import csv
import os
import random
import re
from io import BytesIO
from typing import Any, Dict, List, Sequence
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import clean_cell


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch.upper()) - ord("A") + 1)
    return value - 1


def _read_xlsx_simple(path: str) -> List[Dict[str, Any]]:
    with ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XLSX_NS):
                texts = [node.text or "" for node in item.findall(".//a:t", XLSX_NS)]
                shared.append("".join(texts))
        sheet_name = "xl/worksheets/sheet1.xml"
        sheet = ET.fromstring(zf.read(sheet_name))

    raw_rows: List[List[str]] = []
    for row in sheet.findall(".//a:sheetData/a:row", XLSX_NS):
        values: List[str] = []
        for cell in row.findall("a:c", XLSX_NS):
            idx = _xlsx_col_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("a:v", XLSX_NS)
            text = "" if value_node is None else value_node.text or ""
            if cell_type == "s" and text:
                text = shared[int(text)]
            elif cell_type == "inlineStr":
                inline = cell.find(".//a:t", XLSX_NS)
                text = "" if inline is None else inline.text or ""
            values[idx] = text
        raw_rows.append(values)

    if not raw_rows:
        return []
    headers = [str(item).strip() for item in raw_rows[0]]
    rows: List[Dict[str, Any]] = []
    for raw in raw_rows[1:]:
        if not any(str(value).strip() for value in raw):
            continue
        row = {header: (raw[i] if i < len(raw) else "") for i, header in enumerate(headers) if header}
        rows.append(row)
    return rows


def load_table(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"table not found: {path}")

    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        try:
            import pandas as pd
            df = pd.read_excel(path)
            return [
                {str(key): clean_cell(value) for key, value in row.items()}
                for row in df.to_dict(orient="records")
            ]
        except Exception:
            if lower.endswith(".xlsx"):
                return _read_xlsx_simple(path)
            raise

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return [{str(key): clean_cell(value) for key, value in row.items()} for row in reader]


def make_eval_transform(image_size: int = 224):
    crop_pct = 224 / 256 if int(image_size) <= 224 else 1.0
    resize_size = int(int(image_size) / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(int(image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class MultiSliceOCTDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_col: str,
        slice_indices: Sequence[int],
        image_root: str = "",
        image_ext: str = ".jpg",
        image_size: int = 224,
        is_train: bool = False,
        aug_strength: str = "default",
        slice_paths_col: str = "",
        allow_missing_images: bool = False,
    ):
        self.samples = list(samples)
        self.image_col = image_col
        self.slice_indices = [int(x) for x in slice_indices]
        self.image_root = image_root
        self.image_ext = image_ext
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.aug_strength = str(aug_strength).strip().lower()
        self.slice_paths_col = str(slice_paths_col or "").strip()
        self.allow_missing_images = bool(allow_missing_images)
        self.eval_transform = make_eval_transform(self.image_size)

        if self.aug_strength == "none":
            self.pad = 0
            self.flip_p = 0.0
            self.jitter_bcs = (0.0, 0.0, 0.0)
        elif self.aug_strength == "default":
            self.pad = 4
            self.flip_p = 0.5
            self.jitter_bcs = (0.2, 0.2, 0.1)
        elif self.aug_strength == "strong":
            self.pad = 8
            self.flip_p = 0.5
            self.jitter_bcs = (0.35, 0.35, 0.2)
        else:
            raise ValueError("aug_strength must be none, default, or strong")

        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def __len__(self) -> int:
        return len(self.samples)

    def _with_image_root(self, path: str) -> str:
        if self.image_root and path and not os.path.isabs(path):
            return os.path.join(self.image_root, path)
        return path

    def _resolve_slice_paths(self, row: Dict[str, Any]) -> List[str]:
        if self.slice_paths_col and str(row.get(self.slice_paths_col, "")).strip():
            raw_paths = [
                item.strip()
                for item in re.split(r"[|;,]", str(row[self.slice_paths_col]))
                if item.strip()
            ]
            if not raw_paths:
                raise ValueError(f"empty slice paths in column {self.slice_paths_col}")
            return [self._with_image_root(path) for path in raw_paths]

        folder = str(row.get(self.image_col, "")).strip()
        if not folder:
            raise ValueError(f"missing image folder in column {self.image_col}")
        folder = self._with_image_root(folder)
        return [os.path.join(folder, f"{idx}{self.image_ext}") for idx in self.slice_indices]

    def _load_image(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            if self.allow_missing_images:
                return Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
            raise RuntimeError(f"failed to load image: {path}") from exc

    def _apply_consistent_aug(self, pil_list: List[Image.Image]) -> List[torch.Tensor]:
        if self.aug_strength == "none":
            return [self.eval_transform(im) for im in pil_list]

        resized = [
            transforms.Resize(self.image_size, interpolation=Image.BICUBIC)(im)
            for im in pil_list
        ]
        padded = [TF.pad(im, padding=self.pad, fill=0) for im in resized]
        i, j, h, w = transforms.RandomCrop.get_params(
            padded[0], output_size=(self.image_size, self.image_size)
        )
        cropped = [TF.crop(im, i, j, h, w) for im in padded]

        if random.random() < self.flip_p:
            cropped = [TF.hflip(im) for im in cropped]

        b, c, s = self.jitter_bcs
        ops = [
            ("brightness", random.uniform(1 - b, 1 + b)),
            ("contrast", random.uniform(1 - c, 1 + c)),
            ("saturation", random.uniform(1 - s, 1 + s)),
        ]
        random.shuffle(ops)

        tensors = []
        for img in cropped:
            out = img
            for name, factor in ops:
                if name == "brightness":
                    out = TF.adjust_brightness(out, factor)
                elif name == "contrast":
                    out = TF.adjust_contrast(out, factor)
                elif name == "saturation":
                    out = TF.adjust_saturation(out, factor)
            tensor = TF.to_tensor(out)
            tensors.append(TF.normalize(tensor, mean=self.norm_mean, std=self.norm_std))
        return tensors

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.samples[idx]
        slice_paths = self._resolve_slice_paths(row)
        images = [self._load_image(path) for path in slice_paths]
        tensors = self._apply_consistent_aug(images) if self.is_train else [
            self.eval_transform(image) for image in images
        ]
        return {
            "pixel_values": torch.stack(tensors, dim=0),
            "label": torch.tensor(int(row["__label_id"]), dtype=torch.long),
            "sample_index": torch.tensor(idx, dtype=torch.long),
            "row_index": torch.tensor(int(row["__row_index"]), dtype=torch.long),
        }
