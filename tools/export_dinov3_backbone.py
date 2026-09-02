#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a continued DINOv3 teacher checkpoint as a raw backbone state_dict."
    )
    parser.add_argument("--input", required=True, help="Continued-pretraining teacher checkpoint.")
    parser.add_argument("--output", required=True, help="Output raw backbone checkpoint.")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--dinov3_hub_repo", default="")
    parser.add_argument("--dinov3_hub_model", default="dinov3_vitl16")
    return parser.parse_args()


def extract_backbone_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "teacher" in checkpoint:
        checkpoint = checkpoint["teacher"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a state_dict or contain a 'teacher' state_dict")

    backbone_state: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        key = str(key).removeprefix("module.").removeprefix("teacher.")
        if key.startswith("backbone."):
            backbone_state[key.removeprefix("backbone.")] = value
        elif not key.startswith(("dino_head.", "ibot_head.")):
            backbone_state[key] = value
    if not backbone_state:
        raise RuntimeError("no backbone tensors were found in the checkpoint")
    return backbone_state


def verify_checkpoint(repo_dir: str, model_name: str, state_dict: Dict[str, torch.Tensor]) -> None:
    if not repo_dir:
        raise ValueError("--dinov3_hub_repo is required with --verify")
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from dinov3.hub import backbones

    model = getattr(backbones, model_name)(pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    print(f"Verified: {model_name} loads exported checkpoint with strict=True")


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.input, map_location="cpu")
    backbone_state = extract_backbone_state_dict(checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(backbone_state, output)
    print(f"Saved raw backbone checkpoint: {output}")
    print(f"Backbone tensors: {len(backbone_state)}")
    if args.verify:
        verify_checkpoint(args.dinov3_hub_repo, args.dinov3_hub_model, backbone_state)


if __name__ == "__main__":
    main()
