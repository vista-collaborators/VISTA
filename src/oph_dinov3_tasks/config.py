import argparse
from typing import Any, Callable, Dict, Optional, Sequence


def read_yaml(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required when using --config") from exc
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def parse_configured_args(
    build_parser: Callable[[], argparse.ArgumentParser],
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-c", "--config", default="", type=str)
    config_args, remaining = config_parser.parse_known_args(argv)

    parser = build_parser()
    defaults = read_yaml(config_args.config)
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    return args
