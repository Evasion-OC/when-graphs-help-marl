"""CLI entrypoint: ``python -m gnnmarl ...``.

The CLI is a thin tyro-driven shell around :func:`gnnmarl.training.loop.train`.
For sweeps we drive ``train()`` from Python directly — see ``scripts/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import tyro
import yaml

from gnnmarl.training import TrainConfig, train


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path} must be a mapping; got {type(data).__name__}")
    # Path-typed fields need re-wrapping.
    if "log_dir" in data:
        data["log_dir"] = Path(data["log_dir"])
    return data


def main(argv: list[str] | None = None) -> Path:
    """Parse args and run training.

    Supports an optional ``--config path/to.yaml`` that pre-fills the defaults
    for any TrainConfig field; any later flag overrides the YAML value.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    yaml_overrides: dict = {}
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 >= len(argv):
            raise SystemExit("--config requires a path argument")
        yaml_path = Path(argv[i + 1])
        yaml_overrides = _load_yaml(yaml_path)
        # Strip the --config flag so tyro doesn't see it.
        argv = argv[:i] + argv[i + 2 :]

    default = TrainConfig(**yaml_overrides) if yaml_overrides else TrainConfig()
    cfg = tyro.cli(TrainConfig, args=argv, default=default)
    run_dir = train(cfg)
    print(f"run complete: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
