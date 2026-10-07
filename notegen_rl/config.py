"""Config loading and the work-dir layout shared by every stage."""
from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_config(path: str | None = None) -> dict:
    path = Path(path) if path else PROJECT_DIR / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve(p) -> Path:
    """Relative paths in config.yaml are relative to the project directory."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else PROJECT_DIR / p


def work_dir(cfg) -> Path:
    d = resolve(cfg["work_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def wpath(cfg, *parts, mkdir: bool = False) -> Path:
    """Path under work_dir. mkdir=True creates it as a directory."""
    p = work_dir(cfg).joinpath(*parts)
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


def upload_results(cfg, *parts) -> None:
    """Sync a work_dir subpath to `results_uri` (no-op when unset)."""
    uri = cfg.get("results_uri")
    if not uri:
        return
    src = wpath(cfg, *parts)
    if not src.exists():
        return
    dest = uri.rstrip("/") + "/" + "/".join(parts)
    cmd = ["aws", "s3", "sync" if src.is_dir() else "cp", str(src), dest, "--only-show-errors"]
    print("upload:", " ".join(cmd))
    subprocess.run(cmd, check=True)
