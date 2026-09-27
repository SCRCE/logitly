from __future__ import annotations

import os
from pathlib import Path

def project_root() -> Path:
    return Path(os.environ.get("LOGITLY_PROJECT_ROOT", Path.cwd())).resolve()


def hf_cache_dir(root: Path | None = None) -> Path:
    root = root or project_root()
    configured = os.environ.get("HF_HUB_CACHE")
    return Path(configured).resolve() if configured else root / ".cache" / "huggingface" / "hub"
