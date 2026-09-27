from pathlib import Path
import re

from ..errors import CompatibilityError


def require_cuda(device):
    import torch
    parsed = torch.device(device)
    if parsed.type != "cuda" or not torch.cuda.is_available():
        raise CompatibilityError("CUDA is required; CPU fallback/offload is not supported")
    index = 0 if parsed.index is None else parsed.index
    if index >= torch.cuda.device_count():
        raise CompatibilityError(f"CUDA device {index} does not exist")
    return f"cuda:{index}"


def resolve_checkpoint(model, revision=None, cache_dir=None):
    """Resolve a local path or download an immutable safetensors snapshot."""
    from ..storage import hf_cache_dir
    path = Path(model)
    if path.is_dir():
        resolved = path.resolve()
        snapshot_revision = resolved.name if resolved.parent.name == "snapshots" and re.fullmatch(r"[a-f0-9]{40}", resolved.name) else "local"
        return str(resolved), revision or snapshot_revision
    from huggingface_hub import HfApi, snapshot_download
    cache = str(cache_dir or hf_cache_dir())
    info = HfApi().model_info(str(model), revision=revision, files_metadata=True)
    selected = [s for s in info.siblings if s.rfilename.endswith((".json", ".jinja", ".model", ".txt", ".safetensors"))]
    if not any(s.rfilename.endswith(".safetensors") for s in selected):
        raise CompatibilityError("No safetensors checkpoint found; supply a supported local checkpoint")
    local = snapshot_download(str(model), revision=info.sha, cache_dir=cache,
                              allow_patterns=[f.rfilename for f in selected])
    return local, info.sha


class TorchMemory:
    def peak_memory_bytes(self):
        import torch
        return torch.cuda.max_memory_allocated(self.device), torch.cuda.max_memory_reserved(self.device)

    def reset_peak_memory(self):
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)
