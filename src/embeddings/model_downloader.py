"""Model download utility using ModelScope.

Downloads BGE-M3 (BAAI General Embedding) locally so embeddings
run entirely offline without external API calls.
"""

from pathlib import Path

from modelscope import snapshot_download

# ModelScope model ID for BGE-M3
BGE_M3_MODEL_ID = "BAAI/bge-m3"


def download_bge_m3(target_dir: Path | None = None) -> Path:
    """Download BGE-M3 from ModelScope to the project models/ directory.

    BGE-M3 is a multilingual embedding model supporting 100+ languages.
    It produces 1024-dimensional dense embeddings and works well for
    Chinese and English document retrieval.

    Args:
        target_dir: Directory to store the model. Defaults to models/ in
                    the current working directory.

    Returns:
        Path to the downloaded model directory.
    """
    if target_dir is None:
        target_dir = Path("models")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Download to a specific subdirectory
    local_model_dir = target_dir / "bge-m3"

    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        print(f"Model already exists at: {local_model_dir}")
        return local_model_dir

    print(f"Downloading {BGE_M3_MODEL_ID} from ModelScope...")
    snapshot_download(
        BGE_M3_MODEL_ID,
        local_dir=str(local_model_dir.resolve()),
    )
    print(f"Model downloaded to: {local_model_dir.resolve()}")

    return local_model_dir


def _is_valid_model_dir(path: Path) -> bool:
    """Check if a directory contains a valid HuggingFace model (config.json + weights)."""
    if not path.exists() or not (path / "config.json").exists():
        return False
    # Must have at least one pytorch weight file
    # (onnx-only models are not supported by HuggingFaceEmbeddings)
    return any(
        path.glob("pytorch_model.bin")
    ) or any(
        path.glob("model.safetensors")
    )


def get_model_path(model_dir: Path | None = None) -> Path:
    """Get the path to the BGE-M3 model, downloading if needed.

    Searches multiple common locations for the model.

    Args:
        model_dir: Directory containing (or to contain) the model.

    Returns:
        Path to the model directory, ready to load with HuggingFaceEmbeddings.
    """
    if model_dir is None:
        model_dir = Path("models")

    # Search paths in priority order:
    # 1. models/bge-m3 (flat structure from snapshot_download with local_dir)
    # 2. models/BAAI/bge-m3 (ModelScope default cache layout)
    candidates = [
        model_dir / "bge-m3",
        model_dir / "BAAI" / "bge-m3",
    ]

    for candidate in candidates:
        if _is_valid_model_dir(candidate):
            return candidate

    # Download if not found in any candidate location
    return download_bge_m3(model_dir)
