import os
from pathlib import Path
from typing import Iterable, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_REPO_ID = os.getenv("HF_MODEL_REPO_ID", "molka8/worker_safety_models")


def ensure_model_file(filename: str, *, aliases: Optional[Iterable[str]] = None, repo_id: Optional[str] = None) -> str:
    models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    target_path = models_dir / filename
    if target_path.exists():
        return str(target_path)

    candidate_names = [filename, *(aliases or [])]

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        print(f"[HFModelStore] WARNING: huggingface_hub unavailable for {filename}: {exc}")
        return str(target_path)

    repo = repo_id or DEFAULT_REPO_ID
    for candidate in candidate_names:
        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=candidate,
                local_dir=str(models_dir),
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.name != filename:
                try:
                    downloaded_path.replace(target_path)
                except Exception:
                    pass
            if target_path.exists():
                print(f"[HFModelStore] OK: downloaded {filename} from {repo}")
                return str(target_path)
            if downloaded_path.exists():
                print(f"[HFModelStore] OK: downloaded {downloaded_path.name} from {repo}")
                return str(downloaded_path)
        except Exception:
            continue

    print(f"[HFModelStore] WARNING: could not download {filename} from {repo}")
    return str(target_path)
