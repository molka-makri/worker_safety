from pathlib import Path
import os

try:
    from huggingface_hub import HfApi
except ModuleNotFoundError as exc:
    raise SystemExit(
        "huggingface_hub is not installed in the active Python environment. "
        "Activate the correct venv, then run: python -m pip install -U huggingface_hub"
    ) from exc


ROOT = Path(__file__).resolve().parent

ALLOW_PATTERNS = [
    "app/**",
    "config/**",
    "templates/**",
    "static/**",
    "models/.gitkeep",
    "media/.gitkeep",
    "manage.py",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    ".dockerignore",
    "start.sh",
]

IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "db.sqlite3",
    "venv/**",
    ".venv/**",
    "diff_*.txt",
    "proximity_func_backup.txt",
]


def main() -> None:
    repo_id = os.getenv("HF_SPACE_ID")
    if not repo_id:
        raise SystemExit("Set HF_SPACE_ID like username/worker-safety-dashboard")

    include_local_assets = os.getenv("HF_INCLUDE_LOCAL_ASSETS", "").strip().lower() in {"1", "true", "yes", "on"}
    allow_patterns = list(ALLOW_PATTERNS)
    if include_local_assets:
        allow_patterns.extend(["models/**", "media/**"])

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        allow_patterns=allow_patterns,
        ignore_patterns=IGNORE_PATTERNS,
    )
    print(f"Uploaded to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
