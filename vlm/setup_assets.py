from huggingface_hub import list_repo_files, hf_hub_download
import os

# ---- Config ----
REPO_ID = "Orthadanor/vlm-drone-assets"
REPO_TYPE = "dataset"
REMOTE_BASE_FOLDER = "usd"  # remote folder on HF
LOCAL_TARGET_ROOT = os.path.join("Genesis", "genesis", "assets")

# ---- Traverse and Download ----
all_files = list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

# Filter for files under the usd/ directory
usd_files = [f for f in all_files if f.startswith(f"{REMOTE_BASE_FOLDER}/")]

for filepath in usd_files:
    # Compute local directory path
    local_dir = os.path.join(LOCAL_TARGET_ROOT, os.path.dirname(filepath))
    os.makedirs(local_dir, exist_ok=True)

    # Download to HF cache and get the actual path
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filepath,
        repo_type=REPO_TYPE,
        cache_dir=local_dir  # optional; organizes cache better
    )

    print(f"Downloaded: {filepath} → {local_path}")
