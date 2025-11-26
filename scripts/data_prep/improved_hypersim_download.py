#!/usr/bin/env python3

# cd /home/thoalst/dev/ml-hypersim/contrib/99991
# python3 -m venv hypersim-env
# source hypersim-env/bin/activate
# pip install -r requirements.txt
# python improved_hypersim_download.py

import os
import sys
import time
import zipfile
import requests
from tqdm import tqdm

# ==========================
# USER CONFIGURATION
# ==========================

SCENES = [
    "ai_001_001",
    "ai_001_002",
    "ai_001_003",
    "ai_002_001",
    "ai_003_001"
]

TARGET_DIR = "/mnt/vrdata/depth_ground_truth/hypersim"

# Files we want to extract
KEEP_KEYWORDS = [
    "color.hdf5",
    "depth_meters.hdf5",
    "camera_keyframe_positions",
    "camera_keyframe_orientations",
    "metadata_cameras.csv",
    "metadata_scene.csv"
]

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes"

CHUNK_SIZE = 1024 * 1024   # 1MB chunks
MAX_RETRIES = 5


# ==========================
# HELPER FUNCTIONS
# ==========================

def download_with_resume(url, out_path):
    """Download a file with resume, progress bar, and retry logic."""
    temp_path = out_path + ".part"

    # Check existing partial file
    resume_byte_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
    headers = {}
    if resume_byte_pos > 0:
        headers["Range"] = f"bytes={resume_byte_pos}-"

    # Start request
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos

            mode = "ab" if resume_byte_pos > 0 else "wb"

            with open(temp_path, mode) as f, tqdm(
                total=total_size,
                initial=resume_byte_pos,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(out_path)
            ) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            # Move into place
            os.rename(temp_path, out_path)
            return True

        except Exception as e:
            print(f"!! ERROR downloading {url} (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(2)

    print(f"!! FAILED to download {url}")
    return False


def extract_selected(zip_path, scene_name):
    """Extract only files that contain needed keywords."""
    print(f"Extracting needed files from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for entry in z.infolist():
            # Skip directories
            if entry.is_dir():
                continue

            # Check if file should be kept
            if not any(key in entry.filename for key in KEEP_KEYWORDS):
                continue

            # Build output path
            out_path = os.path.join(TARGET_DIR, entry.filename)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # Extract
            print("Extract:", entry.filename)
            with z.open(entry) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

    print(f"Finished extracting {scene_name}")


# ==========================
# MAIN
# ==========================

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)

    for scene in SCENES:
        print("\n======================================")
        print(f"Downloading scene ZIP: {scene}")
        print("======================================")

        url = f"{BASE_URL}/{scene}.zip"
        zip_path = os.path.join(TARGET_DIR, f"{scene}.zip")

        # Download ZIP
        success = download_with_resume(url, zip_path)
        if not success:
            print(f"Skipping {scene} because download failed.")
            continue

        # Extract needed files
        extract_selected(zip_path, scene)

        # Delete ZIP to save space
        print(f"Deleting ZIP: {zip_path}")
        os.remove(zip_path)

    print("\n======================================")
    print(" DONE — Hypersim selective download complete.")
    print(f"Saved to {TARGET_DIR}")
    print("======================================\n")


if __name__ == "__main__":
    main()
