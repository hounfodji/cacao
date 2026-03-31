"""
Step 0: Download datasets.

Sources:
  KaraAgroAI — Harvard Dataverse (doi:10.7910/DVN/BBGQSP)
    Classes: Healthy, CSSVD, Anthracnose
    ~17,703 images, West Africa, CC-BY 4.0

  Sykes Ecuador — OSF (osf.io/2fw6g)
    Classes we use: BPR (Black Pod Rot), Healthy
    Classes we skip: FPR, WBD (Latin American diseases)
    7,220 images total

Output:
  data/raw/karaagro/   — KaraAgroAI images, original folder structure
  data/raw/sykes/      — Sykes Ecuador images, original folder structure

Usage:
  python 00_download_datasets.py --karaagro    # download KaraAgroAI only
  python 00_download_datasets.py --sykes       # download Sykes only
  python 00_download_datasets.py               # download both

Requirements:
  pip install requests tqdm
  For Kaggle datasets: pip install kaggle  (+ configure ~/.kaggle/kaggle.json)
"""

import argparse
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path("data/raw")

# ── Helpers ────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, desc: str = ""):
    """Stream-download url to dest with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        desc=desc or dest.name,
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def unzip(zip_path: Path, dest_dir: Path):
    print(f"Extracting {zip_path.name} → {dest_dir}/")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    zip_path.unlink()
    print("  Done.")


# ── KaraAgroAI — Harvard Dataverse ────────────────────────────────────────────

def download_karaagro():
    """
    Download KaraAgroAI dataset from Harvard Dataverse.

    The Harvard Dataverse API returns file metadata. We fetch the dataset
    and download the zip containing all images.

    If the download fails (large file, auth required), instructions are
    printed for manual download.
    """
    dest_dir = RAW_DIR / "karaagro"
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[SKIP] {dest_dir} already exists and is non-empty.")
        return

    print("=== Downloading KaraAgroAI (Harvard Dataverse) ===")

    # Harvard Dataverse API: list files in the dataset
    doi = "doi:10.7910/DVN/BBGQSP"
    api_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={doi}"

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] Could not reach Harvard Dataverse API: {e}")
        _print_karaagro_manual_instructions()
        return

    # Find downloadable files
    files = data.get("data", {}).get("latestVersion", {}).get("files", [])
    if not files:
        print("  [ERROR] No files found via API.")
        _print_karaagro_manual_instructions()
        return

    print(f"  Found {len(files)} file(s) in dataset:")
    for f in files:
        label = f.get("label", "?")
        size = f.get("dataFile", {}).get("filesize", 0)
        file_id = f.get("dataFile", {}).get("id")
        print(f"    {label}  ({size / 1e6:.1f} MB)  id={file_id}")

    # Download each file
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        label = f.get("label", "file")
        file_id = f.get("dataFile", {}).get("id")
        download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
        dest_file = dest_dir / label

        if dest_file.exists():
            print(f"  [SKIP] {label} already downloaded.")
            continue

        print(f"  Downloading {label}...")
        try:
            download_file(download_url, dest_file, desc=label)
            if dest_file.suffix.lower() == ".zip":
                unzip(dest_file, dest_dir)
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            _print_karaagro_manual_instructions()
            return

    print(f"\nKaraAgroAI downloaded to {dest_dir}/")
    print("Run 01_data_curation.py to verify class coverage.\n")


def _print_karaagro_manual_instructions():
    print("""
  Manual download instructions:
  1. Go to: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/BBGQSP
  2. Click "Access Dataset" → "Original Format ZIP" or download individual files
  3. Extract to: ml/data/raw/karaagro/
  4. Expected folder structure after extraction:
       data/raw/karaagro/
         Healthy/
         CSSVD/
         Anthracnose/
""")


# ── Sykes Ecuador — OSF ────────────────────────────────────────────────────────

def download_sykes():
    """
    Download Sykes Ecuador dataset from OSF (osf.io/2fw6g).

    Only downloads Healthy and BPR (Black Pod Rot) folders.
    Skips FPR (Frosty Pod Rot) and WBD (Witches' Broom) — Latin American diseases.
    """
    dest_dir = RAW_DIR / "sykes"
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[SKIP] {dest_dir} already exists and is non-empty.")
        return

    print("=== Downloading Sykes Ecuador (OSF) ===")

    # OSF API: list files in the project
    project_id = "2fw6g"
    api_url = f"https://api.osf.io/v2/nodes/{project_id}/files/osfstorage/"

    ALLOWED_FOLDERS = {"Healthy", "BPR"}  # exclude FPR, WBD

    try:
        resp = requests.get(api_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [ERROR] Could not reach OSF API: {e}")
        _print_sykes_manual_instructions()
        return

    items = data.get("data", [])
    if not items:
        print("  [ERROR] No files/folders found via OSF API.")
        _print_sykes_manual_instructions()
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        name = item["attributes"]["name"]
        kind = item["attributes"]["kind"]

        if kind == "folder":
            if name not in ALLOWED_FOLDERS:
                print(f"  [SKIP] {name}/ (excluded class)")
                continue
            print(f"  Downloading folder: {name}/")
            _download_osf_folder(item["relationships"]["files"]["links"]["related"]["href"],
                                  dest_dir / name)
        elif kind == "file":
            download_url = item["links"]["download"]
            dest_file = dest_dir / name
            print(f"  Downloading: {name}")
            try:
                download_file(download_url, dest_file, desc=name)
                if dest_file.suffix.lower() == ".zip":
                    unzip(dest_file, dest_dir)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")

    print(f"\nSykes Ecuador downloaded to {dest_dir}/")
    print("Run 01_data_curation.py to verify class coverage.\n")


def _download_osf_folder(folder_api_url: str, dest_dir: Path):
    """Recursively download an OSF folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(folder_api_url, timeout=30)
        resp.raise_for_status()
        items = resp.json().get("data", [])
    except Exception as e:
        print(f"    [ERROR] Could not list folder: {e}")
        return

    for item in items:
        name = item["attributes"]["name"]
        kind = item["attributes"]["kind"]
        if kind == "file":
            download_url = item["links"]["download"]
            dest_file = dest_dir / name
            if not dest_file.exists():
                try:
                    download_file(download_url, dest_file, desc=f"  {name}")
                except Exception as e:
                    print(f"    [ERROR] {name}: {e}")
        elif kind == "folder":
            sub_url = item["relationships"]["files"]["links"]["related"]["href"]
            _download_osf_folder(sub_url, dest_dir / name)

    # Handle pagination
    next_url = resp.json().get("links", {}).get("next")
    if next_url:
        _download_osf_folder(next_url, dest_dir)


def _print_sykes_manual_instructions():
    print("""
  Manual download instructions:
  1. Go to: https://osf.io/2fw6g
  2. Download only the "Healthy" and "BPR" folders (skip FPR, WBD)
  3. Extract to: ml/data/raw/sykes/
  4. Expected folder structure:
       data/raw/sykes/
         Healthy/
         BPR/
""")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download cacao disease datasets")
    parser.add_argument("--karaagro", action="store_true", help="Download KaraAgroAI only")
    parser.add_argument("--sykes", action="store_true", help="Download Sykes Ecuador only")
    args = parser.parse_args()

    if not args.karaagro and not args.sykes:
        # Default: download both
        download_karaagro()
        download_sykes()
    else:
        if args.karaagro:
            download_karaagro()
        if args.sykes:
            download_sykes()

    print("\nNext: run 01_data_curation.py")


if __name__ == "__main__":
    main()
