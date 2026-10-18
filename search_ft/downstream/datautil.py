"""Data-directory helpers shared by the downstream benchmarks.

Every benchmark keeps its raw data under ``$DOWNSTREAM_DATA_DIR/<benchmark>/`` and
fetches it on first use (zip download, shallow git clone, or a deterministic
generator). The HuggingFace datasets used by several benchmarks go through the
usual ``datasets`` cache (``HF_HOME`` / ``HF_DATASETS_CACHE``).
"""
import glob
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from typing import Optional


def data_dir() -> Optional[str]:
    return os.environ.get("DOWNSTREAM_DATA_DIR") or None


def require_data_dir() -> str:
    d = data_dir()
    if d is None:
        raise RuntimeError(
            "DOWNSTREAM_DATA_DIR is not set. Pass --data-dir to the runner or export "
            "DOWNSTREAM_DATA_DIR=<directory for benchmark data>."
        )
    os.makedirs(d, exist_ok=True)
    return d


def download_zip(url: str, dest_dir: str, tag: str, size_hint: str = "",
                 member_filter=None) -> bool:
    """Download ``url`` into ``dest_dir`` and extract it there. With ``member_filter``
    (a predicate on the archive path) only the matching members are extracted.
    Returns success."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, os.path.basename(url))
    try:
        print(f"  [{tag}] downloading {url} {size_hint}".rstrip())
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            members = z.namelist()
            if member_filter is not None:
                members = [m for m in members if member_filter(m)]
            z.extractall(dest_dir, members=members)
            print(f"  [{tag}] extracted {len(members)} file(s) to {dest_dir}")
        os.remove(zip_path)
        return True
    except Exception as e:  # noqa: BLE001 - report and let the caller decide
        print(f"  [{tag}] download failed: {e}")
        return False


def _sparse_clone(url: str, dest: str, patterns) -> bool:
    """Shallow partial clone that checks out only ``patterns`` (git >= 2.25 and a server
    that supports blob filters, which GitHub does). Returns False if any step fails, in
    which case the caller falls back to a plain shallow clone."""
    try:
        subprocess.check_call(["git", "clone", "-q", "--depth=1", "--filter=blob:none",
                               "--no-checkout", url, dest], timeout=600)
        subprocess.check_call(["git", "sparse-checkout", "set", "--no-cone", *patterns],
                              cwd=dest, timeout=60)
        subprocess.check_call(["git", "checkout", "-q"], cwd=dest, timeout=600)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  sparse clone of {url} failed ({e}); falling back to a full shallow clone")
        shutil.rmtree(dest, ignore_errors=True)
        return False


def clone_repo(url: str, dest: str, marker_glob: str, tag: str,
               commit: Optional[str] = None, sparse=None) -> Optional[str]:
    """Shallow-clone ``url`` to ``dest`` unless ``marker_glob`` (relative to ``dest``)
    already matches files. Checks for FILES, not just the directory: scratch purges
    delete file contents but leave the directory tree. With ``commit`` the exact
    revision is fetched (GitHub allows fetching a reachable SHA) and checked out. With
    ``sparse`` (a list of git sparse-checkout patterns) only those paths are fetched.
    Returns ``dest`` or None on failure."""
    if glob.glob(os.path.join(dest, marker_glob), recursive=True):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        if commit is None:
            if not (sparse and _sparse_clone(url, dest, sparse)):
                subprocess.check_call(["git", "clone", "--depth=1", url, dest], timeout=600)
        else:
            os.makedirs(dest, exist_ok=True)
            subprocess.check_call(["git", "init", "-q"], cwd=dest, timeout=60)
            subprocess.check_call(["git", "remote", "add", "origin", url], cwd=dest, timeout=60)
            try:
                subprocess.check_call(["git", "fetch", "-q", "--depth=1", "origin", commit],
                                      cwd=dest, timeout=600)
            except subprocess.CalledProcessError:
                # fallback for servers that refuse SHA fetches: full history, then checkout
                subprocess.check_call(["git", "fetch", "-q", "origin"], cwd=dest, timeout=900)
            subprocess.check_call(["git", "checkout", "-q", commit], cwd=dest, timeout=60)
        if not glob.glob(os.path.join(dest, marker_glob), recursive=True):
            print(f"  [{tag}] clone of {url} succeeded but {marker_glob} is missing")
            return None
        print(f"  [{tag}] cloned {url}" + (f" @ {commit[:10]}" if commit else "") + f" -> {dest}")
        return dest
    except Exception as e:  # noqa: BLE001
        print(f"  [{tag}] clone failed: {e}")
        return None


def run_generator(module: str, *args: str) -> None:
    """Run one of the deterministic generators in a fresh interpreter. The PrOntoQA
    generator imports the official generator in place (it chdirs and patches
    ``numpy.random.choice``), so it must not run inside the evaluation process."""
    cmd = [sys.executable, "-m", module, *args]
    print("  [generator] " + " ".join(cmd))
    subprocess.check_call(cmd)
