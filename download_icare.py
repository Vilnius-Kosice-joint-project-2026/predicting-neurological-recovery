"""
PhysioNet I-CARE 2.1 Downloader  -  parallel edition
Downloads data where the recording START hour is within the first N hours.

Filename format: PPPP_SSS_EEE_TYPE.ext
  PPPP = patient ID
  SSS  = start hour (zero-padded, 1-based: 001 = first hour)
  EEE  = end hour
  TYPE = EEG | ECG | OTHER | REF

"First 2 hours" means SSS <= 002 (start hour 001 or 002).

How it works:
  Phase 1 - crawl all patient directory listings IN PARALLEL (fast, just HTML)
  Phase 2 - download all matching files IN PARALLEL (2 workers by default)
  Resume-safe: skips files that already exist on disk.

Usage:
    pip install requests
    python download_icare.py                          # first 2 hours
    python download_icare.py --hours 6                # first 6 hours
"""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://physionet.org/files/i-care/2.1/"
MAX_HOURS = 72
DEFAULT_HOURS = 2
DEFAULT_WORKERS = 2
DOWNLOAD_MANIFEST = "icare_download_manifest.json"

# Matches:  PPPP_SSS_EEE_TYPE.ext
# Group 1 = SSS (start hour), Group 2 = TYPE
FILE_RE = re.compile(r"^\d+_(\d{3})_\d{3}_(EEG|ECG|OTHER|REF)\.\w+$")

# ── Manifest management ───────────────────────────────────────────────────────


def load_manifest() -> dict:
    """Load the download manifest (tracks already-downloaded files)."""
    if not Path(DOWNLOAD_MANIFEST).exists():
        return {"downloaded": set(), "version": 1}
    try:
        with open(DOWNLOAD_MANIFEST, "r") as f:
            data = json.load(f)
            data["downloaded"] = set(data.get("downloaded", []))
            return data
    except Exception as e:
        print(f"Warning: Could not load manifest: {e}. Starting fresh.")
        return {"downloaded": set(), "version": 1}


def save_manifest(manifest: dict) -> None:
    """Save the download manifest."""
    try:
        # Convert set to list for JSON serialization
        data = manifest.copy()
        data["downloaded"] = sorted(list(manifest["downloaded"]))
        manifest_path = Path(DOWNLOAD_MANIFEST)
        temp_manifest_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with open(temp_manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        temp_manifest_path.replace(manifest_path)
    except Exception as e:
        print(f"Warning: Could not save manifest: {e}")


def mark_downloaded(rel_path: str, manifest: dict) -> None:
    """Mark a file as downloaded in the manifest."""
    manifest["downloaded"].add(rel_path)


# ── Helpers ───────────────────────────────────────────────────────────────────


def want_file(
    filename: str,
    hours: int,
    patient_id: Optional[str] = None,
    min_patient: Optional[int] = None,
    max_patient: Optional[int] = None,
) -> bool:
    """
    Return True if this file's START hour is within the first `hours` hours.
    Start hours are 1-based (001 = first hour, 002 = second, ...).
    Also accept .txt files unconditionally.
    Optionally filter by patient ID range (min_patient to max_patient).
    """
    # Filter by patient ID range if specified
    if patient_id and (min_patient is not None or max_patient is not None):
        try:
            pid = int(patient_id)
            if min_patient is not None and pid < min_patient:
                return False
            if max_patient is not None and pid > max_patient:
                return False
        except ValueError:
            pass

    if filename.endswith(".txt"):
        return True
    m = FILE_RE.match(filename)
    if not m:
        return False
    return 1 <= int(m.group(1)) <= hours


def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def hour_str(i: int) -> str:
    return f"{i:03d}"


# ── Phase 1: parallel directory crawl ────────────────────────────────────────


def crawl_all_patients(
    hours: int,
    auth,
    workers: int,
    min_patient: Optional[int] = None,
    max_patient: Optional[int] = None,
) -> list:
    """
    Crawl the PhysioNet directory tree and return a list of
    (file_url, local_relative_path) tuples for files we want to download.
    Uses a thread pool so all 607 patient listings are fetched in parallel.
    Optionally filter by patient ID range (min_patient to max_patient).
    """
    try:
        from html.parser import HTMLParser
        from urllib.parse import urljoin

        import requests
    except ImportError:
        sys.exit("Run: pip install requests")

    class HrefParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                for k, v in attrs:
                    if k == "href" and v and not v.startswith("?") and v != "../":
                        self.links.append(v)

    def list_dir(url):
        try:
            r = requests.get(url, auth=auth, timeout=30)
            r.raise_for_status()
            p = HrefParser()
            p.feed(r.text)
            return [urljoin(url, lnk) for lnk in p.links]
        except Exception as e:
            print(f"  [crawl] WARNING: could not list {url}: {e}")
            return []

    def crawl_patient(pdir_url):
        """Return list of (file_url, rel_path) for one patient directory."""
        results = []
        # Extract patient ID from URL
        patient_id = pdir_url.rstrip("/").split("/")[-1]
        for furl in list_dir(pdir_url):
            fname = furl.rstrip("/").split("/")[-1]
            if want_file(fname, hours, patient_id, min_patient, max_patient):
                rel = furl.replace(BASE_URL, "")
                results.append((furl, rel))
        return results

    print("Phase 1/2 - crawling directory listings …", flush=True)
    t0 = time.time()

    # Top level: training/, validation/, etc.
    top = list_dir(BASE_URL)
    section_dirs = [u for u in top if u.endswith("/") and u != BASE_URL]
    if not section_dirs:
        section_dirs = [BASE_URL]

    # Collect all patient sub-directories
    patient_dirs = []
    for sdir in section_dirs:
        entries = list_dir(sdir)
        pdirs = [u for u in entries if u.endswith("/")]
        if pdirs:
            patient_dirs.extend(pdirs)
        else:
            patient_dirs.append(sdir)

    print(
        f"  Found {len(patient_dirs)} patient directories. "
        f"Crawling with {workers} workers …",
        flush=True,
    )

    all_files = []
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(crawl_patient, pd): pd for pd in patient_dirs}
        for future in as_completed(futures):
            files = future.result()
            with lock:
                all_files.extend(files)
                completed += 1
                if completed % 50 == 0 or completed == len(patient_dirs):
                    print(
                        f"  Crawled {completed}/{len(patient_dirs)} dirs, "
                        f"{len(all_files)} files queued …",
                        flush=True,
                    )

    print(
        f"  Crawl done in {time.time()-t0:.1f}s  -  "
        f"{len(all_files)} files to download.\n",
        flush=True,
    )
    return all_files


# ── Phase 2: parallel file download ──────────────────────────────────────────


def download_files_parallel(
    file_list: list, dest: str, auth, workers: int, manifest: dict
) -> dict:
    """
    Download all files in file_list using a thread pool.
    Each worker uses its own requests.Session for connection reuse.
    Returns totals dict.
    Skips files already in manifest (downloaded in previous batches).
    """
    import requests

    # Thread-local sessions - one persistent TCP connection per worker thread
    _local = threading.local()

    def get_session():
        if not hasattr(_local, "session"):
            s = requests.Session()
            if auth:
                s.auth = auth
            # Retry adapter: 3 retries with backoff on 5xx / connection errors
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            retry = Retry(
                total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
            )
            s.mount("https://", HTTPAdapter(max_retries=retry))
            _local.session = s  # ← was missing: assign to thread-local
        return _local.session

    totals = {"ok": 0, "skip": 0, "err": 0}
    lock = threading.Lock()
    manifest_lock = threading.Lock()
    done = [0]  # mutable counter for progress
    pending_manifest_updates = 0
    manifest_save_interval = 25

    def fetch(furl: str, rel: str):
        # Check if already downloaded (in manifest or exists locally)
        if rel in manifest["downloaded"]:
            return "skip", rel, 0, "already_downloaded"

        local = Path(dest) / rel
        if local.exists():
            # Backfill manifest with already-existing local files.
            with manifest_lock:
                manifest["downloaded"].add(rel)
            return "skip", rel, 0, "exists_locally"

        ensure_dir(str(local.parent))
        tmp = local.with_suffix(local.suffix + ".part")
        try:
            session = get_session()
            with session.get(furl, timeout=300, stream=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                        f.write(chunk)
            tmp.rename(local)
            with manifest_lock:
                mark_downloaded(rel, manifest)
            size_mb = local.stat().st_size / 1e6
            return "ok", rel, size_mb, None
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            return "err", rel, str(e), None

    total_files = len(file_list)
    print(
        f"Phase 2/2 - downloading {total_files} files " f"with {workers} workers …",
        flush=True,
    )
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch, furl, rel): (furl, rel) for furl, rel in file_list}
        for future in as_completed(futures):
            should_save_manifest = False
            status, rel, extra, reason = future.result()
            with lock:
                totals[status] += 1
                done[0] += 1
                n = done[0]
                if status == "ok" or (status == "skip" and reason == "exists_locally"):
                    pending_manifest_updates += 1
                    if pending_manifest_updates >= manifest_save_interval:
                        pending_manifest_updates = 0
                        should_save_manifest = True

                if status == "ok":
                    print(
                        f"  [{n}/{total_files}] ↓  {rel}  ({extra:.1f} MB)", flush=True
                    )
                elif status == "skip":
                    # Only print every 50 skips to avoid flooding
                    if n % 50 == 0:
                        reason_str = f" ({reason})" if reason else ""
                        print(
                            f"  [{n}/{total_files}] … {totals['skip']} skipped so far{reason_str}",
                            flush=True,
                        )
                else:
                    print(f"  [{n}/{total_files}] ✗  {rel}: {extra}", flush=True)

            if should_save_manifest:
                with manifest_lock:
                    save_manifest(manifest)

    if pending_manifest_updates:
        with manifest_lock:
            save_manifest(manifest)

    elapsed = time.time() - t0
    mb_total = sum(
        (Path(dest) / rel).stat().st_size / 1e6
        for _, rel in file_list
        if (Path(dest) / rel).exists()
    )
    print(f"\n  Download done in {elapsed:.1f}s  " f"({mb_total/1024:.2f} GB on disk)")
    return totals


# ── Python method (parallel crawl + parallel download) ───────────────────────


def download_python(
    hours: int,
    dest: str,
    workers: int,
    min_patient: Optional[int] = None,
    max_patient: Optional[int] = None,
) -> None:
    auth = None
    manifest = load_manifest()

    file_list = crawl_all_patients(hours, auth, workers, min_patient, max_patient)
    if not file_list:
        print(
            "  No files matched. Check --hours value, patient range, or network access."
        )
        return

    totals = download_files_parallel(file_list, dest, auth, workers, manifest)
    save_manifest(manifest)  # Save updated manifest
    print(
        f"\n  Summary - downloaded: {totals['ok']}, "
        f"skipped: {totals['skip']}, errors: {totals['err']}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download PhysioNet I-CARE 2.1 - first N hours (parallel)."
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help=f"Hours to download (default: {DEFAULT_HOURS})",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="icare_data",
        help="Destination directory (default: icare_data)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel download workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--min-patient",
        type=int,
        default=None,
        help="Minimum patient ID to download (inclusive, e.g. 0284)",
    )
    parser.add_argument(
        "--max-patient",
        type=int,
        default=None,
        help="Maximum patient ID to download (inclusive, e.g. 0330)",
    )
    parser.add_argument(
        "--method",
        choices=["python"],
        default="python",
        help="Download backend (default: python)",
    )
    args = parser.parse_args()

    hours = min(max(1, args.hours), MAX_HOURS)
    ensure_dir(args.dest)

    print("=" * 60)
    print(f"  PhysioNet I-CARE 2.1 - first {hours} hour(s)")
    print(f"  Destination : {Path(args.dest).resolve()}")
    print(f"  Method      : {args.method}")
    print(f"  Workers     : {args.workers}")
    print(f"  Start-hour  : 001 - {hour_str(hours)}  (inclusive, 1-based)")
    if args.min_patient or args.max_patient:
        min_str = f"{args.min_patient:04d}" if args.min_patient else "first"
        max_str = f"{args.max_patient:04d}" if args.max_patient else "last"
        print(f"  Patients    : {min_str} to {max_str}")
    print("=" * 60)

    t0 = time.time()
    if args.method == "python":
        download_python(
            hours, args.dest, args.workers, args.min_patient, args.max_patient
        )

    print(f"\nTotal time: {time.time()-t0:.1f}s  →  {Path(args.dest).resolve()}")


if __name__ == "__main__":
    main()
