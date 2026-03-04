"""
PhysioNet I-CARE 2.1 Downloader  -  parallel edition
Downloads data where the recording START hour is within the first N hours.

Filename format: PPPP_SSS_EEE_TYPE.ext
  PPPP = patient ID
  SSS  = start hour (zero-padded, 1-based: 001 = first hour)
  EEE  = end hour
  TYPE = EEG | ECG | OTHER | REF

"First 2 hours" means SSS <= 002 (start hour 001 or 002).

How it works (python method):
  Phase 1 - crawl all patient directory listings IN PARALLEL (fast, just HTML)
  Phase 2 - download all matching files IN PARALLEL (8 workers by default)
  Resume-safe: skips files that already exist on disk.

Usage:
    pip install requests
    python download_icare.py                          # first 2 hours, 8 workers
    python download_icare.py --hours 6                # first 6 hours
    python download_icare.py --workers 16             # more parallelism
    python download_icare.py --username USER --password PASS
"""

import argparse
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://physionet.org/files/i-care/2.1/"
MAX_HOURS = 72
DEFAULT_HOURS = 2
DEFAULT_WORKERS = 8

# Matches:  PPPP_SSS_EEE_TYPE.ext
# Group 1 = SSS (start hour), Group 2 = TYPE
FILE_RE = re.compile(r"^\d+_(\d{3})_\d{3}_(EEG|ECG|OTHER|REF)\.\w+$")

# ── Helpers ───────────────────────────────────────────────────────────────────


def want_file(filename: str, hours: int) -> bool:
    """
    Return True if this file's START hour is within the first `hours` hours.
    Start hours are 1-based (001 = first hour, 002 = second, ...).
    Also accept .txt files unconditionally.
    """
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


def crawl_all_patients(hours: int, auth, workers: int) -> list:
    """
    Crawl the PhysioNet directory tree and return a list of
    (file_url, local_relative_path) tuples for files we want to download.
    Uses a thread pool so all 607 patient listings are fetched in parallel.
    """
    try:
        import requests
        from urllib.parse import urljoin
        from html.parser import HTMLParser
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
        for furl in list_dir(pdir_url):
            fname = furl.rstrip("/").split("/")[-1]
            if want_file(fname, hours):
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


def download_files_parallel(file_list: list, dest: str, auth, workers: int) -> dict:
    """
    Download all files in file_list using a thread pool.
    Each worker uses its own requests.Session for connection reuse.
    Returns totals dict.
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
    done = [0]  # mutable counter for progress

    def fetch(furl: str, rel: str):
        local = Path(dest) / rel
        if local.exists():
            return "skip", local.name, 0

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
            size_mb = local.stat().st_size / 1e6
            return "ok", local.name, size_mb
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            return "err", local.name, str(e)

    total_files = len(file_list)
    print(
        f"Phase 2/2 - downloading {total_files} files " f"with {workers} workers …",
        flush=True,
    )
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fetch, furl, rel): (furl, rel) for furl, rel in file_list}
        for future in as_completed(futures):
            status, name, extra = future.result()
            with lock:
                totals[status] += 1
                done[0] += 1
                n = done[0]
                if status == "ok":
                    print(
                        f"  [{n}/{total_files}] ↓  {name}  ({extra:.1f} MB)", flush=True
                    )
                elif status == "skip":
                    # Only print every 50 skips to avoid flooding
                    if n % 50 == 0:
                        print(
                            f"  [{n}/{total_files}] … {totals['skip']} skipped so far",
                            flush=True,
                        )
                else:
                    print(f"  [{n}/{total_files}] ✗  {name}: {extra}", flush=True)

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
    hours: int, dest: str, workers: int, username: str = "", password: str = ""
) -> None:
    auth = (username, password) if username else None

    file_list = crawl_all_patients(hours, auth, workers)
    if not file_list:
        print("  No files matched. Check --hours value or network access.")
        return

    totals = download_files_parallel(file_list, dest, auth, workers)
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
        "--method",
        choices=["python"],
        default="python",
        help="Download backend (default: python)",
    )
    parser.add_argument("--username", type=str, default="")
    parser.add_argument("--password", type=str, default="")
    args = parser.parse_args()

    hours = min(max(1, args.hours), MAX_HOURS)
    ensure_dir(args.dest)

    print("=" * 60)
    print(f"  PhysioNet I-CARE 2.1 - first {hours} hour(s)")
    print(f"  Destination : {Path(args.dest).resolve()}")
    print(f"  Method      : {args.method}")
    print(f"  Workers     : {args.workers}")
    print(f"  Start-hour  : 001 - {hour_str(hours)}  (inclusive, 1-based)")
    print("=" * 60)

    t0 = time.time()
    if args.method == "python":
        download_python(hours, args.dest, args.workers, args.username, args.password)

    print(f"\nTotal time: {time.time()-t0:.1f}s  →  {Path(args.dest).resolve()}")


if __name__ == "__main__":
    main()
