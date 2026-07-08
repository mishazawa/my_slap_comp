#!/usr/bin/env python3
"""
Batch-send EXR files in a folder to a local processing server.

Usage:
    python batch_process_exr.py /path/to/folder
    python batch_process_exr.py /path/to/folder --url http://localhost:9865/process
"""

import argparse
import os
import random
import sys
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def map_working_dir_to_hip(filepath: str) -> str:
    working_dir = os.getenv("WORKING_DIR", "/app/working_dir")
    if filepath.startswith(working_dir):
        return filepath.replace(working_dir, "$HIP", 1)
    return filepath


def process_file(f, url, seed_max, timeout):
    seed = random.randint(0, seed_max)
    resolved_path = map_working_dir_to_hip(str(f.resolve()))
    payload = {"filepath": resolved_path, "seed": seed}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.ok:
            print(f"[OK]   {f.name} (seed={seed}) -> {resp.status_code}")
            return True
        else:
            print(
                f"[FAIL] {f.name} (seed={seed}) -> {resp.status_code}: {resp.text[:200]}"
            )
            return False
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {f.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process EXR files via HTTP POST."
    )
    parser.add_argument("folder", type=str, help="Folder containing .exr files")
    parser.add_argument(
        "--url", default="http://localhost:9865/process", help="Endpoint URL"
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Search subfolders too"
    )
    parser.add_argument(
        "--seed-max", type=int, default=2**31 - 1, help="Max value for random seed"
    )
    parser.add_argument("--workers", type=int, default=10, help="Max workers")
    parser.add_argument(
        "--timeout", type=int, default=600, help="Per-request timeout in seconds"
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: '{folder}' is not a valid folder.")
        sys.exit(1)

    pattern = "**/*.exr" if args.recursive else "*.exr"
    files = sorted(folder.glob(pattern))

    if not files:
        print(f"No .exr files found in '{folder}'.")
        sys.exit(0)

    print(f"Found {len(files)} EXR file(s). Sending to {args.url} ...\n")

    ok, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(process_file, f, args.url, args.seed_max, args.timeout)
            for f in files
        ]
        for future in as_completed(futures):
            if future.result():
                ok += 1
            else:
                failed += 1
    print(f"\nDone. {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
