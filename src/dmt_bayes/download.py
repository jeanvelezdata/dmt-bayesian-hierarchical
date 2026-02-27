from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
import requests


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: str | Path, expected_md5: Optional[str] = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If already downloaded and checksum matches, do nothing
    if out_path.exists() and expected_md5:
        got = md5_file(out_path)
        if got.lower() == expected_md5.lower():
            return out_path

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if expected_md5:
        got = md5_file(out_path)
        if got.lower() != expected_md5.lower():
            raise ValueError(f"MD5 mismatch for {out_path}: expected {expected_md5}, got {got}")

    return out_path