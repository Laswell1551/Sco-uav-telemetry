"""Record filenames, sizes, and hashes for local third-party inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_entry(path: Path | None) -> dict:
    if path is None or not path.is_dir():
        return {"status": "not present", "files": []}
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append({
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        })
    return {
        "status": "locally verified",
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def file_entry(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {"status": "not present"}
    return {
        "status": "locally verified",
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uzh-root", type=Path)
    parser.add_argument("--m3ed-root", type=Path)
    parser.add_argument("--sixgl-archive", type=Path)
    parser.add_argument("--access-date", default=date.today().isoformat())
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data" / "external-data-manifest.json",
    )
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "access_date": args.access_date,
        "privacy": "No absolute local path is recorded.",
        "datasets": {
            "uzh_fpv": {
                "official_url": "https://fpv.ifi.uzh.ch/",
                "license": "CC BY-NC-SA 3.0",
                **directory_entry(args.uzh_root),
            },
            "m3ed_falcon": {
                "official_url": "https://m3ed.io/",
                "license": "CC BY-SA 4.0",
                **directory_entry(args.m3ed_root),
            },
            "6gl_cld26_v2": {
                "doi": "10.5281/zenodo.21240929",
                "license": "not explicitly displayed; redistribution disabled",
                **file_entry(args.sixgl_archive),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(args.output)


if __name__ == "__main__":
    main()
