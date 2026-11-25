"""Audit source ownership and current per-video licence on Flickr for AVCaps."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from .acquire import USER_AGENT, save_json, sha

MODIFIABLE = re.compile(r"https://creativecommons\.org/(?:licenses/(?:by|by-sa|by-nc|by-nc-sa)/[\d.]+/|publicdomain/zero/1\.0/)")


def records(html: str, key: str) -> list[dict]:
    match = re.search(rf'"{key}"\s*:\s*', html)
    if not match:
        return []
    entries, _ = json.JSONDecoder().raw_decode(html[match.end():])
    return [entry["data"] for entry in entries]


def lookup(source: str) -> dict:
    url = f"https://www.flickr.com/photo.gne?id={source}"
    row = dict(video_id=source, lookup_url=url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=18) as response:
            row.update(http_status=response.status, canonical_url=response.url)
            page = response.read(3_000_001)
        if len(page) > 3_000_000:
            return dict(row, status="page_too_large")
        row["page_sha256"] = sha(page)
        html = page.decode("utf8", errors="replace")
        targets = [record for record in records(html, "photo-models") if str(record.get("id")) == source]
        if len(targets) != 1 or targets[0].get("mediaType") != "video":
            return dict(row, status="exact_public_video_not_verified")
        item = targets[0]
        owner = str(item.get("owner", {}).get("data", {}).get("id", ""))
        licenses = {str(license["id"]): license for license in records(html, "license-models")}
        license = licenses.get(str(item.get("license")), {})
        license_url = license.get("url", "")
        owner_key = sha(owner.encode()) if re.fullmatch(r"\d+@N\d+", owner) else None
        eligible = bool(owner_key and MODIFIABLE.fullmatch(license_url))
        return dict(row, status="eligible_source_provenance" if eligible else "licence_or_owner_restricted",
                    owner_key=owner_key, license=license.get("name"), license_url=license_url)
    except urllib.error.HTTPError as exc:
        return dict(row, http_status=exc.code, status="http_error")
    except (OSError, TypeError, KeyError, ValueError) as exc:
        return dict(row, status="unverified", failure=type(exc).__name__)


def audit(ids: list[str], root: Path) -> dict:
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r"\d+", source) for source in ids):
        raise ValueError("expected unique Flickr video IDs")
    checks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for i in range(0, len(ids), 3):
            batch = list(pool.map(lookup, ids[i:i + 3]))
            checks.extend(batch)
            if any(row.get("http_status") == 429 for row in batch):
                break
            time.sleep(0.7)
    eligible = [row for row in checks if row["status"] == "eligible_source_provenance"]
    report = dict(total_ids=len(ids), checked=len(checks), statuses=dict(Counter(row["status"] for row in checks)),
                  verified_licences=dict(Counter(row["license"] for row in checks if row.get("license"))),
                  eligible_owner_components=len({row["owner_key"] for row in eligible}), records=checks,
                  limit="Current publisher metadata establishes candidate provenance. It does not establish "
                        "archive-byte identity or event observability.")
    save_json(root / "avcaps-provenance.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(sorted(json.loads(args.annotations.read_text(encoding="utf8"))), args.root)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
