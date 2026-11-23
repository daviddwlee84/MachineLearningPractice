#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""fetch_datasette.py — download one table from a public Datasette instance to CSV, politely.

    fetch_datasette.py https://datasette.exe.xyz everest summit_master data/everest.csv

Walks the JSON API page by page (`_size=100`, follows `next_url`), sleeps between pages and backs
off on HTTP 429. (On shared hosts the streaming `.csv?_stream=on` export and `_size=max` pages get 429.)
Run it once locally and upload the CSV, rather than having every VM hit the site.
stdout: {"rows": N, "columns": [...], "path": "..."}.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def get(url: str, tries: int = 6) -> dict:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "colab-remote-runner-example"}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 2 ** (attempt + 1))
            print(f"429, sleeping {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base")
    ap.add_argument("database")
    ap.add_argument("table")
    ap.add_argument("out")
    ap.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="rows per request (`max` pages get 429 on shared hosts)",
    )
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between pages")
    a = ap.parse_args()
    url = f"{a.base.rstrip('/')}/{a.database}/{a.table}.json?" + urllib.parse.urlencode(
        {"_size": a.page_size, "_shape": "objects"}
    )
    rows: list[dict] = []
    while url:
        page = get(url)
        rows += page["rows"]
        nxt = page.get("next_url")
        url = nxt.replace("http://", "https://", 1) if nxt else ""
        print(f"{len(rows)} rows", file=sys.stderr, flush=True)
        if url:
            time.sleep(a.pause)
    if not rows:
        raise SystemExit("no rows")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"rows": len(rows), "columns": list(rows[0]), "path": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
