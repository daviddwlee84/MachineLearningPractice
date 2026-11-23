#!/usr/bin/env python3
"""Run selected experiments from experiments.toml, one after another (VM side).

Runs from the repo root (scripts read `Datasets/...` relative to it). For each
experiment: stream its output, save it to OUT/logs/<name>.log, save any
matplotlib figures to OUT/figures/<name>/, and record exit code, wall seconds
and peak RSS in OUT/summary.json.

Always exits 0 once the summary is written: colab_job.py only downloads the
output dir when the remote command succeeds, and a failed experiment must not
hide the results of the others. Read `summary.json` for per-experiment status.

usage: vm_run_all.py --out DIR NAME [NAME ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SITE = HERE / "_site"  # sitecustomize.py: plt.show() -> savefig


def load_manifest() -> dict[str, dict]:
    raw = tomllib.loads((HERE / "experiments.toml").read_text())
    defaults = raw.get("defaults", {})
    return {e["name"]: {**defaults, **e} for e in raw["experiment"]}


def run_one(exp: dict, out: Path) -> dict:
    name = exp["name"]
    log_path = out / "logs" / f"{name}.log"
    fig_dir = out / "figures" / name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "COLAB_FIG_DIR": str(fig_dir),
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(SITE), os.environ.get("PYTHONPATH", "")) if p
        ),
    }
    timeout_s = int(float(exp["timeout_min"]) * 60)
    print(f"\n===== {name} ({exp['script']}, timeout {timeout_s}s) =====", flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(
        [sys.executable, exp["script"]],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    timed_out = threading.Event()

    def kill() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, kill)
    timer.start()
    assert proc.stdout is not None
    with log_path.open("w") as log:
        for line in proc.stdout:  # keep streaming: silence trips the client timeout
            log.write(line)
            print(f"[{name}] {line}", end="", flush=True)
    _, status, rusage = os.wait4(proc.pid, 0)
    timer.cancel()
    rc = os.waitstatus_to_exitcode(status)
    proc.returncode = rc
    secs = round(time.monotonic() - t0, 1)
    rec = {
        "name": name,
        "script": exp["script"],
        "tier": exp.get("tier"),
        "rc": rc,
        "ok": rc == 0 and not timed_out.is_set(),
        "timed_out": timed_out.is_set(),
        "seconds": secs,
        "peak_rss_gb": round(rusage.ru_maxrss / 1024**2, 3),  # KiB on Linux
        "figures": sorted(p.name for p in fig_dir.glob("*.png")),
    }
    print(f"===== {name}: rc={rc} {secs}s ok={rec['ok']} =====", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("names", nargs="+")
    a = ap.parse_args()
    manifest = load_manifest()
    unknown = [n for n in a.names if n not in manifest]
    if unknown:
        print(f"unknown experiments: {unknown}", file=sys.stderr)
        return 2
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for n in a.names:
        results.append(run_one(manifest[n], a.out))
        # rewrite after every experiment so a lost connection still leaves partial results
        (a.out / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    failed = [r["name"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} ok; failed: {failed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
