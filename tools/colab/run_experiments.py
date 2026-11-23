#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Run Algorithm/ experiments on a Google Colab VM, or estimate their cost.

Thin project wrapper around the colab-remote-runner skill's colab_job.py:
it picks experiments from tools/colab/experiments.toml, ships the scripts plus
only the Datasets/ files they read as one tarball, runs them sequentially in a
single Colab session (tools/colab/vm_run_all.py), downloads logs/figures/summary
and writes a report with per-experiment time and compute-unit share.

Examples
  tools/colab/run_experiments.py --estimate                     # table, no Colab call
  tools/colab/run_experiments.py --tier A --dry-run             # planned colab commands
  tools/colab/run_experiments.py --tier A                       # smoke run on a CPU VM
  tools/colab/run_experiments.py --tier A,B,C --rate 0.07       # everything that is ready
  tools/colab/run_experiments.py --only svm-mnist-binary-scratch --keep -s work

stdout: one JSON object (result + report path, or the estimate / plan).
stderr: progress and remote output. Exit codes: 0 ok (see report for failed
experiments), 1 Colab / CLI failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
COLAB_JOB = ROOT / ".claude/skills/colab-remote-runner/scripts/colab_job.py"
REMOTE_TGZ = "/content/job/repo.tgz"
REMOTE_OUT = "/content/job/out"

# Approximate Colab compute-unit burn rates (units/hour). They depend on the
# plan and change over time: the real rate is measured during a run
# (`cost.usage_during.rate_per_hr`) - pass it back with --rate.
APPROX_RATE = {
    "cpu": 0.08,  # measured 2026-11 (standard CPU VM)
    "cpu-highmem": 0.13,
    "T4": 1.8,
    "L4": 4.8,
    "A100": 12.0,
    "H100": 18.0,
}
# Per-session overhead: VM boot + upload + pip install + download + stop.
OVERHEAD_MIN = 1.0  # measured 2026-11: ~25 s boot+upload+install+download+stop
RUNTIME_FLAGS = {  # runtime -> (colab_job --gpu, --high-mem)
    "cpu": ("none", False),
    "cpu-highmem": ("none", True),
    "T4": ("T4", False),
    "L4": ("L4", False),
    "A100": ("A100", False),
    "H100": ("H100", False),
}


def log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def load_manifest() -> list[dict]:
    raw = tomllib.loads((HERE / "experiments.toml").read_text())
    defaults = raw.get("defaults", {})
    return [{"pip": [], "data": [], **defaults, **e} for e in raw["experiment"]]


def select(exps: list[dict], a: argparse.Namespace) -> list[dict]:
    if a.only:
        names = {n.strip() for n in a.only.split(",")}
        unknown = names - {e["name"] for e in exps}
        if unknown:
            raise SystemExit(f"unknown experiments: {sorted(unknown)}")
        return [e for e in exps if e["name"] in names]
    tiers = {t.strip().upper() for t in a.tier.split(",")} if a.tier else None
    status = {s.strip() for s in a.status.split(",")}
    return [
        e
        for e in exps
        if e["status"] in status and (tiers is None or e["tier"] in tiers)
    ]


def estimate(sel: list[dict], a: argparse.Namespace) -> dict:
    rows, by_runtime = [], {}
    for e in sel:
        rt = a.runtime or e["runtime"]
        rate = a.rate if a.rate is not None else APPROX_RATE[rt]
        units = e["est_min"] / 60 * rate
        rows.append({"name": e["name"], "tier": e["tier"], "status": e["status"],
                     "runtime": rt, "est_min": e["est_min"], "units": round(units, 4),
                     "note": e.get("note", "")})  # fmt: skip
        g = by_runtime.setdefault(rt, {"experiments": 0, "est_min": 0.0, "rate_per_hr": rate})
        g["experiments"] += 1
        g["est_min"] += e["est_min"]
    for g in by_runtime.values():  # one session per runtime shape
        g["session_min"] = round(g["est_min"] + OVERHEAD_MIN, 1)
        g["units"] = round(g["session_min"] / 60 * g["rate_per_hr"], 3)
    total = round(sum(g["units"] for g in by_runtime.values()), 3)
    # human table on stderr
    log(f"{'experiment':34} {'tier':4} {'status':8} {'runtime':11} {'min':>6} {'CU':>7}")
    for r in rows:
        log(f"{r['name']:34} {r['tier']:4} {r['status']:8} {r['runtime']:11} {r['est_min']:6.1f} {r['units']:7.3f}")
    for rt, g in by_runtime.items():
        log(f"session {rt:11}: {g['experiments']} experiments, ~{g['session_min']} min incl. "
            f"{OVERHEAD_MIN} min overhead, {g['rate_per_hr']} CU/h -> ~{g['units']} CU")  # fmt: skip
    log(f"total ~{total} CU (rates are approximate unless --rate was given)")
    return {"estimate": rows, "sessions": by_runtime, "total_units": total}


def build_tarball(sel: list[dict], dest: Path) -> tuple[Path, int]:
    files = {Path("pyproject.toml")}
    files |= {p.relative_to(ROOT) for p in (ROOT / "Algorithm").rglob("*.py")}
    files |= {p.relative_to(ROOT) for p in HERE.rglob("*.py")}
    files.add(Path("tools/colab/experiments.toml"))
    for e in sel:
        files |= {Path(d) for d in e["data"]}
    missing = [str(f) for f in files if not (ROOT / f).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    tgz = dest / "repo.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        for f in sorted(files):
            if "__pycache__" not in f.parts:
                tar.add(ROOT / f, arcname=str(f))
    return tgz, len(files)


def parse_result(stdout: str) -> dict:
    """colab_job.py prints one JSON object, but the colab CLI's own progress
    lines (`[colab] Session READY.` ...) leak onto the same stdout before it."""
    start = stdout.find("\n{\n") + 1 if not stdout.startswith("{") else 0
    return json.loads(stdout[start:])


def write_report(run_dir: Path, result: dict, rate_hint: float) -> Path:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else []
    cost = result.get("cost", {})
    timing = result.get("timing_s", {})
    units = cost.get("units_estimated")
    rate = (cost.get("usage_during") or {}).get("rate_per_hr") or rate_hint
    run_s = sum(r["seconds"] for r in summary) or 1.0
    lines = [
        f"# Colab run {run_dir.parent.name}",
        "",
        f"- session: `{result.get('session')}`  VM: {json.dumps(result.get('vm', {}))}",
        f"- measured rate: {rate} CU/h; this call ~{units} CU "
        f"(balance delta {cost.get('units_balance_delta')})",
        "- phases (s): " + ", ".join(f"{k}={v}" for k, v in timing.items()),
        "",
        "| experiment | tier | ok | rc | seconds | peak RSS GB | figures | CU share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary:
        share = round(units * r["seconds"] / run_s, 4) if units else "-"
        ok = "yes" if r["ok"] else ("TIMEOUT" if r["timed_out"] else "NO")
        lines.append(f"| {r['name']} | {r['tier']} | {ok} | {r['rc']} | {r['seconds']} | "
                     f"{r['peak_rss_gb']} | {len(r['figures'])} | {share} |")  # fmt: skip
    n_ok = sum(r["ok"] for r in summary)
    lines += ["", f"{n_ok}/{len(summary)} ok. Logs: `logs/<name>.log`, figures: `figures/<name>/`."]
    report = run_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated experiment names")
    ap.add_argument("--tier", help="comma-separated tiers, e.g. A,B,C (default: all)")
    ap.add_argument("--status", default="ready", help="comma-separated: ready,optional,followup")
    ap.add_argument("--runtime", choices=sorted(RUNTIME_FLAGS), help="override the manifest runtime")
    ap.add_argument("--rate", type=float, help="CU/hour to use for --estimate (measured value)")
    ap.add_argument("--estimate", action="store_true", help="print time/CU estimate only")
    ap.add_argument("--session", "-s", default="ml-practice")
    ap.add_argument("--keep", action="store_true", help="leave the session running")
    ap.add_argument("--use-keyring", action="store_true",
                    help="let the colab CLI use the OS keyring (default: --no-keyring)")  # fmt: skip
    ap.add_argument("--out", default="runs/colab", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    sel = select(load_manifest(), a)
    if not sel:
        log("no experiments selected")
        return 2
    if a.estimate:
        print(json.dumps(estimate(sel, a), indent=2, ensure_ascii=False))
        return 0
    not_ready = [e["name"] for e in sel if e["status"] == "followup"]
    if not_ready:
        log(f"refusing to run follow-up experiments (see their `note`): {not_ready}")
        return 2
    runtimes = {a.runtime or e["runtime"] for e in sel}
    if len(runtimes) > 1:
        log(f"selection mixes runtimes {sorted(runtimes)}; run them separately or pass --runtime")
        return 2
    runtime = runtimes.pop()
    gpu, high_mem = RUNTIME_FLAGS[runtime]
    pip = sorted({p for e in sel for p in e["pip"]})
    names = [e["name"] for e in sel]
    run_dir = (ROOT / a.out / time.strftime("%Y%m%d-%H%M%S")).resolve()

    with tempfile.TemporaryDirectory(prefix="colab-ml-") as tmp:
        tgz, n_files = build_tarball(sel, Path(tmp))
        log(f"tarball: {n_files} files, {tgz.stat().st_size / 1e6:.1f} MB; "
            f"{len(names)} experiments on {runtime}; pip: {pip or '-'}")  # fmt: skip
        remote = (
            f"mkdir -p repo && tar xzf {REMOTE_TGZ} -C repo && cd repo && "
            f"python tools/colab/vm_run_all.py --out {REMOTE_OUT} {' '.join(names)}"
        )
        cmd = [sys.executable, str(COLAB_JOB)]
        cmd += ["--dry-run"] if a.dry_run else []
        cmd += [] if a.use_keyring else ["--no-keyring"]
        cmd += ["run", "-s", a.session, "--gpu", gpu, "--upload", f"{tgz}:{REMOTE_TGZ}",
                "--cmd", remote, "--collect", REMOTE_OUT, "--out", str(run_dir),
                "--tag", f"runtime={runtime}", "--tag", f"experiments={','.join(names)}"]  # fmt: skip
        cmd += ["--high-mem"] if high_mem else []
        cmd += ["--keep"] if a.keep else []
        for p in pip:
            cmd += ["--pip", p]
        proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, text=True)

    try:
        result = parse_result(proc.stdout)
    except json.JSONDecodeError:
        log(proc.stdout)
        log(f"colab_job.py exited {proc.returncode} without JSON output")
        return 1
    if a.dry_run:
        print(json.dumps(result, indent=2))
        return proc.returncode
    out_dir = Path(result.get("result_dir", run_dir / "out"))
    if out_dir.is_dir():
        result["report"] = str(write_report(out_dir, result, APPROX_RATE[runtime]))
        log(Path(result["report"]).read_text())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
