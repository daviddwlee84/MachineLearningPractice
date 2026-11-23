#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""colab_job.py — run a batch job on a Google Colab VM through the official Colab CLI (`colab`).

Subcommands
  exec    send a Python file with `colab exec`, a long --timeout, and an optional success marker
  upload  upload a file or directory, chunked (Colab rejects single uploads above ~60-94 MB),
          4 parallel requests with retries, reassembled + sha256-checked on the VM
  run     end to end: session -> wheels/uploads -> pip install -> data -> command -> download
          -> stop (unless --keep or the session already existed)

Examples
  colab_job.py run --session gpu1 --gpu T4 --keep --wheel . --pip catboost \\
      --data-spec data.json --cmd "my-train --out /content/job/out" --out runs/colab
  colab_job.py run --dry-run --session s --wheel . --cmd "python -m mypkg.smoke"
  colab_job.py upload --session s ./big.parquet /content/data/big.parquet
  colab_job.py exec --session s --file step.py --marker __DONE__

stdout: one JSON object (the result, or the plan with --dry-run). stderr: progress + remote output.
Exit codes: 0 ok, 1 remote/CLI failure, 2 usage error.
Prereqs: `uv tool install google-colab-cli` and one OAuth sign-in (any `colab` command triggers it).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

CHUNK = 48 * 1024 * 1024  # 60 MB single uploads pass, ~94 MB+ get `400 Bad Request`
EXEC_TIMEOUT = 4 * 3600  # CLI default is 30 s of *silence*; long quiet steps trip it
WORK = "/content/job"
REMOTE_STEPS = Path(__file__).resolve().parent.parent / "assets" / "remote_steps.py"

DRY = False
PLAN: list[str] = []


def log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def colab(*args: str, capture: bool = False) -> str:
    cmd = ["colab", *args]
    PLAN.append(" ".join(cmd))
    log("+", " ".join(cmd))
    if DRY:
        return ""
    out = subprocess.run(cmd, check=True, text=True, capture_output=capture)
    return out.stdout if capture else ""


def exec_file(
    session: str, path: Path, marker: str | None, timeout: int = EXEC_TIMEOUT
) -> None:
    """`colab exec -f` with streamed output. With `marker`, success means the marker was printed:
    a remote exception can still exit 0, and a client timeout after the marker is harmless."""
    cmd = ["colab", "exec", "-s", session, "-f", str(path), "--timeout", str(timeout)]
    PLAN.append(" ".join(cmd) + (f"   # expects {marker!r}" if marker else ""))
    log("+", " ".join(cmd))
    if DRY:
        return
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    seen = False
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stderr.write(line)
        seen |= bool(marker) and marker in line
    rc = proc.wait()
    if marker and seen:
        return
    if rc != 0 or marker:
        raise RuntimeError(
            f"remote step failed (exit {rc}, marker {marker!r} seen: {seen})"
        )


def exec_code(session: str, code: str, marker: str | None = None) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        exec_file(session, Path(f.name), marker)
    finally:
        Path(f.name).unlink()


def remote_step(session: str, params: dict) -> None:
    body = REMOTE_STEPS.read_text()
    code = f"import json\nPARAMS = json.loads({json.dumps(json.dumps(params))})\n{body}"
    exec_code(session, code, marker=f"__STEP_OK__ {params['step']}")


MADE_DIRS: set[tuple[str, str]] = (
    set()
)  # (session, dir) already created in this process


def remote_mkdir(session: str, dirs: list[str]) -> None:
    """One `colab exec` round trip (~5-10 s) per call, so skip dirs made earlier."""
    todo = sorted({d for d in dirs if (session, d) not in MADE_DIRS})
    if not todo:
        return
    code = f"import os\nfor d in {todo!r}: os.makedirs(d, exist_ok=True)\nprint('__MKDIR_OK__')\n"
    exec_code(session, code, marker="__MKDIR_OK__")
    MADE_DIRS.update((session, d) for d in todo)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_retry(session: str, local: Path, remote: str, tries: int = 3) -> None:
    for attempt in range(1, tries + 1):
        try:
            colab("upload", "-s", session, str(local), remote)
            return
        except subprocess.CalledProcessError:
            if attempt == tries:
                raise
            log(f"retry {attempt}/{tries - 1}: {remote}")
            time.sleep(2 * attempt)


def upload_files(
    session: str, pairs: list[tuple[Path, str]], parallel: int = 4
) -> None:
    """Upload local files to remote paths. Large files go up as `.chunk-NNN` pieces and are
    joined + sha256-verified on the VM. Parent dirs are created first (upload needs them)."""
    tmp = Path(tempfile.mkdtemp(prefix="colab-chunks-"))
    jobs: list[tuple[Path, str]] = []
    joins: list[dict] = []
    try:
        for src, dst in pairs:
            if src.stat().st_size <= CHUNK:
                jobs.append((src, dst))
                continue
            with src.open("rb") as fh:
                for i, piece in enumerate(iter(lambda: fh.read(CHUNK), b"")):
                    part = tmp / f"{abs(hash(dst))}.chunk-{i:03d}"
                    if not DRY:
                        part.write_bytes(piece)
                    jobs.append((part, f"{dst}.chunk-{i:03d}"))
            joins.append({"target": dst, "sha256": None if DRY else sha256(src)})
        remote_mkdir(session, [str(Path(r).parent) for _, r in jobs])
        with cf.ThreadPoolExecutor(parallel) as pool:
            list(pool.map(lambda j: upload_retry(session, *j), jobs))
        if joins:
            exec_code(
                session, JOIN_CODE.replace("JOINS", repr(joins)), marker="__JOIN_OK__"
            )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


JOIN_CODE = """
import hashlib, pathlib
for j in JOINS:
    t = pathlib.Path(j["target"])
    parts = sorted(t.parent.glob(t.name + ".chunk-*"))
    with t.open("wb") as out:
        for c in parts:
            out.write(c.read_bytes()); c.unlink()
    h = hashlib.sha256()
    with t.open("rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    if j["sha256"] and h.hexdigest() != j["sha256"]:
        raise RuntimeError(f"sha256 mismatch: {t}")
    print("joined", t, len(parts), "chunks")
print("__JOIN_OK__")
"""


def upload_path(session: str, local: Path, remote: str) -> None:
    """A file goes to `remote`; a directory is tarred and extracted *as* `remote`."""
    if local.is_file():
        upload_files(session, [(local, remote)])
        return
    if not local.is_dir():
        raise SystemExit(f"no such file or directory: {local}")
    tmp = Path(tempfile.mkdtemp(prefix="colab-dir-"))
    try:
        tgz = tmp / f"{local.name}.tgz"
        with tarfile.open(tgz, "w:gz") as tar:
            tar.add(local, arcname=".")
        rtgz = f"{WORK}/_upload/{local.name}-{abs(hash(remote))}.tgz"
        upload_files(session, [(tgz, rtgz)])
        exec_code(
            session,
            f"import os, tarfile\nos.makedirs({remote!r}, exist_ok=True)\n"
            f"tarfile.open({rtgz!r}).extractall({remote!r}, filter='data')\nos.remove({rtgz!r})\n"
            "print('__UNTAR_OK__')\n",
            marker="__UNTAR_OK__",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def kv(items: list[str], what: str) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"{what} must be KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def session_exists(name: str) -> bool:
    if DRY:
        return False
    return f"[{name}]" in colab("sessions", capture=True)


def usage_snapshot() -> dict:
    """Parse `colab usage` (CLI >= 0.8): balance, burn rate, active assignments."""
    if DRY:
        return {}
    try:
        txt = colab("usage", capture=True)
    except subprocess.CalledProcessError:
        return {}
    snap: dict = {}
    for key, pat in {
        "balance": r"balance:\s*([\d.]+)",
        "rate_per_hr": r"rate:\s*([\d.]+)",
        "active": r"assignments:\s*(\d+)",
    }.items():
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            snap[key] = float(m.group(1))
    return snap


class Phases:
    """Wall-clock seconds per phase, for the result metadata."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.secs: dict[str, float] = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        t = time.monotonic()
        try:
            yield
        finally:
            self.secs[name] = round(self.secs.get(name, 0) + time.monotonic() - t, 1)

    def total(self) -> float:
        return round(time.monotonic() - self.t0, 1)


def cost_summary(before: dict, during: dict, after: dict, seconds: float) -> dict:
    out: dict = {"usage_before": before, "usage_during": during, "usage_after": after}
    if "balance" in before and "balance" in after:
        out["units_balance_delta"] = round(before["balance"] - after["balance"], 3)
    if during.get("rate_per_hr"):
        out["units_estimated"] = round(during["rate_per_hr"] * seconds / 3600, 3)
    return out


def cmd_run(a: argparse.Namespace, result: dict) -> dict:
    """Fills `result` in place so timing/cost survive a failed step."""
    spec = json.loads(Path(a.data_spec).read_text()) if a.data_spec else None
    if spec and spec.get("source") not in {"drive", "upload", "present"}:
        raise SystemExit("data-spec.source must be drive | upload | present")
    ph = Phases()
    usage_before = usage_snapshot()
    usage_during: dict = {}
    reused = session_exists(a.session)
    if not reused:
        new = ["new", "-s", a.session]
        if a.gpu.lower() != "none":
            new += ["--gpu", a.gpu]
        if a.high_mem:
            new.append("--high-mem")
        with ph("session_new"):
            colab(*new)
    stop = not (a.keep or reused)
    result |= {"session": a.session, "reused": reused}
    try:
        usage_during = usage_snapshot()  # burn rate while this VM is assigned
        # 1) code: wheels built locally (no git credentials needed on the VM)
        with ph("upload"):
            remote_mkdir(a.session, [f"{WORK}/wheels", a.collect])
            if a.wheel:
                dist = Path(tempfile.mkdtemp(prefix="colab-dist-"))
                for proj in a.wheel:
                    cmd = ["uv", "build", "--wheel", "-q", "-o", str(dist), proj]
                    PLAN.append(" ".join(cmd))
                    log("+", " ".join(cmd))
                    if not DRY:
                        subprocess.run(cmd, check=True, stdout=sys.stderr)
                wheels = sorted(dist.glob("*.whl"))
                upload_files(
                    a.session, [(w, f"{WORK}/wheels/{w.name}") for w in wheels]
                )
                shutil.rmtree(dist, ignore_errors=True)
            for item in a.upload:
                local, _, remote = item.partition(":")
                if not remote:
                    raise SystemExit(f"--upload must be LOCAL:REMOTE, got {item!r}")
                upload_path(a.session, Path(local), remote)
        if a.wheel or a.pip:
            with ph("install"):
                remote_step(
                    a.session,
                    {"step": "install", "wheel_dir": f"{WORK}/wheels", "pip": a.pip},
                )
        # 2) data
        if spec:
            with ph("data"):
                dest = spec.get("dest", "/content/data")
                if spec["source"] == "upload":
                    local_dir = Path(spec["local_dir"])
                    pairs = [
                        (local_dir / f["path"], f"{dest}/{f['path']}")
                        for f in spec["files"]
                    ]
                    upload_files(a.session, pairs)
                remote_step(a.session, {
                    "step": "data", "dest": dest, "files": spec["files"],
                    "source": "drive" if spec["source"] == "drive" else "present",
                    "drive_folder": spec.get("drive_folder", ""),
                })  # fmt: skip
        # 3) command, then collect
        tags = {
            "compute": "colab",
            "colab_gpu": a.gpu,
            "colab_high_mem": str(a.high_mem),
            **kv(a.tag, "--tag"),
        }
        archive = f"{WORK}/result.tgz"
        with ph("run"):
            remote_step(a.session, {
                "step": "run", "cmd": a.cmd, "cwd": WORK, "env": kv(a.env, "--env"),
                "collect": a.collect, "archive": archive, "tags": tags,
            })  # fmt: skip
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        local_tgz = out / f"{a.session}-{int(time.time())}.tgz"
        with ph("download"):
            colab("download", "-s", a.session, archive, str(local_tgz))
            if not DRY:
                with tarfile.open(local_tgz) as tar:
                    try:
                        tar.extractall(out, filter="data")
                    except TypeError:  # Python without tarfile filters
                        tar.extractall(out)
                local_tgz.unlink()
        result |= {"result_dir": str(out / Path(a.collect).name), "tags": tags}
        if a.export_log:
            log_path = out / f"{a.session}-log.{a.export_log}"
            colab("log", "-s", a.session, "-o", str(log_path))
            result["session_log"] = str(log_path)
    finally:
        if stop:
            with ph("session_stop"):
                colab("stop", "-s", a.session)
        result["stopped"] = stop
        seconds = ph.total()
        # The VM keeps billing while kept, so "units" covers only this call's wall time.
        result["timing_s"] = {**ph.secs, "total": seconds}
        result["cost"] = cost_summary(
            usage_before, usage_during, usage_snapshot(), seconds
        )
        meta_path = Path(result.get("result_dir", "")) / "_colab_meta.json"
        if not DRY and meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            meta |= {
                "timing_s": result["timing_s"],
                "cost": result["cost"],
                "session": a.session,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            result["vm"] = meta.get("vm", {})
    return result


def main() -> int:
    global DRY
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned colab/uv commands, run nothing",
    )
    sub = ap.add_subparsers(dest="action", required=True)

    e = sub.add_parser("exec", help="run a Python file on the VM")
    e.add_argument("--session", "-s", required=True)
    e.add_argument("--file", "-f", required=True)
    e.add_argument("--marker", help="success iff this string is printed")
    e.add_argument("--timeout", type=int, default=EXEC_TIMEOUT)

    u = sub.add_parser("upload", help="chunked upload of a file or directory")
    u.add_argument("--session", "-s", required=True)
    u.add_argument("local")
    u.add_argument("remote")

    r = sub.add_parser("run", help="end-to-end job")
    r.add_argument(
        "--session",
        "-s",
        required=True,
        help="created if missing; reused (and kept) if it exists",
    )
    r.add_argument(
        "--gpu",
        default="none",
        help="T4 | L4 | A100 | ... | none (see `colab new --help`)",
    )
    r.add_argument(
        "--high-mem", action="store_true", help="high-RAM shape (paid plans)"
    )
    r.add_argument(
        "--keep", action="store_true", help="do not stop a session this call created"
    )
    r.add_argument(
        "--wheel",
        action="append",
        default=[],
        help="project dir to `uv build --wheel` and install",
    )
    r.add_argument(
        "--pip", action="append", default=[], help="extra requirement to pip install"
    )
    r.add_argument("--upload", action="append", default=[], metavar="LOCAL:REMOTE")
    r.add_argument(
        "--data-spec",
        help="JSON: {source, dest, drive_folder|local_dir, files:[{path, sha256}]}",
    )
    r.add_argument("--cmd", required=True, help=f"shell command, cwd={WORK}")
    r.add_argument("--env", action="append", default=[], metavar="K=V")
    r.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="K=V",
        help="written to _colab_meta.json",
    )
    r.add_argument("--collect", default=f"{WORK}/out", help="remote dir to bring back")
    r.add_argument("--out", default="colab-out", help="local dir to extract into")
    r.add_argument(
        "--export-log",
        choices=["jsonl", "md", "ipynb", "txt"],
        help="also save `colab log` of the session next to the results",
    )
    for p in (e, u, r):
        p.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    ap.add_argument(
        "--no-keyring",
        action="store_true",
        help="CLI >= 0.8 keeps its OAuth token in the OS keyring; a locked keyring (SSH, "
        "headless) stalls every call ~60 s. This forces the keyring to fail fast so the CLI "
        "falls back to ~/.config/colab-cli/token.json (plaintext, if present).",
    )
    a = ap.parse_args()
    DRY = a.dry_run
    if a.no_keyring:
        os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    if shutil.which("colab") is None and not DRY:
        log("colab CLI not found: uv tool install google-colab-cli")
        return 2
    res: dict = {}
    try:
        if a.action == "exec":
            exec_file(a.session, Path(a.file), a.marker, a.timeout)
            res["ok"] = True
        elif a.action == "upload":
            upload_path(a.session, Path(a.local), a.remote)
            res |= {"ok": True, "remote": a.remote}
        else:
            cmd_run(a, res)
    except (RuntimeError, subprocess.CalledProcessError) as err:
        log(f"error: {err}")
        res |= {"ok": False, "error": str(err), "commands": PLAN if DRY else None}
        print(json.dumps(res, indent=2))
        return 1
    res.setdefault("ok", True)
    if DRY:
        res |= {"dry_run": True, "commands": PLAN}
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
