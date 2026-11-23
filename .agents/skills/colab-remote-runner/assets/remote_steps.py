"""Runs ON the Colab VM. `colab_job.py` prepends `PARAMS = {...}` and sends this file with
`colab exec -f`, one step per call (PARAMS["step"]):

install   pip install uploaded wheels (+ extra requirements Colab does not preinstall)
data      make a local data dir match a manifest: keep sha256-verified files, fetch the rest
          from Google Drive (API; needs `colab auth` in this session). With source=present it
          only verifies (colab_job.py has already uploaded + reassembled the files).
run       run a shell command, then tar a results dir for `colab download`

Every step ends by printing `__STEP_OK__ <step>`. The caller treats a missing marker as failure,
because `colab exec` can exit 0 even when this code raised.
"""

# ruff: noqa: F821  (PARAMS is injected by colab_job.py)
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd: list, **kw) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def step_install(p: dict) -> None:
    wheel_dir = Path(p["wheel_dir"])
    wheels = (
        sorted(str(w) for w in wheel_dir.glob("*.whl")) if wheel_dir.exists() else []
    )
    reqs = [*wheels, *p.get("pip", [])]
    if reqs:
        sh(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--find-links",
                str(wheel_dir),
                *reqs,
            ]
        )
    print(
        json.dumps(
            {"installed": [Path(w).name for w in wheels], "pip": p.get("pip", [])}
        )
    )


# --- Drive API (credentials come from `colab auth -s <session>`; drivemount fails headless) ---


def _drive_service():
    import google.auth
    from googleapiclient.discovery import build

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _child(svc, parent: str, name: str, folder: bool) -> str:
    q = f"name = '{name}' and '{parent}' in parents and trashed = false"
    if folder:
        q += " and mimeType = 'application/vnd.google-apps.folder'"
    found = svc.files().list(q=q, fields="files(id)", pageSize=2).execute()["files"]
    if not found:
        raise FileNotFoundError(
            f"Drive {'folder' if folder else 'file'} not found: {name}"
        )
    return found[0]["id"]


def _drive_folder(svc, path: str) -> str:
    """Folder id for 'a/b/c' under My Drive root."""
    parent = "root"
    for name in path.strip("/").split("/"):
        parent = _child(svc, parent, name, folder=True)
    return parent


def _drive_get(svc, folder: str, rel: str, dest: Path) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    *dirs, name = rel.split("/")
    for d in dirs:
        folder = _child(svc, folder, d, folder=True)
    file_id = _child(svc, folder, name, folder=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        dl = MediaIoBaseDownload(
            f, svc.files().get_media(fileId=file_id), chunksize=1 << 26
        )
        done = False
        while not done:
            _, done = dl.next_chunk()
    tmp.replace(dest)


def step_data(p: dict) -> None:
    dest = Path(p["dest"])
    files = p["files"]  # [{"path": "rel/path", "sha256": "..." (optional)}]
    todo = [
        f
        for f in files
        if not (dest / f["path"]).exists()
        or (f.get("sha256") and sha256(dest / f["path"]) != f["sha256"])
    ]
    print(
        json.dumps(
            {"verified": len(files) - len(todo), "to_fetch": [f["path"] for f in todo]}
        )
    )
    if todo and p["source"] == "drive":
        svc = _drive_service()
        try:
            folder = _drive_folder(svc, p["drive_folder"])
        except Exception as e:
            if "metadata" in str(e) or "RefreshError" in type(e).__name__:
                raise SystemExit(
                    "No Google credentials on this VM: run `colab auth -s <session>` once"
                    " (interactive), then rerun against the same session; or use source=upload."
                ) from None
            raise
        for f in todo:
            print("drive ->", f["path"], flush=True)
            _drive_get(svc, folder, f["path"], dest / f["path"])
    elif todo:
        raise FileNotFoundError(
            f"missing or mismatched (source={p['source']}): {[f['path'] for f in todo]}"
        )
    bad = [
        f["path"]
        for f in files
        if f.get("sha256") and sha256(dest / f["path"]) != f["sha256"]
    ]
    if bad:
        raise RuntimeError(f"sha256 mismatch after fetch: {bad}")
    print(json.dumps({"data_ok": len(files), "dest": str(dest)}))


def vm_info() -> dict:
    info = {"python": platform.python_version(), "cpus": os.cpu_count()}
    try:
        with open("/proc/meminfo") as f:
            kb = int(next(line for line in f if line.startswith("MemTotal")).split()[1])
        info["ram_gb"] = round(kb / 1024**2, 1)
    except (OSError, StopIteration, ValueError):
        pass
    try:
        q = "--query-gpu=name,memory.total,driver_version"
        out = subprocess.run(
            ["nvidia-smi", q, "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        info["gpus"] = (
            [g.strip() for g in out.stdout.strip().splitlines()]
            if out.returncode == 0
            else []
        )
    except FileNotFoundError:
        info["gpus"] = []
    return info


def step_run(p: dict) -> None:
    env = {**os.environ, **p.get("env", {})}
    collect = Path(p["collect"])
    collect.mkdir(parents=True, exist_ok=True)
    print("+", p["cmd"], flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(
        p["cmd"], shell=True, cwd=p["cwd"], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    for line in proc.stdout:  # stream: a quiet step is what trips client timeouts
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"command exited {proc.returncode}")
    meta = {
        "tags": p.get("tags", {}),
        "vm": vm_info(),
        "cmd": p["cmd"],
        "cmd_seconds": round(time.monotonic() - t0, 1),
        # ru_maxrss is KiB on Linux: the largest single descendant process, not the sum
        "peak_rss_gb": round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2, 2
        ),
    }
    (collect / "_colab_meta.json").write_text(json.dumps(meta, indent=2))
    with tarfile.open(p["archive"], "w:gz") as tar:
        tar.add(collect, arcname=collect.name)
    print(json.dumps({"collected": str(collect), "archive": p["archive"]}))


{"install": step_install, "data": step_data, "run": step_run}[PARAMS["step"]](PARAMS)
print(f"__STEP_OK__ {PARAMS['step']}", flush=True)
