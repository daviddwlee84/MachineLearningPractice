---
name: colab-remote-runner
description: Use Google Colab as a remote batch-compute backend through the official Colab CLI (`colab new/exec/upload/download/stop/auth`) — ship code as wheels, pull large data from Google Drive via the Drive API, run a command on a GPU / high-RAM VM, download the results, and stop the session. Use when the user says "run this on Colab", "offload to a Colab GPU", "my local GPU is busy", "colab CLI", "colab exec", "colab upload fails with 400", "drivemount fails", or wants to fan out a few configs to Colab. Not for interactive notebook work (use the Colab web UI) or for scheduled/queued cluster jobs (use slurm-hpc / pueue-job-queue).
---

# colab-remote-runner — Colab as a remote compute resource

Treat a Colab VM like a disposable remote box: create a session, push code and
data, run one or more batch commands, pull results back, stop it. Everything
goes through the official `colab` CLI (`uv tool install google-colab-cli`).
Verified with google-colab-cli 0.7.5 (2026-10) and 0.8.0 (2026-11); re-check
the gotchas after upgrading — the CLI is young and moves fast (`colab skill`
prints the CLI's own bundled agent notes).

## When to use / not

- ✅ Local GPU/RAM is busy or too small; want a T4/L4/A100 or a ~51 GB high-RAM
  shape for a few hours; want to run 2–3 configs in parallel off-box.
- ✅ The job is a **batch command** with a known input set and an output dir.
- ❌ Interactive exploration → Colab web UI. ❌ Queued / scheduled work across
  many jobs → `slurm-hpc`, `pueue-job-queue`. ❌ Data that may not leave your
  machine or must not sit in Google Drive.

## The shape of a run

```
local                                    Colab VM (/content/job)
uv build --wheel  ──colab upload──▶  wheels/ ── pip install (+ extras)
data in Drive     ─────────────────▶  Drive API download (Google network, fast)
                                      run CMD → OUT dir → tar
runs/ ◀──────────colab download────  result.tgz
colab stop (unless --keep)
```

`scripts/colab_job.py` implements all of it. Default flow:

```bash
# 1) One kept session per working block; Drive auth is interactive, once per session
colab new -s work --gpu T4 --high-mem
colab auth -s work            # browser sign-in, paste the code back (needed for Drive)

# 2) Plan first, then run any number of jobs on the same session
scripts/colab_job.py --dry-run run -s work --wheel . --data-spec data.json --cmd "..." --out runs/colab
scripts/colab_job.py run -s work --wheel . --pip catboost \
    --data-spec data.json --cmd "my-train --config cfg.yaml --out /content/job/out" \
    --tag config=cfg --out runs/colab
# later jobs on the same session: set "source": "present" in data.json (sha256 re-checked)

# 3) ALWAYS stop when done — idle sessions keep burning compute units
colab stop -s work            # `colab sessions` to see what is running
```

`run` creates the session if it does not exist and stops it at the end; an
existing session (or `--keep`) is left running. stdout is one JSON object;
everything else streams to stderr. Show the user its cost/time block:

| key | what |
|---|---|
| `timing_s` | wall seconds per phase: `session_new`, `upload`, `install`, `data`, `run`, `download`, `session_stop`, `total` |
| `cost.usage_before/during/after` | `colab usage` snapshots: `balance` (compute units), `rate_per_hr`, `active` |
| `cost.units_estimated` | `rate_per_hr` while assigned × this call's wall time |
| `cost.units_balance_delta` | balance before − after (can lag; trust the estimate for short runs) |
| `vm` | Python, CPUs, RAM, `nvidia-smi` GPUs on the VM |
| `result_dir` | local copy of the output dir; its `_colab_meta.json` holds all of the above + `cmd_seconds`, `peak_rss_gb`, tags |

`--export-log jsonl|md|ipynb` also saves `colab log` (the session's executed cells).

For a single self-contained script with no data or wheels, CLI ≥ 0.8's
`colab run [--gpu T4] script.py args…` (new + exec + stop in one call, also
usable as a shebang) is simpler than this script — but verify how it reports a
remote failure before scripting around its exit code.

## Decisions (defaults first)

| Question | Default | Why |
|---|---|---|
| How to ship code | `--wheel <project dir>` (local `uv build`) | No git token on the VM; exact local version; local path deps → pass each as its own `--wheel` |
| Where large data comes from | **Drive** (`"source": "drive"`) | Local upload is ~2 MB/s and chunked; Drive → VM is Google-internal and fast |
| When to `upload` data | `--smoke`-sized inputs (≤ ~1–2 GB) or no `colab auth` yet | ~10 min per GB-ish; fine for a smoke test only |
| Session lifetime | one named session + `--keep` for a block of runs | Amortizes boot, pip install, Drive auth and data download |
| Results | tar the output dir, `colab download`, log **locally** | Credentials for your tracker (MLflow etc.) never go to the VM |
| Provenance | tag runs `compute=colab`, GPU, high-mem | Colab package versions differ from local; you must be able to tell runs apart |

Data spec (`--data-spec data.json`; details in `references/drive-data.md`):

```json
{"source": "drive", "dest": "/content/data", "drive_folder": "datasets/my-dataset-v1",
 "files": [{"path": "features/2020.parquet", "sha256": "…"}, {"path": "labels.parquet", "sha256": "…"}]}
```

Only list the files **this** job needs (e.g. the years a config covers) —
fetching everything is the most common waste of time.

## Integrating into a project

Don't make the project depend on this script. Write a thin project wrapper that
derives the file list from the project's config, calls `colab_job.py run`
(or vendors its functions), then hands `result_dir` to the project's own
logger. Pattern + example in `references/adapting-to-a-project.md`.

## Gotchas

Each has symptom → cause → fix in `references/gotchas.md`.

- **`colab upload` → `400 Client Error: Bad Request`**: single request above
  ~60–94 MB is rejected, **and** the parent dir must already exist. Upload
  48 MiB chunks, mkdir first, reassemble + sha256 on the VM (`upload` does this).
- **`colab exec` exits 0 even when the remote code raised** (or called
  `sys.exit(3)`; re-checked on 0.8.0). Make every remote
  step print a marker (`__STEP_OK__ <step>`) and treat a missing marker as failure.
- **`colab exec --timeout` defaults to 30 s of silence**, not total runtime.
  pip / tar / a quiet `fit` trip it while the VM is still working. Pass a long
  timeout (script: 4 h) and treat "marker seen" as success even if the client
  then times out.
- **`colab drivemount` fails headless** (`ValueError: mount failed`). Use
  `colab auth -s S` (interactive, scope includes Drive) + the Drive API instead.
- **`RefreshError … metadata.google.internal … 404`** on the VM = no Google
  credentials yet; Colab VMs have no default service account. Run `colab auth`.
- **Auth can't be fully unattended.** `colab auth` needs a human in a browser
  once per session → that's why sessions are kept and reused.
- **CLI ≥ 0.8 keeps its OAuth token in the OS keyring.** Over SSH / headless the
  keyring is often locked: every `colab` call then waits 2 × 30 s, warns, and
  may re-prompt for sign-in. Unlock it, use `colab --auth=adc` (gcloud ADC with
  the scopes in `colab skill`), or pass `colab_job.py --no-keyring`
  (`PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring`: falls back to the
  plaintext `~/.config/colab-cli/token.json` left from 0.7 — chmod it 600).
- **`colab auth` ≠ CLI auth.** `colab auth -s S` puts Google credentials *on the
  VM* (for Drive / GCS); a 401/403 from the CLI itself is a local credential or
  scope problem.
- **`RuntimeError: Connection was lost.`** mid-step: the kernel websocket
  dropped and the session may be gone server-side (`colab sessions` prunes it).
  Nothing to resume — rerun on a fresh session; keep steps idempotent.
- **Idle sessions bill.** `colab sessions`; `colab stop -s S`. Check
  `colab usage` for compute units.
- **Environment drift**: Colab ships its own Python (3.13 at time of writing),
  torch, xgboost, lightgbm; some libs (e.g. catboost) are absent → `--pip`.
  Expect tiny numeric differences vs local; CPU-deterministic paths can match
  bit-for-bit.
- **Credentials live only on the VM** and vanish on `colab stop`; nothing to
  clean up locally — but also nothing to reuse next session.
- **Speed calibration**: T4 is several × slower than a desktop RTX-class GPU;
  a full run that is mostly upload time is a sign you should move data to Drive.

## Validation loop

1. `--dry-run` → read the planned commands (session, uploads, steps, download, stop).
2. First real run with a smoke-sized config on a CPU/T4 session.
3. Confirm the result dir arrived and `_colab_meta.json` carries the tags.
4. `colab sessions` → nothing left running that you did not mean to keep.

## Reference files

- `scripts/colab_job.py` — `exec` / `upload` / `run`; `--help`, `--dry-run`, JSON stdout, exit 0/1/2.
- `assets/remote_steps.py` — the VM-side `install` / `data` / `run` steps (sent by `colab exec`).
- `assets/examples/` — verified smoke jobs: `fetch_datasette.py` (public Datasette table → CSV,
  429-aware), `lightgbm_smoke.py` (CPU), `keras_smoke.py` (GPU); commands in `references/smoke-test.md`.
- `references/gotchas.md` — symptom-titled traps with causes and fixes.
- `references/drive-data.md` — putting data on Drive, manifest format, Drive API fetch.
- `references/adapting-to-a-project.md` — thin project wrapper pattern.

## See also

- `long-running-jobs` — waiting for a multi-hour Colab run without polling in the model loop.
- `slurm-hpc`, `pueue-job-queue` — when the remote resource is a cluster or a local queue.
- `mlflow-tracking` — logging the downloaded run locally.
