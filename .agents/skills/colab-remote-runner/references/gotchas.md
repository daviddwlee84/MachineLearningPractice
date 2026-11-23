# Colab CLI gotchas (symptom-titled)

Observed with google-colab-cli 0.7.x (2026-10) and 0.8.0 (2026-11). Grep by the error text.

## `Upload failed: 400 Client Error: Bad Request`

- **Cause**: `colab upload` goes through the Jupyter contents API. The parent
  directory must already exist, and a single request above roughly 60–94 MB is
  rejected (60 MB passed, 94 MB and 120 MB failed).
- **Fix**: create remote dirs first (`colab exec` with `os.makedirs`), split
  into 48 MiB chunks, upload 4 at a time with retries, join on the VM and verify
  sha256. `colab_job.py upload` does all of this.
- **Also**: throughput is ~0.9 MB/s per stream, ~2 MB/s with 4 streams from a
  typical home/office uplink. 5–6 GB took ~15–20 min. Put big data on Drive.

## `colab drivemount` → `ValueError: mount failed`

- **Cause**: headless Drive mount needs the interactive auth flow; known
  upstream issue.
- **Fix**: `colab auth -s <session>` once (browser sign-in, paste code back),
  then download with the Drive API (`google.auth.default()` +
  `googleapiclient`), see `drive-data.md`.

## Remote step failed but `colab exec` exited 0

- **Cause**: the CLI prints the cell's traceback but does not propagate a
  failing exit status — neither for an exception nor for `sys.exit(3)`
  (re-checked on 0.8.0, 2026-11).
- **Fix**: every remote step prints `__STEP_OK__ <step>` as its last line; the
  caller fails if the marker never appeared.

## `TimeoutError: Timeout waiting for reply` after the work finished

- **Cause**: `colab exec --timeout` defaults to 30 s and fires when the remote
  side is *quiet* that long (pip install, tar of a results dir, a model fit
  without logging) — not after 30 s of total runtime.
- **Fix**: pass a long `--timeout` (e.g. 14400); stream output line by line;
  if the success marker was seen, a later client timeout is harmless. The first
  time this bit, the results archive was never downloaded although the run had
  succeeded.

## `RefreshError … metadata.google.internal … 404` on the VM

- **Cause**: Colab VMs have no default service account; Google APIs have no
  credentials until a user signs in.
- **Fix**: `colab auth -s <session>`. Credentials live only on that VM and
  disappear with `colab stop`.

## Session still running hours later

- **Cause**: `--keep` or a crash before `colab stop`; idle sessions keep using
  compute units.
- **Fix**: `colab sessions` → `colab stop -s <name>`; keep the `stop` in a
  `finally:` block of any wrapper; check `colab usage`.

## Results differ slightly from the local run

- **Cause**: Colab's preinstalled versions (Python, torch, xgboost, lightgbm…)
  differ from your lockfile; GPU kernels can be non-deterministic.
- **Fix**: tag runs with `compute=colab`; compare metrics with a tolerance;
  pin critical libs via `--pip name==version` if bit-equality matters.

## Every `colab` call hangs ~60 s, then `Could not read the OAuth token from the OS keyring`

- **Cause**: CLI 0.8 moved its OAuth token into the OS keyring (Secret Service
  on Linux). Over SSH / in a headless shell the keyring is locked or absent;
  each call waits for it twice (read + save) before falling back, and may warn
  "You will be asked to sign in again".
- **Fix**: unlock the keyring in that login session, or `colab --auth=adc`
  (gcloud application-default credentials with the scopes `colab skill` lists),
  or force a fast fail with `PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring`
  (`colab_job.py --no-keyring`). With the fail backend the CLI uses the
  plaintext `~/.config/colab-cli/token.json` written by 0.7 — `chmod 600` it.

## Shared Datasette host returns `HTTP 429` (examples only)

- **Cause**: `.csv?_stream=on` and `_size=max` pages are rate-limited on
  shared public instances.
- **Fix**: `assets/examples/fetch_datasette.py` pages 100 rows at a time with a
  pause and honours `Retry-After`. Fetch once locally and upload the CSV; do not
  let every VM hit the site.

## `RuntimeError: Connection was lost.` during a step

- **Cause**: the CLI's kernel connection dropped mid-`colab exec` (seen once on
  a fresh T4 session at the start of a training step, no remote output). The
  session was gone afterwards: `colab sessions` printed "Pruned 1 stale local
  session(s)".
- **Fix**: nothing to resume — rerun on a new session (the identical retry
  succeeded). Make steps idempotent (sha256-verified data, results written to a
  fresh dir) so a rerun is cheap. `colab_job.py` still reports `timing_s` and
  `cost` in its failure JSON.
