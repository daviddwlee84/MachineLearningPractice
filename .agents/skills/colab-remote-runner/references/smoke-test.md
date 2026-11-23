# Smoke test: public data → Colab → metrics back

Two tiny jobs that exercise the whole path (session, upload of a file and a
directory, remote run, download, cost/time report, stop) in about a minute of
VM time. Use them after a CLI upgrade or before trusting a new project wrapper.

Data: the Everest `summit_master` table (45 000 rows, binary target
`summited`, ~38 % positive) from a public [Datasette](https://datasette.io/)
instance. Drop the leaky columns `turnaround_reason serious_incident` and the
ids `rowid climber_id`.

```bash
cd skills/local/colab-remote-runner
# 1) fetch once, locally (polite: 100-row pages, 429-aware)
./assets/examples/fetch_datasette.py https://datasette.exe.xyz everest summit_master /tmp/smoke/everest.csv

# 2) optional local sanity run
uv run --with lightgbm --with pandas --with scikit-learn \
    assets/examples/lightgbm_smoke.py --data /tmp/smoke/everest.csv --target summited \
    --drop turnaround_reason serious_incident rowid climber_id --out /tmp/smoke/local

# 3) LightGBM on a CPU VM (cheapest shape); session created and stopped by the script
./scripts/colab_job.py run -s smoke-lgbm \
    --upload /tmp/smoke/everest.csv:/content/job/data/everest.csv \
    --upload assets/examples:/content/job/examples \
    --cmd "python examples/lightgbm_smoke.py --data data/everest.csv --target summited \
           --drop turnaround_reason serious_incident rowid climber_id --out out" \
    --tag example=lightgbm --out /tmp/smoke/colab-lgbm

# 4) Keras MLP on a T4 (Colab ships Keras 3 + TensorFlow)
./scripts/colab_job.py run -s smoke-keras --gpu T4 \
    --upload /tmp/smoke/everest.csv:/content/job/data/everest.csv \
    --upload assets/examples:/content/job/examples \
    --cmd "python examples/keras_smoke.py --data data/everest.csv --target summited \
           --drop turnaround_reason serious_incident rowid climber_id --out out" \
    --tag example=keras --out /tmp/smoke/colab-keras

colab sessions   # expect nothing left running
```

Add `--no-keyring` (before `run`) if every `colab` call stalls ~60 s on a
locked keyring (CLI ≥ 0.8, see `gotchas.md`).

## What came back (2026-11, google-colab-cli 0.8.0)

LightGBM, CPU VM (Python 3.13, 2 vCPU, 12.7 GB RAM, lightgbm 4.6.0 vs 4.7.0 locally):

| | Colab | local |
|---|---|---|
| test AUC / logloss | 0.758 / 0.5608 | 0.758 / 0.5608 |
| best iteration | 165 | 165 |

`timing_s`: `session_new` 4.7 · `upload` 35.4 · `run` 14.1 (fit 3.4 s,
`cmd_seconds` 10.1, `peak_rss_gb` 0.27) · `download` 2.0 · `session_stop` 2.3 ·
**total 63 s**. `cost`: rate 0.08 units/hr while assigned →
`units_estimated` 0.001; `units_balance_delta` 0.0 (balance updates lag).

Keras MLP, T4 VM (Python 3.13, 2 vCPU, 12.7 GB RAM, Tesla T4 15 GB; Keras 3.13.2
on TensorFlow): early-stopped after 11 epochs, test AUC 0.7575 (local Keras
3.15 / torch on an RTX-class GPU: 0.7566 after 3 epochs), fit 11.7 s, `nan_free`.
`timing_s`: `session_new` 4.8 · `upload` 33.7 · `run` 24.8 · `download` 0.9 ·
total 68.7 s (session kept, then stopped by hand). `cost`: rate 1.15 units/hr
while assigned → `units_estimated` 0.022.

The first T4 attempt died with `RuntimeError: Connection was lost.` at the
start of the run step; the identical retry on a new session succeeded.
`units_balance_delta` stayed 0.0 for every call — the balance shown by
`colab usage` is updated in coarse, delayed steps, so use `units_estimated`
for per-job accounting and the balance for weekly totals.

## Reading the result

- `ok: true`, `result_dir` exists, `result_dir/metrics.json` is there.
- `result_dir/_colab_meta.json` carries `tags`, `vm`, `timing_s`, `cost`.
- Upload dominated a 1-minute job even for a 6 MB CSV plus a small dir: each
  `colab upload` / `colab exec` round trip costs seconds. Batch inputs into as
  few uploads as possible (one dir → one tar) and reuse sessions.
- CPU-deterministic models (LightGBM here) should match local metrics up to
  library-version noise; GPU / Keras runs only within a tolerance.
