# Adapting to a project: the thin-wrapper pattern

`colab_job.py` is project-agnostic. A project gets the best results with a
~100–200 line wrapper (e.g. `scripts/colab/run.py`) that knows three
project-specific things and delegates the rest.

## What the wrapper owns

1. **Which code to ship** — the project plus any local path dependencies
   (submodules, sibling packages) as wheels. Ship config files too (tar the
   `configs/` dir and `--upload` it), because wheels usually don't include them.
2. **Which data a job needs** — derive the file list from the job config
   (e.g. the years/segments it trains and tests on) and the project's data
   manifest; write the data spec JSON.
3. **What to do with results** — the remote command writes to a fixed output
   root with tracking disabled (`--no-tracking`-style flag); after download the
   wrapper calls the project's own "log this run dir" command locally, so
   tracker credentials never leave the machine.

## Sketch

```python
spec = {
    "source": args.data,
    "dest": "/content/data",
    "drive_folder": DRIVE_FOLDER,
    "files": needed_files(cfg),
}  # project logic
Path(tmp, "data.json").write_text(json.dumps(spec))
cmd = f"DATA_DIR=/content/data my-run {args.config} --no-tracking --runs-root /content/job/out"
subprocess.run(
    [
        COLAB_JOB,
        "run",
        "-s",
        session,
        "--wheel",
        ".",
        "--wheel",
        "libs/dep",
        "--upload",
        "configs:/content/job/configs",
        "--data-spec",
        f"{tmp}/data.json",
        "--cmd",
        cmd,
        "--tag",
        f"config={args.config}",
        "--out",
        "runs/colab",
        *(["--keep"] if args.keep else []),
    ],
    check=True,
    stdout=PIPE,
)
result = json.loads(proc.stdout)
log_run_dirs(Path(result["result_dir"]).iterdir())  # project logger, locally
```

## Choices that paid off

- **Model-specific extras** map (`{"CatBoost...": ["catboost>=1.2"]}`) — install
  only what this config needs on top of Colab's preinstalled stack.
- **Memory-lean modes** (streaming / external-memory training) let a standard
  ~12.7 GB VM run what otherwise needs the ~51 GB high-RAM shape.
- **A persistent session for a sweep**: first job uses `source: drive`, later
  ones `source: present`.
- **Record a measured comparison** (local vs Colab: wall time split into boot /
  install / data / fit / download, peak RSS, metric deltas) in the project's
  docs, so the next person knows whether Colab is worth it for their job.
- Expose the same thing as a backend in any job launcher you have
  (pueue / Slurm / Colab behind one `submit` interface).
