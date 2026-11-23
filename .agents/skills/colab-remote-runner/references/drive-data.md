# Data on Google Drive → Colab VM

## Why Drive

Local → VM upload is slow (~2 MB/s) and capped per request. Drive → VM runs on
Google's network: a few GB in a minute or two. So: upload a dataset to Drive
**once** (rclone, Drive desktop, or the web UI), then every session pulls from it.

## Layout and manifest

Keep a versioned folder per dataset snapshot, e.g. `datasets/my-dataset-v1/`
(path relative to *My Drive*). Keep a manifest next to your code (not
necessarily on Drive) listing relative paths and sha256:

```json
{"files": [{"path": "features/2019.parquet", "sha256": "…"},
           {"path": "labels.parquet", "sha256": "…"}]}
```

Generate sha256s locally (`sha256sum`, or `rclone hashsum sha256` against the
local copy). With hashes, the VM skips files that are already present and
correct — rerunning on a kept session costs nothing — and catches truncated
downloads.

## Data spec for `colab_job.py run --data-spec`

| key | meaning |
|---|---|
| `source` | `drive` (Drive API), `upload` (chunked from `local_dir`), `present` (only verify) |
| `dest` | VM directory, default `/content/data` |
| `drive_folder` | folder path under My Drive (for `drive`) |
| `local_dir` | local root that `files[].path` is relative to (for `upload`) |
| `files` | `[{path, sha256?}]` — only what this job needs |

## How the VM fetches (see `assets/remote_steps.py`)

1. `google.auth.default(scopes=[drive.readonly])` — works only after `colab auth`.
2. Walk the folder path by name (`files().list` with `'<parent>' in parents`).
3. `MediaIoBaseDownload` in 64 MiB chunks to `<dest>/<path>.part`, then rename.
4. Re-hash everything; fail on mismatch.

Name lookups take the first match — avoid duplicate file/folder names in the
same Drive folder (Drive allows them).

## Privacy

The dataset sits in your Drive and is downloaded to a Google VM. Don't use this
path for data whose license or policy forbids that.
