# Running the experiments on Google Colab

`tools/colab/` runs the scripts under `Algorithm/` on a Colab VM through the
official `colab` CLI, using the `colab-remote-runner` skill's
`.claude/skills/colab-remote-runner/scripts/colab_job.py` for the session /
upload / exec / download / stop plumbing.

| file | role |
|---|---|
| `experiments.toml` | manifest: script, data files, extra pip packages, tier, recommended runtime, timeout, estimated minutes, status |
| `run_experiments.py` | local entry point: select experiments, `--estimate`, or ship + run + download + `report.md` |
| `vm_run_all.py` | VM side: run the selected experiments one by one, per-experiment logs / figures / `summary.json` |
| `_site/sitecustomize.py` | on the VM, `plt.show()` saves figures to `figures/<name>/` instead of dropping them |

## Quick start

```bash
uv tool install google-colab-cli        # once; any colab command triggers the OAuth sign-in
tools/colab/run_experiments.py --estimate              # time / compute-unit table, no Colab call
tools/colab/run_experiments.py --tier A --dry-run      # planned colab commands
tools/colab/run_experiments.py --tier A,B              # run every ready experiment on one CPU VM
tools/colab/run_experiments.py --only svm-mnist-binary-scratch
colab sessions                                         # make sure nothing is left running
```

Results land in `runs/colab/<timestamp>/out/` (git-ignored): `report.md`,
`summary.json`, `logs/<name>.log`, `figures/<name>/*.png`, `_colab_meta.json`.

The CLI stores its OAuth token in the OS keyring; when the keyring is locked
(SSH / headless) every call waits 2 x 30 s. The wrapper therefore passes
`--no-keyring` by default (falls back to `~/.config/colab-cli/token.json`);
use `--use-keyring` to opt out.

## Runtime choice and cost

None of the experiments benefits from a GPU: they are small scikit-learn / NumPy
/ pure-Python workloads, and the only PyTorch script trains a single linear
layer on a few hundred rows. A standard CPU VM is the right shape for all of
them; GPU runtimes burn compute units 20-100x faster for no speed-up.

| tier | experiments | runtime | Colab CPU wall time |
|---|---|---|---|
| A | 23 sklearn / from-scratch scripts (AdaBoost, LightGBM, KMeans, LinearRegression x3 incl. PyTorch, LogisticRegression, NaiveBayes, PCA, SVD, VSM/LSI gensim, SVM Iris / Post-Operative, kNN) | cpu | 1.6-40 s each |
| B | DecisionTree from scratch, FP-Growth (retail), SVM MNIST binary / multiclass, CLiMF from scratch | cpu | 1.3-11.6 min each |
| D (optional) | HMM jieba vs. HanLP | cpu | not run yet: needs a JDK + ~600 MB HanLP data download |

### Measured on Colab (2026-11, standard CPU VM: Python 3.13.18, 2 vCPU, 12.7 GB RAM)

| run | experiments | ok | session wall time | compute units |
|---|---|---|---|---|
| tier A | 23 | 23/23 | 171 s (run 145 s; boot + upload + install + download + stop ~26 s) | ~0.004 |
| tier B | 5 | 5/5 | 1520 s (run 1497 s) | ~0.034 |
| **all ready** | **28** | **28/28** | **~28 min** | **~0.04** |

- Measured burn rate: **0.08 CU/h** for the standard CPU VM (`colab usage` while assigned).
- Per-experiment Colab times are in `experiments.toml` (`est_min`); peak RSS stayed below 0.8 GB,
  so `--high-mem` is never needed for `Algorithm/`.
- Colab's 2 vCPUs are 4-6x slower than a desktop for the pure-Python loops in tier B
  (FP-Growth: 163 s locally vs. 696 s on Colab). Offloading only pays off when the local
  machine is busy, not for speed.
- For comparison, the same ~28 min on a T4 would cost ~0.8 CU and on an A100 ~5.6 CU,
  with no speed-up.

## Follow-ups (not runnable yet)

| experiment | blocker | suggested runtime |
|---|---|---|
| `Algorithm/CLiMF/CLiMF_Epinions/CLiMF_Epinions_TensorFlow.py` | `tensorflow.contrib.eager` (TF1) - needs a TF2 rewrite | cpu is enough (T4 only if the model grows) |
| `Project/KaggleElo` | ~3 GB Kaggle data (put it on Drive, use a data spec) and > 12.7 GB pandas peak RAM | `cpu-highmem` (~51 GB) |
| `Project/DCIC2019` | competition data not in the repo; `sklearn.externals.joblib` removed; expects feature pickles | cpu |

## Compatibility fixes applied to the scripts

To run on a current stack (Colab ships Python 3.12+/3.13, pandas 2, NumPy 2,
scikit-learn >= 1.6; verified locally with pandas 3.1, NumPy 2.5, scikit-learn 1.9):

- `df.drop([...], 1)` -> `df.drop(columns=[...])` (positional axis removed in pandas 2)
- `np.mat` -> `np.asmatrix` (removed in NumPy 2.0)
- `float()` on a 1x1 matrix -> `as_float()` helper (NumPy 2.5 rejects it)
- `np.matrix` converted with `np.asarray` before it reaches scikit-learn (rejected since 1.8)
- `sklearn.cross_validation` -> `sklearn.model_selection`; `calinski_harabaz_score` -> `calinski_harabasz_score`
- `LogisticRegression(multi_class="ovr")` -> `OneVsRestClassifier(LogisticRegression())`
- `df.mean()` -> `df.mean(numeric_only=True)` (string columns no longer silently skipped)
- `SVM_Iris_Multiclass.py` resolves `../SVM_MNIST` relative to the file, not the cwd

`Algorithm/Adaboost/` is a byte-identical duplicate of `Algorithm/AdaBoost/`
(a case-only rename artifact); the manifest only lists `AdaBoost/`.
