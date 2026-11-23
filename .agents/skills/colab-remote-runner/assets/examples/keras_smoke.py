"""Keras smoke job: a small MLP on the same tabular CSV, on GPU when the VM has one.
Keras 3 picks its backend from KERAS_BACKEND; this script uses TensorFlow if importable,
else PyTorch (both ship preinstalled on Colab, versions vary).

    python keras_smoke.py --data /content/job/data/everest.csv --target summited \
        --drop turnaround_reason serious_incident rowid climber_id --out /content/job/out

Writes metrics.json and history.csv into --out.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = (
        "tensorflow" if importlib.util.find_spec("tensorflow") else "torch"
    )

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def devices() -> list[str]:
    backend = keras.backend.backend()
    if backend == "tensorflow":
        import tensorflow as tf

        return [d.name for d in tf.config.list_physical_devices("GPU")]
    if backend == "torch":
        import torch

        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    keras.utils.set_random_seed(0)

    df = pd.read_csv(a.data).drop(columns=a.drop, errors="ignore")
    y = df.pop(a.target).astype("float32").to_numpy()
    df = pd.get_dummies(df, columns=list(df.select_dtypes("object")), dtype="float32")
    x = df.fillna(0).astype("float32").to_numpy()
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=0, stratify=y
    )
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd

    model = keras.Sequential([
        keras.Input((x.shape[1],)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])  # fmt: skip
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["AUC"],
    )
    t0 = time.monotonic()
    hist = model.fit(
        x_tr, y_tr, validation_split=0.1, epochs=a.epochs, batch_size=256, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )  # fmt: skip
    fit_s = time.monotonic() - t0
    p = model.predict(x_te, batch_size=1024, verbose=0).ravel()
    metrics = {
        "rows": len(df), "features": int(x.shape[1]), "epochs_run": len(hist.history["loss"]),
        "test_auc": round(float(roc_auc_score(y_te, p)), 4),
        "fit_seconds": round(fit_s, 2), "keras": keras.__version__,
        "backend": keras.backend.backend(), "gpus": devices(),
        "nan_free": bool(np.isfinite(p).all()),
    }  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(hist.history).to_csv(out / "history.csv", index_label="epoch")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
