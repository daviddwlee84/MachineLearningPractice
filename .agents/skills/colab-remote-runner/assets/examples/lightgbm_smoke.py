"""LightGBM smoke job: binary classification on a small tabular CSV (e.g. Datasette's Everest
`summit_master`, target `summited`). Uses only Colab-preinstalled libs (pandas, sklearn, lightgbm).

    python lightgbm_smoke.py --data /content/job/data/everest.csv --target summited \
        --drop turnaround_reason serious_incident rowid climber_id --out /content/job/out

Writes metrics.json, model.txt, feature_importance.csv into --out.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument(
        "--drop", nargs="*", default=[], help="ids and post-outcome (leaky) columns"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--device", default="cpu", help="cpu | gpu (needs a GPU-enabled LightGBM build)"
    )
    a = ap.parse_args()

    df = pd.read_csv(a.data).drop(columns=a.drop, errors="ignore")
    y = df.pop(a.target).astype(int)
    for c in df.select_dtypes("object"):
        df[c] = df[c].astype("category")
    x_tr, x_te, y_tr, y_te = train_test_split(
        df, y, test_size=0.2, random_state=0, stratify=y
    )

    t0 = time.monotonic()
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        device_type=a.device,
        verbose=-1,
    )
    model.fit(
        x_tr, y_tr, eval_set=[(x_te, y_te)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
    )  # fmt: skip
    fit_s = time.monotonic() - t0
    p = model.predict_proba(x_te)[:, 1]
    metrics = {
        "rows": len(df), "features": df.shape[1], "positive_rate": round(float(y.mean()), 4),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "test_auc": round(float(roc_auc_score(y_te, p)), 4),
        "test_logloss": round(float(log_loss(y_te, p)), 4),
        "fit_seconds": round(fit_s, 2), "lightgbm": lgb.__version__, "device": a.device,
    }  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    model.booster_.save_model(str(out / "model.txt"))
    imp = pd.Series(
        model.booster_.feature_importance("gain"), index=df.columns, name="gain"
    )
    imp.sort_values(ascending=False).to_csv(out / "feature_importance.csv")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
