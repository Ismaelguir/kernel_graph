from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .splits import fixed_split
from .plots import make_plots


@dataclass(frozen=True)
class MeanCfg:
    mode: str = "fixed"
    train_end: str = "2019-12-31"
    val_end: str = "2020-12-31"
    labels_path: str = "data/processed/labels.csv"
    results_dir: str = "results"


def _next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not existing:
        run_id = 1
    else:
        last = existing[-1].name.replace("run_", "")
        run_id = int(last) + 1 if last.isdigit() else 1
    run_dir = base / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def mse(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean((y - yhat) ** 2))


def mae(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def r2(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--train_end", default="2019-12-31")
    p.add_argument("--val_end", default="2020-12-31")
    args = p.parse_args()

    t0 = time.time()
    cfg = MeanCfg(train_end=args.train_end, val_end=args.val_end, labels_path=args.labels_path)

    base = Path(cfg.results_dir) / cfg.mode / args.tag / "baseline" / "mean"
    run_dir = _next_run_dir(base)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(cfg.labels_path)
    labels["date"] = pd.to_datetime(labels["date"]).dt.date.astype(str)
    labels = labels.sort_values("date").reset_index(drop=True)

    train_idx, val_idx, test_idx = fixed_split(labels, cfg.train_end, cfg.val_end)
    train_dates = labels.loc[train_idx, "date"].to_list()
    val_dates = labels.loc[val_idx, "date"].to_list()
    test_dates = labels.loc[test_idx, "date"].to_list()

    y_train = labels.loc[labels["date"].isin(train_dates), "y"].to_numpy(dtype=float)
    y_val = labels.loc[labels["date"].isin(val_dates), "y"].to_numpy(dtype=float)
    y_test = labels.loc[labels["date"].isin(test_dates), "y"].to_numpy(dtype=float)

    ybar = float(np.mean(y_train))
    yhat_val = np.full_like(y_val, ybar, dtype=float)
    yhat_test = np.full_like(y_test, ybar, dtype=float)

    metrics = {
        "model": "mean",
        "best_params": {"ybar_train": ybar},
        "val_mse": mse(y_val, yhat_val),
        "val_mae": mae(y_val, yhat_val),
        "val_r2": r2(y_val, yhat_val),
        "test_mse": mse(y_test, yhat_test),
        "test_mae": mae(y_test, yhat_test),
        "test_r2": r2(y_test, yhat_test),
        "sizes": {"train": len(train_dates), "val": len(val_dates), "test": len(test_dates)},
    }
    timings = {"total_sec": float(time.time() - t0)}

    cfg_dump = dict(cfg.__dict__)
    cfg_dump["tag"] = args.tag
    (run_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    pred_rows = []
    for d, yt, yh in zip(val_dates, y_val, yhat_val):
        pred_rows.append((d, "val", float(yt), float(yh)))
    for d, yt, yh in zip(test_dates, y_test, yhat_test):
        pred_rows.append((d, "test", float(yt), float(yh)))
    pd.DataFrame(pred_rows, columns=["date", "split", "y", "yhat"]).to_csv(run_dir / "predictions.csv", index=False)

    make_plots(run_dir)

    print("OK")
    print("Run dir:", run_dir)
    print("Metrics:", {k: metrics[k] for k in ["best_params", "test_mse", "test_mae", "test_r2"]})
    print("Timings:", timings)


if __name__ == "__main__":
    main()