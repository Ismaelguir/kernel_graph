from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from ..core.splits import fixed_split
from ..core.plots import make_plots


@dataclass(frozen=True)
class BaselineConfig:
    mode: str = "fixed"
    train_end: str = "2019-12-31"
    val_end: str = "2020-12-31"

    # data inputs
    labels_path: str = "data/processed/labels.csv"
    raw_prices_path: str = "data/raw/adj_close_2014-01-01_2024-12-31.csv"

    # feature window (doit matcher ton build_dataset)
    corr_window: int = 60

    # ridge grid
    lambdas: Tuple[float, ...] = (
        1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
        1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0
    )

    results_dir: str = "results_final"


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


def _log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna()


def _ridge_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, lam: float) -> np.ndarray:
    # ridge fermé: beta=(X'X+lam I)^-1 X'y
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + float(lam) * np.eye(d)
    b = Xtr.T @ ytr
    beta = np.linalg.solve(A, b)
    return Xte @ beta


def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    e = y - yhat
    return float(np.mean(e * e))


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = y.astype(float)
    yhat = yhat.astype(float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sig = X.std(axis=0)
    sig[sig == 0] = 1.0
    return (X - mu) / sig, mu, sig


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    return (X - mu) / sig


def _build_features(labels_dates: List[str], rets: pd.DataFrame, window: int) -> np.ndarray:
    # features basiques sur le portefeuille équipondéré sur la fenêtre passée
    # x1 = mean(r_past), x2 = std(r_past), x3 = cum_return_past (sum log-returns)
    idx = pd.Index(rets.index.date.astype(str))
    R = rets.to_numpy(dtype=float)  # (T,N)

    X = np.zeros((len(labels_dates), 3), dtype=float)
    for k, d in enumerate(labels_dates):
        t = int(np.where(idx == d)[0][0])  # date existe (doit)
        past = R[t - window + 1 : t + 1, :]  # (window,N)
        r_port = past.mean(axis=1)           # (window,)
        X[k, 0] = float(np.mean(r_port))
        X[k, 1] = float(np.std(r_port))
        X[k, 2] = float(np.sum(r_port))
    return X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ridge", "mean"], default="ridge")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--train_end", default="2019-12-31")
    parser.add_argument("--val_end", default="2020-12-31")
    parser.add_argument("--labels_path", default=None)
    parser.add_argument("--raw_prices_path", default=None)
    parser.add_argument("--results_dir", default="results_final")
    args = parser.parse_args()

    t0 = time.time()
    cfg = BaselineConfig(
        train_end=args.train_end,
        val_end=args.val_end,
        labels_path=args.labels_path if args.labels_path else BaselineConfig.labels_path,
        raw_prices_path=args.raw_prices_path if args.raw_prices_path else BaselineConfig.raw_prices_path,
        results_dir=args.results_dir,
    )

    # results dir
    base = Path(cfg.results_dir) / cfg.mode / args.tag / "baseline" / args.model
    run_dir = _next_run_dir(base)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    # load labels
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

    if args.model == "ridge":
        # load prices + compute returns
        prices = pd.read_csv(cfg.raw_prices_path, index_col=0, parse_dates=True)
        prices = prices.dropna(axis=1, how="any")  # univers commun strict
        rets = _log_returns(prices)

        # build features
        t_feat = time.time()
        X_train = _build_features(train_dates, rets, window=cfg.corr_window)
        X_val = _build_features(val_dates, rets, window=cfg.corr_window)
        X_test = _build_features(test_dates, rets, window=cfg.corr_window)
        feat_sec = time.time() - t_feat

        # standardize using train
        Xtr_s, mu, sig = _standardize_fit(X_train)
        Xva_s = _standardize_apply(X_val, mu, sig)
        Xte_s = _standardize_apply(X_test, mu, sig)

        # select lambda on val (min MSE)
        t_fit = time.time()
        best_lam = None
        best_val = None
        scores = {}
        for lam in cfg.lambdas:
            yhat = _ridge_fit_predict(Xtr_s, y_train, Xva_s, lam=float(lam))
            v = _mse(y_val, yhat)
            scores[str(lam)] = float(v)
            if (best_val is None) or (v < best_val):
                best_val = v
                best_lam = float(lam)
        assert best_lam is not None

        # refit implicit (closed-form already uses train) + predict
        yhat_val = _ridge_fit_predict(Xtr_s, y_train, Xva_s, lam=best_lam)
        yhat_test = _ridge_fit_predict(Xtr_s, y_train, Xte_s, lam=best_lam)
        fit_sec = time.time() - t_fit

        metrics = {
            "model": "ridge",
            "best_params": {"lambda": best_lam},
            "val_scores_grid_mse": scores,
            "val_mse": _mse(y_val, yhat_val),
            "val_mae": _mae(y_val, yhat_val),
            "val_r2": _r2(y_val, yhat_val),
            "test_mse": _mse(y_test, yhat_test),
            "test_mae": _mae(y_test, yhat_test),
            "test_r2": _r2(y_test, yhat_test),
            "sizes": {"train": len(train_dates), "val": len(val_dates), "test": len(test_dates)},
            "features": ["mean60_eq", "std60_eq", "sum60_eq"],
        }

        timings = {
            "feature_sec": float(feat_sec),
            "fit_sec": float(fit_sec),
            "total_sec": float(time.time() - t0),
        }
    else:
        ybar = float(np.mean(y_train))
        yhat_val = np.full_like(y_val, ybar, dtype=float)
        yhat_test = np.full_like(y_test, ybar, dtype=float)
        metrics = {
            "model": "mean",
            "best_params": {"ybar_train": ybar},
            "val_mse": _mse(y_val, yhat_val),
            "val_mae": _mae(y_val, yhat_val),
            "val_r2": _r2(y_val, yhat_val),
            "test_mse": _mse(y_test, yhat_test),
            "test_mae": _mae(y_test, yhat_test),
            "test_r2": _r2(y_test, yhat_test),
            "sizes": {"train": len(train_dates), "val": len(val_dates), "test": len(test_dates)},
        }
        timings = {
            "feature_sec": float("nan"),
            "fit_sec": float("nan"),
            "total_sec": float(time.time() - t0),
        }

    cfg_dump = dict(cfg.__dict__)
    cfg_dump["tag"] = args.tag
    cfg_dump["kernel"] = "baseline"
    cfg_dump["model"] = args.model

    (run_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")

    # predictions.csv (val/test)
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