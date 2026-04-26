from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from ..core.splits import fixed_split
from ..data.graph_io import load_grakel_graphs
from ..kernels.wl import wl_gram
from ..kernels.shortest_path import sp_gram
from ..models.krr import select_lambda_krr, fit_krr, predict_krr
from ..models.svr import select_params_svr, fit_svr_precomputed, predict_svr_precomputed
from ..core.metrics import mse, mae, r2

from ..core.plots import make_plots

from ..core.utils import _append_summary

@dataclass(frozen=True)
class RunConfig:
    mode: str = "fixed"
    kernel: str = "wl"  # "wl" or "sp"
    train_end: str = "2019-12-31"
    val_end: str = "2020-12-31"

    labels_path: str = "data/processed/tau_0.40/labels.csv"
    graphs_root: str = "data/graphs"
    processed_root: str = "data/processed"
    results_dir: str = "results_final"

    # WL params
    wl_n_iter: int = 3

    # KRR lambda grid (logspace)
    lambdas: Tuple[float, ...] = (
        1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
        1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0
    )

    # Optional subsampling for quick runs (0 means full)
    n_train_cap: int = 0
    n_val_cap: int = 0
    n_test_cap: int = 0


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


def _cap(dates: List[str], cap: int) -> List[str]:
    return dates if (cap is None or cap <= 0) else dates[:cap]


def _compute_kernel_gram(graphs: List, kernel: str, wl_n_iter: int) -> np.ndarray:
    if kernel == "wl":
        return wl_gram(graphs, n_iter=wl_n_iter, normalize=True)
    if kernel == "sp":
        return sp_gram(graphs, normalize=True)
    raise ValueError("kernel must be 'wl' or 'sp'")


def _resolve_tau_tags(tau_tags_arg: str, graphs_root: Path) -> List[str]:
    if tau_tags_arg.strip():
        tags = [t.strip() for t in tau_tags_arg.split(",") if t.strip()]
        if not tags:
            raise ValueError("--tau_tags fourni mais vide après parsing")
        return tags
    tags = sorted([p.name for p in graphs_root.iterdir() if p.is_dir() and p.name.startswith("tau_")])
    if not tags:
        raise RuntimeError(f"Aucun dossier tau_* trouvé sous {graphs_root}")
    return tags


def _safe_float(v: object) -> float:
    return float(v) if v is not None else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["wl", "sp"], default="wl")
    parser.add_argument("--model", choices=["krr", "svr"], default="krr")
    parser.add_argument("--train_end", default="2019-12-31")
    parser.add_argument("--val_end", default="2020-12-31")
    parser.add_argument("--wl_n_iter", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="cap train/val/test sizes for speed")
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--graphs_root", default="data/graphs")
    parser.add_argument("--processed_root", default="data/processed")
    parser.add_argument(
        "--tau_tags",
        default="",
        help="liste CSV de tags tau (ex: tau_0.25,tau_0.30). Si vide: auto-détection depuis graphs_root",
    )
    parser.add_argument("--tag", default="default")
    parser.add_argument("--results_dir", default="results_final")
    args = parser.parse_args()
    

    t0 = time.time()
    cfg = RunConfig(
        kernel=args.kernel,
        train_end=args.train_end,
        val_end=args.val_end,
        wl_n_iter=args.wl_n_iter,
        n_train_cap=400 if args.quick else 0,
        n_val_cap=120 if args.quick else 0,
        n_test_cap=400 if args.quick else 0,
        labels_path=args.labels_path,
        graphs_root=args.graphs_root,
        processed_root=args.processed_root,
        results_dir=args.results_dir,
    )
    C_grid = (0.1, 1.0, 10.0, 100.0)
    eps_grid = (0.001, 0.01, 0.05)

    labels_path = Path(cfg.labels_path)
    graphs_root = Path(cfg.graphs_root)
    processed_root = Path(cfg.processed_root)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv introuvable: {labels_path}")
    if not graphs_root.exists():
        raise FileNotFoundError(f"dossier graphs_root introuvable: {graphs_root}")
    if not processed_root.exists():
        raise FileNotFoundError(f"dossier processed_root introuvable: {processed_root}")

    labels = pd.read_csv(labels_path)
    labels["date"] = pd.to_datetime(labels["date"]).dt.date.astype(str)
    labels = labels.sort_values("date").reset_index(drop=True)

    train_idx, val_idx, test_idx = fixed_split(labels, cfg.train_end, cfg.val_end)

    train_dates = labels.loc[train_idx, "date"].to_list()
    val_dates = labels.loc[val_idx, "date"].to_list()
    test_dates = labels.loc[test_idx, "date"].to_list()

    # optional caps (quick mode)
    train_dates = _cap(train_dates, cfg.n_train_cap)
    val_dates = _cap(val_dates, cfg.n_val_cap)
    test_dates = _cap(test_dates, cfg.n_test_cap)

    y_train = labels.loc[labels["date"].isin(train_dates), "y"].to_numpy(dtype=float)
    y_val = labels.loc[labels["date"].isin(val_dates), "y"].to_numpy(dtype=float)
    y_test = labels.loc[labels["date"].isin(test_dates), "y"].to_numpy(dtype=float)

    tau_tags = _resolve_tau_tags(args.tau_tags, graphs_root)

    # Results folder
    model_name = args.model
    base = Path(cfg.results_dir) / cfg.mode / args.tag / cfg.kernel / model_name
    run_dir = _next_run_dir(base)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        d=dict(cfg.__dict__)
        d["tag"]=args.tag
        d["model"]=model_name
        json.dump(d, f, indent=2)

    n_tr = len(train_dates)
    n_va = len(val_dates)
    n_te = len(test_dates)

    # indices in the concatenated order
    tr = slice(0, n_tr)
    va = slice(n_tr, n_tr + n_va)
    te = slice(n_tr + n_va, n_tr + n_va + n_te)

    candidate_runs = []
    dates_all = train_dates + val_dates + test_dates

    for tau_tag in tau_tags:
        graphs_dir = graphs_root / tau_tag
        tickers_path = processed_root / tau_tag / "tickers.json"
        if not graphs_dir.exists():
            raise FileNotFoundError(f"graphs dir introuvable pour {tau_tag}: {graphs_dir}")
        if not tickers_path.exists():
            raise FileNotFoundError(f"tickers.json introuvable pour {tau_tag}: {tickers_path}")

        t_graph = time.time()
        graphs_all = load_grakel_graphs(
            dates_all,
            graphs_dir=str(graphs_dir),
            tickers_json=str(tickers_path),
        )
        t_graph_elapsed = time.time() - t_graph

        t_gram = time.time()
        K_all = _compute_kernel_gram(graphs_all, kernel=cfg.kernel, wl_n_iter=cfg.wl_n_iter)
        t_gram_elapsed = time.time() - t_gram

        K_tr = K_all[tr, tr]
        K_va_tr = K_all[va, tr]
        K_te_tr = K_all[te, tr]

        t_fit = time.time()
        run_pack: Dict[str, object] = {
            "tau_tag": tau_tag,
            "load_graphs_sec": float(t_graph_elapsed),
            "gram_sec": float(t_gram_elapsed),
        }

        if model_name == "krr":
            best_lam, val_scores = select_lambda_krr(
                K_train=K_tr,
                y_train=y_train,
                K_val_train=K_va_tr,
                y_val=y_val,
                lambdas=cfg.lambdas,
                metric_fn=mse,
            )
            alpha = fit_krr(K_tr, y_train, best_lam)
            yhat_val = predict_krr(K_va_tr, alpha)
            yhat_test = predict_krr(K_te_tr, alpha)
            best_params = {"tau": tau_tag, "lambda": float(best_lam)}
            search_scores = {str(k): float(v) for k, v in val_scores.items()}
            sv_payload = None

        elif model_name == "svr":
            best, scores = select_params_svr(
                K_train=K_tr,
                y_train=y_train,
                K_val_train=K_va_tr,
                y_val=y_val,
                C_grid=C_grid,
                eps_grid=eps_grid,
                metric_fn=mse,
            )
            m = fit_svr_precomputed(K_tr, y_train, C=best.C, epsilon=best.epsilon)
            yhat_val = predict_svr_precomputed(m, K_va_tr)
            yhat_test = predict_svr_precomputed(m, K_te_tr)
            best_params = {"tau": tau_tag, "C": float(best.C), "epsilon": float(best.epsilon)}
            search_scores = {k: float(v) for k, v in scores.items()}
            sv_idx = [int(i) for i in m.support_.tolist()]
            sv_payload = {
                "n_support_vectors": int(len(sv_idx)),
                "frac_support_vectors": float(len(sv_idx) / len(y_train)),
                "support_vector_dates": [train_dates[i] for i in sv_idx],
            }
        else:
            raise ValueError("unknown model")

        t_fit_elapsed = time.time() - t_fit
        run_pack.update(
            {
                "fit_sec": float(t_fit_elapsed),
                "best_params": best_params,
                "val_scores_grid_mse": search_scores,
                "val_mse": float(mse(y_val, yhat_val)),
                "val_mae": float(mae(y_val, yhat_val)),
                "val_r2": float(r2(y_val, yhat_val)),
                "test_mse": float(mse(y_test, yhat_test)),
                "test_mae": float(mae(y_test, yhat_test)),
                "test_r2": float(r2(y_test, yhat_test)),
                "yhat_val": yhat_val,
                "yhat_test": yhat_test,
                "sv_payload": sv_payload,
            }
        )
        candidate_runs.append(run_pack)

    best_candidate = min(candidate_runs, key=lambda x: float(x["val_mse"]))
    best_params = dict(best_candidate["best_params"])  # type: ignore[arg-type]
    selected_tau = str(best_candidate["tau_tag"])
    search_scores = dict(best_candidate["val_scores_grid_mse"])  # type: ignore[arg-type]
    yhat_val = best_candidate["yhat_val"]  # type: ignore[assignment]
    yhat_test = best_candidate["yhat_test"]  # type: ignore[assignment]

    metrics = {
        "model": model_name,
        "best_params": best_params,
        "val_scores_grid_mse": search_scores,
        "selected_tau": selected_tau,
        "tau_grid": tau_tags,
        "tau_search": [
            {
                "tau": str(c["tau_tag"]),
                "val_mse": _safe_float(c["val_mse"]),
                "test_mse": _safe_float(c["test_mse"]),
                "load_graphs_sec": _safe_float(c["load_graphs_sec"]),
                "gram_sec": _safe_float(c["gram_sec"]),
                "fit_sec": _safe_float(c["fit_sec"]),
            }
            for c in candidate_runs
        ],
        "val_mse": _safe_float(best_candidate["val_mse"]),
        "val_mae": _safe_float(best_candidate["val_mae"]),
        "val_r2": _safe_float(best_candidate["val_r2"]),
        "test_mse": _safe_float(best_candidate["test_mse"]),
        "test_mae": _safe_float(best_candidate["test_mae"]),
        "test_r2": _safe_float(best_candidate["test_r2"]),
        "sizes": {"train": n_tr, "val": n_va, "test": n_te},
    }
    if model_name == "svr":
        sv_payload = best_candidate.get("sv_payload")
        if isinstance(sv_payload, dict):
            metrics.update(sv_payload)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_rows = []
    for d, yt, yh in zip(val_dates, y_val, yhat_val):
        pred_rows.append((d, "val", float(yt), float(yh)))
    for d, yt, yh in zip(test_dates, y_test, yhat_test):
        pred_rows.append((d, "test", float(yt), float(yh)))

    pd.DataFrame(pred_rows, columns=["date", "split", "y", "yhat"]).to_csv(
        run_dir / "predictions.csv", index=False
    )

    make_plots(run_dir)
    tau_search_total = float(
        sum(
            float(c["load_graphs_sec"]) + float(c["gram_sec"]) + float(c["fit_sec"])
            for c in candidate_runs
        )
    )
    timings = {
        "selected_tau": selected_tau,
        "load_graphs_sec": _safe_float(best_candidate["load_graphs_sec"]),
        "gram_sec": _safe_float(best_candidate["gram_sec"]),
        "fit_sec": _safe_float(best_candidate["fit_sec"]),
        "tau_search_total_sec": tau_search_total,
        "total_sec": float(time.time() - t0),
    }
    with open(run_dir / "timings.json", "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)
    mode_dir = Path(cfg.results_dir) / cfg.mode
    best_lambda = best_params.get("lambda", "")
    best_C = best_params.get("C", "")
    best_epsilon = best_params.get("epsilon", "")
    _append_summary(mode_dir, {
        "run_dir": str(run_dir),
        "tag": args.tag,
        "kernel": cfg.kernel,
        "model": model_name,
        "selected_tau": selected_tau,
        "train_end": cfg.train_end,
        "val_end": cfg.val_end,
        "best_lambda": best_lambda,
        "best_C": best_C,
        "best_epsilon": best_epsilon,
        "val_mse": float(metrics["val_mse"]),
        "test_mse": float(metrics["test_mse"]),
        "test_mae": float(metrics["test_mae"]),
        "test_r2": float(metrics["test_r2"]),
        "load_graphs_sec": float(timings["load_graphs_sec"]),
        "gram_sec": float(timings["gram_sec"]),
        "fit_sec": float(timings["fit_sec"]),
        "total_sec": float(timings["total_sec"]),
        "n_train": int(n_tr),
        "n_val": int(n_va),
        "n_test": int(n_te),
    })
    print("OK")
    print("Run dir:", run_dir)
    print("Metrics:", {k: metrics[k] for k in ["best_params", "test_mse", "test_mae", "test_r2"]})
    print("Timings:", timings)


if __name__ == "__main__":
    main()