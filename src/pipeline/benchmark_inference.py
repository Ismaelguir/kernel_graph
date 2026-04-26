from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.splits import fixed_split
from ..data.graph_io import load_grakel_graphs
from ..kernels.wl import wl_gram
from ..kernels.shortest_path import sp_gram
from ..models.krr import fit_krr
from ..models.svr import fit_svr_precomputed


def _latest_run_dir(base: Path) -> Path:
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not runs:
        raise RuntimeError(f"Aucun run trouvé sous {base}")
    return runs[-1]


def _parse_tau_to_float(tau_tag: str) -> float:
    raw = tau_tag.replace("tau_", "")
    return float(raw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--kernel", choices=["wl", "sp"], required=True)
    p.add_argument("--model", choices=["krr", "svr"], required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--graphs_root", default="data/graphs")
    p.add_argument("--processed_root", default="data/processed")
    p.add_argument("--results_dir", default="results_final")
    p.add_argument("--train_end", default="2019-12-31")
    p.add_argument("--val_end", default="2020-12-31")
    p.add_argument("--wl_n_iter", type=int, default=3)
    p.add_argument("--n_bench", type=int, default=50)
    args = p.parse_args()

    model_base = Path(args.results_dir) / "fixed" / args.tag / args.kernel / args.model
    run_dir = _latest_run_dir(model_base)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    selected_tau = str(metrics.get("selected_tau") or metrics.get("best_params", {}).get("tau", ""))
    if not selected_tau:
        raise RuntimeError("selected_tau introuvable dans metrics.json")

    graphs_dir = Path(args.graphs_root) / selected_tau
    tickers_path = Path(args.processed_root) / selected_tau / "tickers.json"
    if not graphs_dir.exists():
        raise FileNotFoundError(f"graphs_dir introuvable: {graphs_dir}")
    if not tickers_path.exists():
        raise FileNotFoundError(f"tickers.json introuvable: {tickers_path}")

    labels = pd.read_csv(args.labels_path)
    labels["date"] = pd.to_datetime(labels["date"]).dt.date.astype(str)
    labels = labels.sort_values("date").reset_index(drop=True)

    train_idx, val_idx, test_idx = fixed_split(labels, args.train_end, args.val_end)
    train_dates = labels.loc[train_idx, "date"].to_list()
    test_dates = labels.loc[test_idx, "date"].to_list()

    y_train = labels.loc[labels["date"].isin(train_dates), "y"].to_numpy(dtype=float)

    graphs_train = load_grakel_graphs(
        train_dates,
        graphs_dir=str(graphs_dir),
        tickers_json=str(tickers_path),
    )

    if args.kernel == "wl":
        K_tr = wl_gram(graphs_train, n_iter=args.wl_n_iter, normalize=True)
    else:
        K_tr = sp_gram(graphs_train, normalize=True)

    row_params = metrics.get("best_params", {}) or {}

    if args.model == "krr":
        lam = float(row_params["lambda"])
        alpha = fit_krr(K_tr, y_train, lam)
    else:
        C = float(row_params["C"])
        eps = float(row_params["epsilon"])
        model = fit_svr_precomputed(K_tr, y_train, C=C, epsilon=eps)

    bench_dates = test_dates[: args.n_bench]
    times = []

    for d in bench_dates:
        G_new = load_grakel_graphs([d], graphs_dir=str(graphs_dir), tickers_json=str(tickers_path))[0]
        t0 = time.time()

        graphs_tmp = [G_new] + graphs_train
        if args.kernel == "wl":
            K_tmp = wl_gram(graphs_tmp, n_iter=args.wl_n_iter, normalize=True)
        else:
            K_tmp = sp_gram(graphs_tmp, normalize=True)

        k_new_train = K_tmp[0, 1:].reshape(1, -1)

        if args.model == "krr":
            _ = float(k_new_train @ alpha)
        else:
            _ = float(model.predict(k_new_train)[0])

        times.append(time.time() - t0)

    out = {
        "tag": args.tag,
        "selected_tau": selected_tau,
        "selected_tau_value": _parse_tau_to_float(selected_tau),
        "kernel": args.kernel,
        "model": args.model,
        "n_bench": len(times),
        "mean_infer_sec": float(np.mean(times)),
        "median_infer_sec": float(np.median(times)),
        "p95_infer_sec": float(np.quantile(times, 0.95)),
    }

    out_path = Path(args.results_dir) / "fixed" / args.tag / f"inference_{args.kernel}_{args.model}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Wrote:", out_path)
    print(out)


if __name__ == "__main__":
    main()