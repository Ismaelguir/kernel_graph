from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .splits import fixed_split
from .graph_io import load_grakel_graphs
from .kernels.wl import wl_gram
from .kernels.shortest_path import sp_gram


def main() -> None:
    t0 = time.time()
    labels = pd.read_csv("data/processed/labels.csv")
    labels["date"] = pd.to_datetime(labels["date"]).dt.date.astype(str)
    labels = labels.sort_values("date").reset_index(drop=True)

    train_idx, val_idx, _ = fixed_split(labels, "2019-12-31", "2020-12-31")
    train_dates = labels.loc[train_idx, "date"].to_list()
    val_dates = labels.loc[val_idx, "date"].to_list()

    # subset quick
    n_tr, n_va = 200, 80
    dates = train_dates[:n_tr] + val_dates[:n_va]

    graphs = load_grakel_graphs(dates)

    # WL
    K_wl = wl_gram(graphs, n_iter=3, normalize=True)
    # SP
    K_sp = sp_gram(graphs, normalize=True)

    out = Path("results/quick")
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "K_wl.npy", K_wl)
    np.save(out / "K_sp.npy", K_sp)

    stats = {
        "n_graphs": len(graphs),
        "wl_shape": list(K_wl.shape),
        "sp_shape": list(K_sp.shape),
        "wl_sym_err": float(np.abs(K_wl - K_wl.T).max()),
        "sp_sym_err": float(np.abs(K_sp - K_sp.T).max()),
        "wl_min": float(K_wl.min()),
        "wl_max": float(K_wl.max()),
        "sp_min": float(K_sp.min()),
        "sp_max": float(K_sp.max()),
        "elapsed_sec": float(time.time() - t0),
    }
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("OK quick gram ->", out)
    print(stats)


if __name__ == "__main__":
    main()