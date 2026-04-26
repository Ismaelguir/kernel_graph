from __future__ import annotations

import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd


def max_edges_undirected(n: int) -> int:
    return n * (n - 1) // 2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_path", required=True)
    p.add_argument("--tickers_path", required=True)
    p.add_argument("--eps", type=float, default=0.0, help="tolérance en nb d'arêtes (0=exact)")
    args = p.parse_args()

    labels_path = Path(args.labels_path)
    tickers_path = Path(args.tickers_path)

    import json

    labels = pd.read_csv(labels_path)

    with open(tickers_path, "r", encoding="utf-8") as f:
        tick = json.load(f)
    n = len(tick["index_to_ticker"])
    mmax = max_edges_undirected(n)

    num_edges = labels["num_edges"].astype(int).to_numpy()
    dens = num_edges / mmax

    print("labels_path:", str(labels_path))
    print("dens_min:", float(dens.min()))
    print("dens_mean:", float(dens.mean()))
    print("dens_max:", float(dens.max()))

    # "quasi-complet" : >= 90% des arêtes possibles
    thr = 0.90
    mask = dens >= thr
    print("frac_dens_ge_0.90:", float(mask.mean()))
    if mask.any():
        ex = labels.loc[mask, "date"].head(5).to_list()
        print("examples_ge_0.90:", ex)


if __name__ == "__main__":
    main()