from __future__ import annotations

"""
Construction de la base de données: (G_t, y_t)
- G_t : graphe de dépendance via corrélation sur CORR_WINDOW jours
- y_t : rendement futur équipondéré sur FWD_HORIZON jours

Sorties (par tau):
- data/processed/tau_0.40/labels.csv : date_end_window, y, num_edges
- data/graphs/tau_0.40/YYYY-MM-DD.csv : edge-list i,j,w (indices 0..N-1)
- data/processed/tau_0.40/tickers.json : mapping indices -> ticker + paramètres
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    START_DATE, END_DATE, CORR_WINDOW, FWD_HORIZON, ABS_CORR, TAU,
    TICKERS, YAHOO_TICKER_MAP, PATHS
)
from .utils import (
    download_adj_close, align_prices_intersection, log_returns,
    corr_matrix, threshold_edges, portfolio_forward_return, map_tickers, ensure_dir
)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--tau",type=float,default=TAU)
    args=parser.parse_args()
    tau=float(args.tau)

    root=Path(PATHS.project_root)
    raw_dir=root/PATHS.raw_dir
    processed_dir=(root/PATHS.processed_dir)/f"tau_{tau:.2f}"
    graphs_dir=(root/PATHS.graphs_dir)/f"tau_{tau:.2f}"
    ensure_dir(raw_dir)
    ensure_dir(processed_dir)
    ensure_dir(graphs_dir)

    # 1) Tickers Yahoo
    yahoo_tickers=map_tickers(TICKERS,YAHOO_TICKER_MAP)

    # 2) Download prix ajustés
    raw_prices_path=raw_dir/f"adj_close_{START_DATE}_{END_DATE}.csv"
    prices=download_adj_close(
        tickers=yahoo_tickers,
        start=START_DATE,
        end=END_DATE,
        out_csv=raw_prices_path,
    )

    # 3) Alignement dates (intersection stricte)
    prices=align_prices_intersection(prices)

    # 4) Rendements log
    rets=log_returns(prices)
    dates=rets.index.to_list()

    # 5) Vérifs simples
    missing=[t for t in yahoo_tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(
            f"Tickers manquants après download: {missing}. "
            "Relance après fix réseau / retries. "
            "Ne pas continuer avec un univers incomplet."
        )

    # 6) Export mapping indices -> ticker (+ params)
    tickers_json=processed_dir/"tickers.json"
    with open(tickers_json,"w",encoding="utf-8") as f:
        json.dump(
            {"index_to_ticker":{str(i):t for i,t in enumerate(yahoo_tickers)},
             "params":{
                 "start":START_DATE,"end":END_DATE,
                 "corr_window":CORR_WINDOW,"fwd_horizon":FWD_HORIZON,
                 "tau":tau,"abs_corr":ABS_CORR,
             }},
            f,
            indent=2,
        )

    # 7) Boucle temporelle: pour chaque t, construire G_t sur fenêtre passée et y_t sur horizon futur
    R=rets.to_numpy(dtype=float)
    T=R.shape[0]

    t_min=CORR_WINDOW-1
    t_max=T-FWD_HORIZON-1

    labels=[]
    for t in tqdm(range(t_min,t_max+1),desc="Building graphs"):
        past=R[t-CORR_WINDOW+1:t+1,:]
        future=R[t+1:t+1+FWD_HORIZON,:]

        C=corr_matrix(past)
        edges=threshold_edges(C,tau=tau,abs_corr=ABS_CORR)
        y=portfolio_forward_return(future)

        date_t=dates[t].date().isoformat()

        gpath=graphs_dir/f"{date_t}.csv"
        pd.DataFrame(edges,columns=["i","j","w"]).to_csv(gpath,index=False)

        labels.append((date_t,y,len(edges)))

    # 8) Export labels
    labels_df=pd.DataFrame(labels,columns=["date","y","num_edges"])
    labels_path=processed_dir/"labels.csv"
    labels_df.to_csv(labels_path,index=False)

    # 9) Résumé
    print(f"OK: {len(labels_df)} graphes construits")
    print(f"Labels: {labels_path}")
    print(f"Graphs: {graphs_dir} (un fichier CSV par date)")
    print(f"Raw prices: {raw_prices_path}")
    print(f"Tickers map: {tickers_json}")


if __name__=="__main__":
    main()