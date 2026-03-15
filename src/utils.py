"""
Fonctions utilitaires pour download, cleaning, rendements, corrélation, graphes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def map_tickers(tickers: List[str], mapping: Dict[str, str]) -> List[str]:
    return [mapping.get(t, t) for t in tickers]


def download_adj_close(
    tickers: List[str],
    start: str,
    end: str,
    out_csv: Optional[Path] = None,
    chunk_size: int = 15,
    max_retries: int = 5,
    sleep_sec: float = 1.5,
) -> pd.DataFrame:
    """
    Télécharge Adj Close via yfinance, en chunks + retries pour éviter les timeouts.
    """
    import time
    import yfinance as yf

    all_adj = []

    for k in range(0, len(tickers), chunk_size):
        chunk = tickers[k:k + chunk_size]

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                data = yf.download(
                    tickers=chunk,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    actions=False,
                    group_by="column",
                    threads=False,  # IMPORTANT: réduit les timeouts
                )
                if data.empty:
                    raise RuntimeError("Chunk download returned empty dataframe.")

                if isinstance(data.columns, pd.MultiIndex):
                    adj = data["Adj Close"].copy()
                else:
                    adj = data.rename("Adj Close").to_frame()

                adj.index = pd.to_datetime(adj.index)
                adj = adj.sort_index()
                adj = adj.dropna(axis=1, how="all")

                all_adj.append(adj)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep_sec * attempt)

        if last_err is not None:
            # On continue, mais on loggue le chunk problématique
            print(f"[WARN] Chunk failed after retries: {chunk}. Error: {last_err}")

    if not all_adj:
        raise RuntimeError("Téléchargement vide (tous les chunks ont échoué).")

    # concat colonne
    adj_all = pd.concat(all_adj, axis=1)
    # si duplicats de colonnes (rare), on garde la première
    adj_all = adj_all.loc[:, ~adj_all.columns.duplicated()]

    if out_csv is not None:
        ensure_dir(out_csv.parent)
        adj_all.to_csv(out_csv)

    return adj_all


def align_prices_intersection(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Conserve l'intersection des dates et supprime les lignes avec NaN.
    Pour des tickers très liquides, c'est généralement suffisant.
    """
    prices = prices.sort_index()
    # Drop dates où il manque au moins un prix
    aligned = prices.dropna(axis=0, how="any")
    return aligned


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Rendements log: r_{t} = log P_t - log P_{t-1}
    """
    lp = np.log(prices)
    rets = lp.diff().dropna()
    return rets


def corr_matrix(window_returns: np.ndarray) -> np.ndarray:
    """
    Corrélation empirique sur une fenêtre: input shape (L, N) -> output (N, N)
    """
    # np.corrcoef attend variables en lignes si rowvar=True; on veut variables=colonnes
    return np.corrcoef(window_returns, rowvar=False)


def threshold_edges(corr: np.ndarray, tau: float, abs_corr: bool = True) -> List[Tuple[int, int, float]]:
    """
    Convertit une matrice de corrélation en edge-list (i,j,w).
    i,j sont des indices 0..N-1. Graphe non orienté : i<j.
    """
    n = corr.shape[0]
    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            val = float(corr[i, j])
            score = abs(val) if abs_corr else val
            if abs_corr:
                if abs(val) >= tau:
                    edges.append((i, j, abs(val)))
            else:
                # si on garde le signe, on seuille sur |corr| mais on stocke corr signée
                if abs(val) >= tau:
                    edges.append((i, j, val))
    return edges


def portfolio_forward_return(rets: np.ndarray) -> float:
    """
    Rendement futur d'un portefeuille équipondéré sur une fenêtre future.
    Input rets shape (H, N) de rendements log.
    y = somme_{u} (1/N) * somme_i r_{u,i}
    """
    return float(rets.mean(axis=1).sum())