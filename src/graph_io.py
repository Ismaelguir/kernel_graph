from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from grakel import Graph


def load_index_to_ticker(tickers_json: str | Path = "data/processed/tickers.json") -> Dict[int, str]:
    """
    Charge le mapping index -> ticker (utile pour connaître N et, plus tard, labels éventuels).
    """
    tickers_json = Path(tickers_json)
    with open(tickers_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {int(k): v for k, v in meta["index_to_ticker"].items()}


def load_edgelist(date: str, graphs_dir: str | Path = "data/graphs") -> List[Tuple[int, int]]:
    """
    Lit data/graphs/YYYY-MM-DD.csv et retourne la liste des arêtes (i,j) en ignorant le poids w.
    """
    graphs_dir = Path(graphs_dir)
    path = graphs_dir / f"{date}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Graphe introuvable: {path}")

    df = pd.read_csv(path)
    if not {"i", "j"}.issubset(df.columns):
        raise ValueError(f"Colonnes attendues i,j (et éventuellement w) absentes dans {path}")

    edges: List[Tuple[int, int]] = []
    for i, j in df[["i", "j"]].itertuples(index=False):
        ii, jj = int(i), int(j)
        if ii == jj:
            continue
        if ii > jj:
            ii, jj = jj, ii
        edges.append((ii, jj))

    # dédoublonnage
    edges = sorted(set(edges))
    return edges


def to_grakel_graph(
    date: str,
    graphs_dir: str | Path = "data/graphs",
    n_nodes: Optional[int] = None,
    tickers_json: str | Path = "data/processed/tickers.json",
) -> Graph:
    """
    Convertit un graphe stocké en edge-list CSV vers un objet GraKeL Graph.

    Choix méthodo (fixé) :
    - on ignore les poids w et on travaille sur la structure binaire
    - labels de nœuds constants (tous identiques) pour comparer la topologie
    - on inclut explicitement les nœuds isolés (sinon GraKeL peut les perdre)
    """
    if n_nodes is None:
        idx_to_ticker = load_index_to_ticker(tickers_json)
        n_nodes = len(idx_to_ticker)

    edges = load_edgelist(date, graphs_dir)

    # adjacency dict: node -> {neighbor: 1, ...}
    adj: Dict[int, Dict[int, int]] = {i: {} for i in range(n_nodes)}
    for i, j in edges:
        adj[i][j] = 1
        adj[j][i] = 1

    # labels constants (topologie seule)
    node_labels: Dict[int, str] = {i: "1" for i in range(n_nodes)}

    return Graph(adj, node_labels=node_labels)


def load_grakel_graphs(
    dates: List[str],
    graphs_dir: str | Path = "data/graphs",
    tickers_json: str | Path = "data/processed/tickers.json",
) -> List[Graph]:
    """
    Batch loader : liste de dates -> liste de Graph GraKeL.
    """
    idx_to_ticker = load_index_to_ticker(tickers_json)
    n_nodes = len(idx_to_ticker)
    return [to_grakel_graph(d, graphs_dir=graphs_dir, n_nodes=n_nodes, tickers_json=tickers_json) for d in dates]