from pathlib import Path
import pandas as pd
import networkx as nx
import json

GRAPHS_DIR = Path("data/graphs")
TICKERS_JSON = Path("data/processed/tickers.json")
OUT_DIR = Path("data/processed/graphml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE = "2014-11-10"  

# 1) charge mapping index -> ticker (optionnel mais utile pour labels de noeuds)
with open(TICKERS_JSON, "r", encoding="utf-8") as f:
    meta = json.load(f)
idx_to_ticker = {int(k): v for k, v in meta["index_to_ticker"].items()}

# 2) charge edge-list
edge_path = GRAPHS_DIR / f"{DATE}.csv"
df = pd.read_csv(edge_path)

# 3) construit graphe
G = nx.Graph()
# ajoute les noeuds (pour assurer 60 noeuds même si isolés)
for i, tkr in idx_to_ticker.items():
    # use the ticker string itself as the node id so GraphML shows the ticker (e.g. "SPY")
    G.add_node(tkr, ticker=tkr)

# ajoute les arêtes pondérées
for i, j, w in df[["i", "j", "w"]].itertuples(index=False):
    # map numeric indices back to ticker strings and keep the weight (correlation)
    ti = idx_to_ticker.get(int(i))
    tj = idx_to_ticker.get(int(j))
    if ti is None or tj is None:
        # skip edges that reference unknown indices
        continue
    G.add_edge(ti, tj, weight=float(w))

# 4) export graphml
out_path = OUT_DIR / f"{DATE}.graphml"
nx.write_graphml(G, out_path)
print("Wrote:", out_path)