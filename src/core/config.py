"""
Configuration centralisée du projet.
Tout ce qui change rarement doit être ici (tickers, dates, fenêtres, seuil, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# Période d'étude (YYYY-MM-DD)
START_DATE: str = "2014-01-01"
END_DATE: str = "2024-12-31"

# Fenêtres temporelles (en jours de bourse, via index des rendements)
CORR_WINDOW: int = 60   # 60 jours passés pour construire le graphe
FWD_HORIZON: int = 20   # 20 jours futurs pour la cible y_t (rendement futur)

# Règle d'arêtes : seuillage sur |corr|
ABS_CORR: bool = True
TAU: float = 0.40  # seuil sur |corr| (à ajuster après inspection de densité)

# Tickers : univers fixe (60)
TICKERS: List[str] = [
    # Tech / semi / software
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","AVGO","ORCL","AMD","QCOM","CSCO","INTU","ADBE","CRM","TXN",
    # Communication / media / telco
    "NFLX","DIS","CMCSA","VZ","TMUS",
    # Consommation
    "HD","NKE","MCD","SBUX","LOW","WMT","COST","PG","KO","PEP","PM",
    # Santé
    "JNJ","PFE","MRK","ABBV","UNH","TMO","AMGN","GILD",
    # Finance
    "JPM","BAC","WFC","GS","MS","BRK.B","V","MA","AXP",
    # Industriels / défense / transport
    "CAT","GE","HON","UPS","UNP","LMT","DE",
    # Énergie / matériaux / utilities
    "XOM","CVX","LIN","NEE",
]

# Mapping Yahoo Finance : certains tickers utilisent '-' plutôt que '.'
# NB: yfinance/Yahoo utilisent souvent BRK-B au lieu de BRK.B
YAHOO_TICKER_MAP: Dict[str, str] = {
    "BRK.B": "BRK-B",
}

@dataclass(frozen=True)
class Paths:
    project_root: str = "."
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    graphs_dir: str = "data/graphs"

PATHS = Paths()