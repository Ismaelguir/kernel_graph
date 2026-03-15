from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import pandas as pd


@dataclass(frozen=True)
class FixedSplitConfig:
    train_end: str = "2019-12-31"
    val_end: str = "2020-12-31"


def fixed_split(
    labels: pd.DataFrame,
    train_end: str = "2019-12-31",
    val_end: str = "2020-12-31",
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split chronologique (pas de shuffle) sur la colonne 'date' au format YYYY-MM-DD.

    Train: date <= train_end
    Val:   train_end < date <= val_end
    Test:  date > val_end
    """
    if "date" not in labels.columns:
        raise ValueError("labels doit contenir une colonne 'date'.")

    d = pd.to_datetime(labels["date"])
    train_mask = d <= pd.to_datetime(train_end)
    val_mask = (d > pd.to_datetime(train_end)) & (d <= pd.to_datetime(val_end))
    test_mask = d > pd.to_datetime(val_end)

    train_idx = labels.index[train_mask].to_list()
    val_idx = labels.index[val_mask].to_list()
    test_idx = labels.index[test_mask].to_list()

    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError(
            f"Split vide: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}. "
            "Vérifie les dates de coupure et la période couverte par labels.csv."
        )

    # garde l'ordre chronologique (index déjà ordonné si labels l'est)
    return train_idx, val_idx, test_idx