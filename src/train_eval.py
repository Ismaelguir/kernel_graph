from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

from .splits import fixed_split


@dataclass(frozen=True)
class RunConfig:
    mode: str = "fixed"
    kernel: str = "wl"  # placeholder, on implémentera après
    train_end: str = "2019-12-31"
    val_end: str = "2020-12-31"

    labels_path: str = "data/processed/labels.csv"
    graphs_dir: str = "data/graphs"
    results_dir: str = "results"


def _next_run_dir(base: Path) -> Path:
    """
    Crée un dossier run_XXXX avec incrément automatique.
    """
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


def _assert_graphs_exist(dates: List[str], graphs_dir: Path) -> Tuple[int, List[str]]:
    """
    Vérifie que chaque date a un fichier graph CSV correspondant.
    Retourne (nb_ok, missing_files).
    """
    missing = []
    ok = 0
    for dt in dates:
        p = graphs_dir / f"{dt}.csv"
        if p.exists():
            ok += 1
        else:
            missing.append(str(p))
    return ok, missing


def main() -> None:
    t0 = time.time()
    cfg = RunConfig()

    labels_path = Path(cfg.labels_path)
    graphs_dir = Path(cfg.graphs_dir)

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.csv introuvable: {labels_path}")
    if not graphs_dir.exists():
        raise FileNotFoundError(f"dossier graphs introuvable: {graphs_dir}")

    labels = pd.read_csv(labels_path)
    # garde un ordre strictement chronologique
    labels["date"] = pd.to_datetime(labels["date"]).dt.date.astype(str)
    labels = labels.sort_values("date").reset_index(drop=True)

    train_idx, val_idx, test_idx = fixed_split(labels, cfg.train_end, cfg.val_end)

    # Prépare dossier results
    base = Path(cfg.results_dir) / cfg.mode / cfg.kernel
    run_dir = _next_run_dir(base)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Sauvegarde config + split sizes
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    split_sizes = {
        "n_total": int(len(labels)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "train_start": labels.loc[train_idx[0], "date"],
        "train_end": labels.loc[train_idx[-1], "date"],
        "val_start": labels.loc[val_idx[0], "date"],
        "val_end": labels.loc[val_idx[-1], "date"],
        "test_start": labels.loc[test_idx[0], "date"],
        "test_end": labels.loc[test_idx[-1], "date"],
    }
    with open(run_dir / "split_sizes.json", "w", encoding="utf-8") as f:
        json.dump(split_sizes, f, indent=2)

    # Vérifie la présence des graphes
    train_dates = labels.loc[train_idx, "date"].to_list()
    val_dates = labels.loc[val_idx, "date"].to_list()
    test_dates = labels.loc[test_idx, "date"].to_list()

    ok_tr, miss_tr = _assert_graphs_exist(train_dates, graphs_dir)
    ok_va, miss_va = _assert_graphs_exist(val_dates, graphs_dir)
    ok_te, miss_te = _assert_graphs_exist(test_dates, graphs_dir)

    graph_check = {
        "train_ok": ok_tr, "train_missing": len(miss_tr),
        "val_ok": ok_va, "val_missing": len(miss_va),
        "test_ok": ok_te, "test_missing": len(miss_te),
        "missing_examples": (miss_tr + miss_va + miss_te)[:10],
    }
    with open(run_dir / "graph_check.json", "w", encoding="utf-8") as f:
        json.dump(graph_check, f, indent=2)

    if graph_check["train_missing"] or graph_check["val_missing"] or graph_check["test_missing"]:
        raise RuntimeError(
            "Il manque des fichiers de graphes pour certaines dates. "
            f"Voir {run_dir/'graph_check.json'}"
        )

    timings = {
        "elapsed_sec": float(time.time() - t0),
    }
    with open(run_dir / "timings.json", "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)

    print("OK")
    print("Run dir:", run_dir)
    print("Split sizes:", split_sizes)
    print("Graph check:", graph_check)


if __name__ == "__main__":
    main()