# src/plots.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def make_plots(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    pred_path = run_dir / "predictions.csv"
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    df["date"] = pd.to_datetime(df["date"])

    # 1) y vs yhat
    plt.figure()
    plt.scatter(df["y"], df["yhat"], s=10, alpha=0.6)
    plt.xlabel("y")
    plt.ylabel("yhat")
    plt.title("y vs yhat")
    plt.tight_layout()
    plt.savefig(fig_dir / "y_vs_yhat.png", dpi=150)
    plt.close()

    # résidus
    df["resid"] = df["y"] - df["yhat"]

    # 2) résidus dans le temps (plutôt test)
    dft = df[df["split"] == "test"].sort_values("date")
    plt.figure()
    plt.plot(dft["date"], dft["resid"])
    plt.axhline(0)
    plt.xlabel("date")
    plt.ylabel("y - yhat")
    plt.title("Residuals over time (test)")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_time.png", dpi=150)
    plt.close()

    # 3) histogramme résidus (test)
    plt.figure()
    plt.hist(dft["resid"], bins=50)
    plt.xlabel("y - yhat")
    plt.ylabel("count")
    plt.title("Residuals histogram (test)")
    plt.tight_layout()
    plt.savefig(fig_dir / "residuals_hist.png", dpi=150)
    plt.close()