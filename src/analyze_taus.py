from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def _read_table(tag_dir: Path) -> pd.DataFrame:
    csv_path = tag_dir / "table.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # tag déjà dans la table (si tu as gardé la colonne), sinon on l’ajoute
    if "tag" not in df.columns:
        df["tag"] = tag_dir.name
    return df


def main() -> None:
    base = Path("results/fixed")
    tag_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("tau_")])
    if not tag_dirs:
        raise RuntimeError("no tau_* dirs under results/fixed (expected tau_0.30, tau_0.40, ...)")

    tables = []
    for td in tag_dirs:
        df = _read_table(td)
        if not df.empty:
            tables.append(df)

    if not tables:
        raise RuntimeError("no table.csv found under tau_* dirs")

    all_df = pd.concat(tables, ignore_index=True)

    # extraire tau numérique depuis "tau_0.30"
    all_df["tau"] = all_df["tag"].astype(str).str.replace("tau_", "", regex=False).astype(float)

    out_csv = base / "all_taus.csv"
    out_md = base / "all_taus.md"
    all_df.sort_values(["tau", "kernel", "model"]).to_csv(out_csv, index=False)

    # markdown sans tabulate
    md_df = all_df.sort_values(["tau", "kernel", "model"]).copy()
    cols = list(md_df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in md_df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # figures
    fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Perf vs tau (R2)
    plt.figure()
    for (kernel, model), g in all_df.groupby(["kernel", "model"]):
        g = g.sort_values("tau")
        plt.plot(g["tau"], g["test_r2"], marker="o", label=f"{kernel}-{model}")
    plt.xlabel("tau")
    plt.ylabel("test_r2")
    plt.title("Test R2 vs tau")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "r2_vs_tau.png", dpi=150)
    plt.close()

    # MSE vs tau
    plt.figure()
    for (kernel, model), g in all_df.groupby(["kernel", "model"]):
        g = g.sort_values("tau")
        plt.plot(g["tau"], g["test_mse"], marker="o", label=f"{kernel}-{model}")
    plt.xlabel("tau")
    plt.ylabel("test_mse")
    plt.title("Test MSE vs tau")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "mse_vs_tau.png", dpi=150)
    plt.close()

    # Cost vs tau (Gram time)
    plt.figure()
    for (kernel, model), g in all_df.groupby(["kernel", "model"]):
        g = g.sort_values("tau")
        plt.plot(g["tau"], g["gram_sec"], marker="o", label=f"{kernel}-{model}")
    plt.xlabel("tau")
    plt.ylabel("gram_sec")
    plt.title("Gram time vs tau")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "gram_sec_vs_tau.png", dpi=150)
    plt.close()

    print("Wrote:")
    print(" -", out_csv)
    print(" -", out_md)
    print(" -", fig_dir / "r2_vs_tau.png")
    print(" -", fig_dir / "mse_vs_tau.png")
    print(" -", fig_dir / "gram_sec_vs_tau.png")


if __name__ == "__main__":
    main()