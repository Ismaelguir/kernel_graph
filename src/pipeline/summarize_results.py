from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def _run_id(p: Path) -> int:
    try:
        return int(p.name.replace("run_", ""))
    except Exception:
        return -1


def _write_md_table(df: pd.DataFrame, path: Path) -> None:
    cols=list(df.columns)
    lines=[]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results_final")
    args = p.parse_args()

    base = Path(args.results_dir) / "fixed"
    if not base.exists():
        raise FileNotFoundError("missing results/fixed")

    # for each tag, build a table
    tag_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not tag_dirs:
        raise RuntimeError("no tag dirs under results/fixed (expected results/fixed/<tag>/...)")

    global_tables = []

    for tag_dir in tag_dirs:
        tag = tag_dir.name
        rows = []

        # results/fixed/<tag>/<kernel>/<model>/run_xxxx/
        for kernel_dir in sorted([p for p in tag_dir.iterdir() if p.is_dir()]):
            kernel = kernel_dir.name
            for model_dir in sorted([p for p in kernel_dir.iterdir() if p.is_dir()]):
                model = model_dir.name
                for run_dir in sorted([p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]):
                    cfg_p = run_dir / "config.json"
                    met_p = run_dir / "metrics.json"
                    tim_p = run_dir / "timings.json"
                    if not (cfg_p.exists() and met_p.exists() and tim_p.exists()):
                        continue

                    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
                    met = json.loads(met_p.read_text(encoding="utf-8"))
                    tim = json.loads(tim_p.read_text(encoding="utf-8"))

                    best_params = met.get("best_params", {}) or {}
                    sizes = met.get("sizes", {}) or {}

                    rows.append({
                        "tag": tag,
                        "kernel": kernel,
                        "model": model,
                        "run_id": _run_id(run_dir),
                        "run_dir": str(run_dir),
                        "train_end": cfg.get("train_end", ""),
                        "val_end": cfg.get("val_end", ""),
                        "selected_tau": met.get("selected_tau", best_params.get("tau", "")),
                        "best_lambda": best_params.get("lambda", ""),
                        "best_C": best_params.get("C", ""),
                        "best_epsilon": best_params.get("epsilon", ""),
                        "val_mse": met.get("val_mse", None),
                        "test_mse": met.get("test_mse", None),
                        "test_mae": met.get("test_mae", None),
                        "test_r2": met.get("test_r2", None),
                        "n_support_vectors": met.get("n_support_vectors", None),
                        "frac_support_vectors": met.get("frac_support_vectors", None),
                        "load_graphs_sec": tim.get("load_graphs_sec", None),
                        "gram_sec": tim.get("gram_sec", None),
                        "fit_sec": tim.get("fit_sec", None),
                        "tau_search_total_sec": tim.get("tau_search_total_sec", None),
                        "total_sec": tim.get("total_sec", None),
                        "n_train": sizes.get("train", None),
                        "n_val": sizes.get("val", None),
                        "n_test": sizes.get("test", None),
                    })

        if not rows:
            # pas de runs pour ce tag -> on skip
            continue

        df = pd.DataFrame(rows)
        df = df.sort_values(["kernel", "model", "run_id"]).reset_index(drop=True)
        last = df.groupby(["kernel", "model"], as_index=False).tail(1).copy()

        def fmt_params(r):
            if r["model"] == "krr":
                return f"lambda={r['best_lambda']}"
            if r["model"] == "svr":
                return f"C={r['best_C']}, eps={r['best_epsilon']}"
            if r["model"] == "ridge":
                return f"lambda={r['best_lambda']}"
            return ""
        
        last["best_params_str"] = last.apply(fmt_params, axis=1)

        # Load inference timing per (kernel, model) if available
        infer_map: dict[tuple[str, str], dict] = {}
        for kernel_dir in tag_dir.iterdir():
            if not kernel_dir.is_dir():
                continue
            kernel = kernel_dir.name
            for model_dir in kernel_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                infer_path = tag_dir / f"inference_{kernel}_{model}.json"
                if infer_path.exists():
                    infer_map[(kernel, model)] = json.loads(infer_path.read_text(encoding="utf-8"))

        last["mean_infer_sec"] = last.apply(
            lambda r: infer_map.get((r["kernel"], r["model"]), {}).get("mean_infer_sec"), axis=1
        )
        last["median_infer_sec"] = last.apply(
            lambda r: infer_map.get((r["kernel"], r["model"]), {}).get("median_infer_sec"), axis=1
        )
        last["p95_infer_sec"] = last.apply(
            lambda r: infer_map.get((r["kernel"], r["model"]), {}).get("p95_infer_sec"), axis=1
        )

        out = last[[
            "tag","kernel","model","best_params_str",
            "selected_tau",
            "val_mse","test_mse","test_mae","test_r2",
            "n_support_vectors","frac_support_vectors",
            "load_graphs_sec","gram_sec","fit_sec","total_sec",
            "tau_search_total_sec",
            "mean_infer_sec","median_infer_sec","p95_infer_sec",
            "n_train","n_val","n_test",
            "run_dir",
        ]].sort_values(["kernel","model"]).reset_index(drop=True)

        csv_path = tag_dir / "table.csv"
        md_path = tag_dir / "table.md"
        out.to_csv(csv_path, index=False)
        _write_md_table(out.drop(columns=["run_dir"]), md_path)
        global_tables.append(out.copy())

        print(f"[{tag}] wrote {csv_path} and {md_path}")

    if not global_tables:
        return

    all_df = pd.concat(global_tables, ignore_index=True)
    all_df = all_df.sort_values(["tag", "kernel", "model"]).reset_index(drop=True)
    all_csv = base / "all_taus.csv"
    all_md = base / "all_taus.md"
    all_df.to_csv(all_csv, index=False)
    _write_md_table(all_df.drop(columns=["run_dir"]), all_md)

    best = (
        all_df.sort_values(["kernel", "model", "val_mse"])
        .groupby(["kernel", "model"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_csv = base / "best_by_val.csv"
    best_md = base / "best_by_val.md"
    best.to_csv(best_csv, index=False)
    _write_md_table(best.drop(columns=["run_dir"]), best_md)

    fig_df = all_df.copy()
    fig_df["selected_tau_num"] = pd.to_numeric(
        fig_df["selected_tau"].astype(str).str.replace("tau_", "", regex=False),
        errors="coerce",
    )
    fig_df = fig_df.dropna(subset=["selected_tau_num"])
    fig_dir = base / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not fig_df.empty:
        plt.figure()
        for (kernel, model), g in fig_df.groupby(["kernel", "model"]):
            g = g.sort_values("selected_tau_num")
            plt.plot(g["selected_tau_num"], g["test_r2"], marker="o", label=f"{kernel}-{model}")
        plt.xlabel("selected_tau")
        plt.ylabel("test_r2")
        plt.title("Test R2 vs selected tau")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "r2_vs_tau.png", dpi=150)
        plt.close()

        plt.figure()
        for (kernel, model), g in fig_df.groupby(["kernel", "model"]):
            g = g.sort_values("selected_tau_num")
            plt.plot(g["selected_tau_num"], g["test_mse"], marker="o", label=f"{kernel}-{model}")
        plt.xlabel("selected_tau")
        plt.ylabel("test_mse")
        plt.title("Test MSE vs selected tau")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "mse_vs_tau.png", dpi=150)
        plt.close()

        plt.figure()
        for (kernel, model), g in fig_df.groupby(["kernel", "model"]):
            g = g.sort_values("selected_tau_num")
            plt.plot(g["selected_tau_num"], g["gram_sec"], marker="o", label=f"{kernel}-{model}")
        plt.xlabel("selected_tau")
        plt.ylabel("gram_sec")
        plt.title("Gram time vs selected tau")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "gram_sec_vs_tau.png", dpi=150)
        plt.close()

    print(f"[global] wrote {all_csv} and {all_md}")
    print(f"[global] wrote {best_csv} and {best_md}")


if __name__ == "__main__":
    main()