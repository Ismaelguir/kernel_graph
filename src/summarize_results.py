from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


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
    base = Path("results/fixed")
    if not base.exists():
        raise FileNotFoundError("missing results/fixed")

    # for each tag, build a table
    tag_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not tag_dirs:
        raise RuntimeError("no tag dirs under results/fixed (expected results/fixed/<tag>/...)")

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
                        "best_lambda": best_params.get("lambda", ""),
                        "best_C": best_params.get("C", ""),
                        "best_epsilon": best_params.get("epsilon", ""),
                        "val_mse": met.get("val_mse", None),
                        "test_mse": met.get("test_mse", None),
                        "test_mae": met.get("test_mae", None),
                        "test_r2": met.get("test_r2", None),
                        "load_graphs_sec": tim.get("load_graphs_sec", None),
                        "gram_sec": tim.get("gram_sec", None),
                        "fit_sec": tim.get("fit_sec", None),
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

        out = last[[
            "tag","kernel","model","best_params_str",
            "val_mse","test_mse","test_mae","test_r2",
            "load_graphs_sec","gram_sec","fit_sec","total_sec",
            "n_train","n_val","n_test",
            "run_dir",
        ]].sort_values(["kernel","model"]).reset_index(drop=True)

        csv_path = tag_dir / "table.csv"
        md_path = tag_dir / "table.md"
        out.to_csv(csv_path, index=False)
        _write_md_table(out.drop(columns=["run_dir"]), md_path)

        print(f"[{tag}] wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    main()