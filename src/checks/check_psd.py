from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from ..core.splits import fixed_split
from ..data.graph_io import load_grakel_graphs
from ..kernels.wl import wl_gram
from ..kernels.shortest_path import sp_gram


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--tag",required=True)               # ex: tau_0.40
    p.add_argument("--kernel",choices=["wl","sp"],required=True)
    p.add_argument("--train_end",default="2019-12-31")
    p.add_argument("--val_end",default="2020-12-31")
    p.add_argument("--labels_path",required=True)
    p.add_argument("--graphs_dir",required=True)
    p.add_argument("--wl_n_iter",type=int,default=3)
    p.add_argument("--eps",type=float,default=1e-8)
    args=p.parse_args()

    labels=pd.read_csv(args.labels_path)
    labels["date"]=pd.to_datetime(labels["date"]).dt.date.astype(str)
    labels=labels.sort_values("date").reset_index(drop=True)

    train_idx,_,_=fixed_split(labels,args.train_end,args.val_end)
    train_dates=labels.loc[train_idx,"date"].to_list()

    graphs=load_grakel_graphs(train_dates,graphs_dir=args.graphs_dir)

    if args.kernel=="wl":
        K=wl_gram(graphs,n_iter=args.wl_n_iter,normalize=True)
    else:
        K=sp_gram(graphs,normalize=True)

    # sym check
    sym_err=float(np.max(np.abs(K-K.T)))

    # eigenvalues (K is symmetric)
    evals=np.linalg.eigvalsh(K)
    lam_min=float(evals[0])
    neg_mass=float(np.mean(evals < -args.eps))
    most_neg=float(np.min(evals))

    print("tag:",args.tag)
    print("kernel:",args.kernel)
    print("K shape:",K.shape)
    print("sym_err:",sym_err)
    print("lambda_min:",lam_min)
    print("frac_eigs_below_-eps:",neg_mass)
    print("most_negative:",most_neg)
    print("PSD_numeric:", "YES" if lam_min >= -args.eps else "NO")


if __name__=="__main__":
    main()