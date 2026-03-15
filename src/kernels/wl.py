from __future__ import annotations

from typing import List
import numpy as np
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from grakel import Graph


def wl_gram(graphs: List[Graph], n_iter: int = 3, normalize: bool = True) -> np.ndarray:
    """
    Matrice de Gram WL (via GraKeL).
    On compose WL avec VertexHistogram, qui est le choix standard.
    """
    kernel = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=normalize)
    K = kernel.fit_transform(graphs)  # (n, n)
    return np.asarray(K)