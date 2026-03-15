from __future__ import annotations

from typing import List
import numpy as np
from grakel.kernels import ShortestPath
from grakel import Graph


def sp_gram(graphs: List[Graph], normalize: bool = True) -> np.ndarray:
    """
    Matrice de Gram Shortest-Path (version non pondérée).
    """
    kernel = ShortestPath(with_labels=True, normalize=normalize)
    K = kernel.fit_transform(graphs)
    return np.asarray(K)