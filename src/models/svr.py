# src/models/svr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, List
import numpy as np
from sklearn.svm import SVR


@dataclass(frozen=True)
class SVRParams:
    C: float
    epsilon: float


@dataclass(frozen=True)
class SVRResult:
    params: SVRParams
    model: SVR
    yhat_val: np.ndarray
    yhat_test: np.ndarray


def fit_svr_precomputed(K_train: np.ndarray, y_train: np.ndarray, C: float, epsilon: float) -> SVR:
    """
    SVR avec kernel='precomputed': fit sur K_train (n_train,n_train).
    """
    m = SVR(kernel="precomputed", C=float(C), epsilon=float(epsilon))
    m.fit(K_train, y_train)
    return m


def predict_svr_precomputed(model: SVR, K_x_train: np.ndarray) -> np.ndarray:
    """
    Predict sur une matrice (n_x, n_train) de similarités entre x et train.
    """
    return model.predict(K_x_train)


def select_params_svr(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_val_train: np.ndarray,
    y_val: np.ndarray,
    C_grid: Iterable[float],
    eps_grid: Iterable[float],
    metric_fn,
) -> Tuple[SVRParams, Dict[str, float]]:
    """
    Grid-search simple sur (C, epsilon), score sur val via metric_fn (plus petit = meilleur).
    Retourne (best_params, scores) où scores map "C=<..>,eps=<..>" -> score.
    """
    scores: Dict[str, float] = {}
    best = None
    best_score = None

    for C in C_grid:
        for eps in eps_grid:
            m = fit_svr_precomputed(K_train, y_train, C=C, epsilon=eps)
            yhat = predict_svr_precomputed(m, K_val_train)
            score = float(metric_fn(y_val, yhat))
            key = f"C={float(C):g},eps={float(eps):g}"
            scores[key] = score
            if (best_score is None) or (score < best_score):
                best_score = score
                best = SVRParams(float(C), float(eps))

    assert best is not None
    return best, scores