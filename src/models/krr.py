from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, List
import numpy as np


@dataclass(frozen=True)
class KRRResult:
    lam: float
    alpha: np.ndarray  # (n_train,)
    yhat_val: np.ndarray
    yhat_test: np.ndarray


def fit_krr(K_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    """
    KRR: alpha = (K + lam I)^-1 y
    K_train: (n_train, n_train), y_train: (n_train,)
    """
    n = K_train.shape[0]
    A = K_train + lam * np.eye(n)
    alpha = np.linalg.solve(A, y_train)
    return alpha


def predict_krr(K_x_train: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    yhat = K(x, train) alpha
    K_x_train: (n_x, n_train)
    """
    return K_x_train @ alpha


def select_lambda_krr(
    K_train: np.ndarray,
    y_train: np.ndarray,
    K_val_train: np.ndarray,
    y_val: np.ndarray,
    lambdas: Iterable[float],
    metric_fn,
) -> Tuple[float, Dict[float, float]]:
    """
    Choisit lambda par validation. metric_fn(y_true, y_pred) -> float (plus petit = meilleur).
    Retourne (best_lambda, scores_by_lambda).
    """
    scores: Dict[float, float] = {}
    best_lam = None
    best_score = None

    for lam in lambdas:
        alpha = fit_krr(K_train, y_train, lam)
        yhat = predict_krr(K_val_train, alpha)
        score = float(metric_fn(y_val, yhat))
        scores[float(lam)] = score
        if (best_score is None) or (score < best_score):
            best_score = score
            best_lam = float(lam)

    assert best_lam is not None
    return best_lam, scores